#define F_PROTO "/home/jetson/gait/tumcaffe_deploy.prototxt"
#define F_MODEL "/home/jetson/gait/tumcaffe.caffemodel"
#define F_LABEL "/home/jetson/gait/tumgaidtestids.lst"
#define F_VIDEO "/home/jetson/gait/p003-n05.avi"
#define OF_SCALE 1.0
// float OF_SCALE; /* Para reajustar tamaño OF entrada (reducir complejidad) */
#define LASTFRAME 525 /* Para garantizar 100 Inferencias */
int LAST_FRAME; // Ojo, los arrays suponen maximo 1025 frames!
#define MAX_FRAMES 1025

//#define DEBUG

// TBB
#include "tbb/flow_graph.h"
#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/atomic.h"

#include <omp.h>
#include <thread>

// - Antigos Grafo. TODO: simplificar con arriba
// #include <cuda.h>
// #include <cuda_runtime.h>

#include <caffe/caffe.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>

#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudawarping.hpp> 	// resize
#include <opencv2/cudaarithm.hpp> 	// split
#include <opencv2/cudafilters.hpp> 

// Medir consumo con pmlib y Beaglebone UCM
#include "pmlib.h"
counter_t pmlib_counter;

#include <sys/sysinfo.h>
// #include <math.h>
// #include <iostream>

using namespace std;
using namespace tbb::flow;
using namespace cv;
using namespace cv::cuda;
using namespace caffe;

// Mis funciones de Medidas
// -----
#include "medidas.h"
t_sonda sonda;
// -----
#ifdef DEBUG
	std::vector<string> labels;
#endif

// Token que circula por el grafo, cada nodo escribe su parte, no hay colisiones de escritura
// (solo se replican con el divisor, al llegar a CNN hay dos copias disjuntas)
struct sampleOF {
	unsigned int muestra;
	unsigned int index[6]; // Del ultimo frame del sample
	bool OF_usaGPU[6] = { false, false, true, true, true, true }; // frames1..5 de la muestra usan GPU (primero no usado
	Rect BB[5];
	Mat frames[6]; // frames[0] es el ultimo de sample anterior
				   // frames[1..5] frames de la muestras
	Mat flow[5];   // Veremos si luego copiamos directamente en buffer
				   // Incialmente solo reescalamos a 80x60
};

// ------
// INICIALIZACIONES QUE NO METEMOS EN NODOS (intentar meterlas mas adelante y que estas de afuera queden pocas)

	// BB CPU variables
	// ------
	int minArea = 1000; // No lo uso, pero debe eliminar BB pequeños
	bool updateBGModel = true;
	Ptr<cv::BackgroundSubtractorMOG2> bgmodel_CPU = // (history, varThreshold, detectShadows)
		cv::createBackgroundSubtractorMOG2(40, 30, true);
	Mat strel = getStructuringElement(MORPH_RECT, Size(3,3));

	// OF CPU/GPU variables
	cv::Ptr<cv::FarnebackOpticalFlow> fbOF_CPU = 
			cv::FarnebackOpticalFlow::create(5,sqrt(2)/2.0,false,10,2,7,1.5,cv::OPTFLOW_FARNEBACK_GAUSSIAN);
	cv::Ptr<cv::cuda::FarnebackOpticalFlow> fbOF_GPU =
     		cv::cuda::FarnebackOpticalFlow::create(5,sqrt(2)/2.0,false,13,2,7,1.5,cv::OPTFLOW_FARNEBACK_GAUSSIAN);

	// CNN CPU variables
	boost::shared_ptr<Net<float> > net;
	Blob<float>* input_layer;/// = net->input_blobs()[0];
	Blob<float>* output_layer;// = net->output_blobs()[0];

// Source node: devuelve false cuando no tiene mas que suministrar

struct VideoRead {
private:
	unsigned int muestra;
	unsigned int index;
	bool Param_OF_usaGPU[6] = { false, false, true, true, true, true }; // primero no usado
	VideoCapture stream;
	Mat last;
public:
 
	VideoRead(std::string file_name, int initSched) : stream(file_name), index(0), muestra(0) {

		Mat Img;
		
		if(!(stream.read(Img))) //get one frame from video
		{ cout << "Error. Can't open " << file_name << endl; exit(-1);}
			
		cv::cvtColor(Img, Img, CV_BGR2GRAY);
		last = Img.clone();

		if (initSched>=0) {
			for (int i=0, ini=1; i<5; i++, ini = ini*2)
				{
					Param_OF_usaGPU[i+1] = ((initSched & ini)!=0);
				}	
		}
		
		for (int i=1; i<=5; i++)
			cout << "OF_" << i << " -> " << (Param_OF_usaGPU[i]? "GPU": "CPU") << endl;

		// Inicializar modelo de fondo con el primer frame leido
		bgmodel_CPU->apply(last, Img, updateBGModel ? -1 : 0);
		
	};
	
    ~VideoRead() { stream.release(); };
	
	bool operator()(sampleOF& input) {
		
		Mat GetImg, Img;
		
		if (index>=(LAST_FRAME)) return false; // Solo proceso los LAST_FRAME primeros frames

		if (index==25) { // Ultimo frame muestra anterior, comenzamos con 25 -> La primera CNN es mas lenta
			// cout << "** Comienzan Inferencias frames 21..25" << endl;
			pm_start_counter(&pmlib_counter); // O a entrada OF, no sabemos lo que espera aqui
		}

		input.muestra = ++muestra;
		input.frames[0] = last.clone();
		input.index[0] = index;

_start_sonda(&(sonda.PRE), index+5, true);
// cout << "PRE empieza en " << index+5 << endl;
_start_sonda(&(sonda.VIN), muestra*5+1, false);  
		for (int i=1; i<=5; i++) {

			if(!(stream.read(GetImg))) { //get one frame from video
				stream.set(CAP_PROP_POS_FRAMES, 0);// start (rewind)
				if (!(stream.read(GetImg))) { cout << "No pudo rebobinar " << endl; }
			}
			 			
			cv::cvtColor(GetImg, Img, CV_BGR2GRAY);
	
			index++;
			input.index[i] = index;

			// PLANIFICADOR -> Ahora mismo estatico, en la definicion de VideoRead
			input.OF_usaGPU[i] = Param_OF_usaGPU[i];
			//input.OF_usaGPU[i] = ((index%5)!=0);
			//input.BB_usaGPU = false;
			//input.CN_usaGPU = true;
		
			input.frames[i] = Img.clone();
		}
_stop_sonda(&(sonda.VIN),muestra*5+1); 

		last = Img.clone(); // El ultimo frm de esta muestra el primero sgte

		// cout << "* Leidos frames muestra " << muestra << endl;
	
		return true;		
	}
	
};


struct OFComputationCPU {
  sampleOF operator()(sampleOF input) // le he quitado el const &
	{	
	for (int i=1; i<=5; i++) {
		if (!input.OF_usaGPU[i]) { // Solo proceso frames de CPU
		// cout << "OF-CPU " << input.index[i] << endl;
_start_sonda(&(sonda.OF), input.index[i], false);
		fbOF_CPU->calc(input.frames[i-1], input.frames[i], input.flow[i-1]);
		cv::resize(input.flow[i-1], input.flow[i-1], Size(80,60));
_stop_sonda(&(sonda.OF),input.index[i]);
		}
	}
		
	return input; 
	}
};


struct OFComputationGPU {
  sampleOF operator()(sampleOF input) // le he quitado el const &
{	
	for (int i=1; i<=5; i++) {
		if (input.OF_usaGPU[i]) { // Solo proceso frames de GPU
			// cout << "OF-GPU " << input.index[i] << endl;
			GpuMat prvs_gpu, next_gpu, flow_gpu;
			Mat flow_res;
			prvs_gpu.upload(input.frames[i-1]);
			next_gpu.upload(input.frames[i]);
_start_sonda(&(sonda.OF), input.index[i], true);
			fbOF_GPU->calc(prvs_gpu, next_gpu, flow_gpu);
			flow_gpu.download(flow_res);
			cv::resize(flow_res, input.flow[i-1], Size(80,60)); // Resize en GPU no posible por ser 2 canales
_stop_sonda(&(sonda.OF),input.index[i]);
		}
	}

	return input; 
	}
};


struct BBComputationCPU {
  sampleOF operator()(sampleOF input) // le he quitado el const &
	{	
	// OJO supone ahora mismo todo en CPU
	for (int i=1; i<=5; i++) {
		// cout << "BB-CPU " << input.index[i] << endl;
// continue;
		Mat fgmask, fgimg;
_start_sonda(&(sonda.BB), input.index[i], false);
		bgmodel_CPU->apply(input.frames[i], fgmask, updateBGModel ? -1 : 0); // -1: automatically select the learning rate
		morphologyEx(fgmask, fgimg, MORPH_OPEN, strel);
		input.BB[i-1] = boundingRect(fgimg); // Compute BB
_stop_sonda(&(sonda.BB),input.index[i]);
	}
	
	return input; 

	}
};


struct CNNInference {
private:
// Buffers circulares de 25 fotogramas para almacenar la entrada a la CNN, de desplazan de 5 en 5 (muestras)
	vector<int> BBox_izq; // BBoxes x25 frames
	vector<Mat> flow_vec; // OFx, OFy x25 frames
public:	
sampleOF operator()(tuple<sampleOF,sampleOF> samples) { // le he quitado el const &tuple<int,float>

	sampleOF input  = std::get<0>(samples); // flows de CPU
	sampleOF input1 = std::get<1>(samples); // flows de GPU

	int frame_num = input.index[5]; // Ultimo de la muestra

	Mat flow_x_y[2], flujo_crop, imgROI;
	int izq;

	// cout << "CNN-GPU " << input.muestra << endl;

	// Por ahora suponemos que BB siempre viene de la rama CPU (input)
	// Meter en la posición correspondiente frame_num-5..frame_num
	for (int i=0; i<5; i++) {
		izq = input.BB[i].x/8 + input.BB[i].width/(8*2); // - 30 + 30; // Para la escala de /8 (queda imagen de 60x80; sino, no sirve)
		BBox_izq.push_back(izq);
		if (input.OF_usaGPU[i+1])  // split channels into two different images
			cv::split(input1.flow[i], flow_x_y); // OF Vino de GPU
		else
			cv::split(input.flow[i], flow_x_y);  // OF Vino de CPU
		flow_vec.push_back(flow_x_y[0].clone());
		flow_vec.push_back(flow_x_y[1].clone());

		// Scalar media = mean(flow_x_y[0]);			
		// cout << input.index[i+1] << " Mx " << mean(flow_x_y[0]) << endl;			
		// cout << input.index[i+1] << " My " << mean(flow_x_y[1]) << endl;
		// cout << "BB x: " << input.BB[i].x << " BB width " << input.BB[i].width << endl;  
	}  

	// A partir de tener llemos ya las 25 muestras (frames)
	if (frame_num>=25) {		
		int izquierda = BBox_izq[12]; // Central de los 25 (0-24)
// cout << "* Izquierda " << izquierda << endl;

		Rect ROI = Rect(izquierda, 0, 60, 60);
_start_sonda(&(sonda.CNN), frame_num, true);		
		// Preparar entrada Red (25 muestras OFx OFy)
		// Antes de llamar a CAFE hay que recortar con BB centrado todos los OF
		float *bufferBB = input_layer->mutable_cpu_data();		
		for (int i=0; i<50; i++) { 
			cv::copyMakeBorder(flow_vec[i], flujo_crop,0,0,30,30,BORDER_CONSTANT, Scalar(0.0f));
			imgROI = 10*flujo_crop(ROI); // Escalar el OF x 10 (faltaria sumar media de la BD 0.00...) (necesario el x10?)
			memcpy(bufferBB+(60*60*(i)),imgROI.data,60*60*sizeof(float));
		}
		// Asume siempre en GPU, no hace falta cambiar tras inicialización
		net->Forward(); // Hacer prediccion con Caffe 
			// 21-25, 26-30, ...
_stop_sonda(&(sonda.CNN),frame_num);
_stop_sonda(&(sonda.PRE),frame_num);
// cout << "Fin PRE " << frame_num << endl;

#ifdef DEBUG
		const float* salida = output_layer->cpu_data();
		for (int i=0; i<output_layer->channels(); i++) {
			if (salida[i]>0.01)
				std::cout << "Frame: " << frame_num << " Label: " << labels[i] 
						  << " prob = " << salida[i] << std::endl;
		}
/*		
		vector<float> mi_salida;
		mi_salida.insert (mi_salida.begin(), output_layer->cpu_data(), output_layer->cpu_data()+output_layer->channels());
		int max_idx = arg_max(mi_salida);
		cout << "Maxima CNN " << labels[max_idx] << " valor p = " << salida[max_idx] << endl; 
*/
#endif
		// Borrar los últimos 5 frames para dejar el hueco de la siguiente muestra
		// A partir de tener llemos ya las 25 muestras (frames)
		for (int i=0; i<5; i++ ) // Creo que no necesita resize
		{
			BBox_izq.erase(BBox_izq.begin()); // BB
			flow_vec.erase(flow_vec.begin()); // OF x
			flow_vec.erase(flow_vec.begin()); // OF y
		}
		// BBox_izq.resize(25); // ¿?		
		// flow_vec.resize(50); // ¿?

		if (frame_num>=LAST_FRAME) { // fin ultima inferencia
			pm_stop_counter(&pmlib_counter);
		}
	} // from 25
	
	// Liberar frames y OF para despejar Memoria
	// input.frame[0..5].release // flow
	for (int i=0; i<5; i++) {
		input.frames[i].release();
		input1.frames[i].release();
		input.flow[i].release();
		input1.flow[i].release();
		// input.BB[i].release();
		// input1.BB[i].release();
	}
	
	input.frames[5].release();
	input1.frames[5].release();
		
	return input; // std::get<0>(samples); // ¿? continue_msg
	}
};


int main(int argc, char **argv) 
{
	// Lectutra de PARAMETROS
	if (argc < 7) {
		std::cerr << "Usage: " << argv[0]
		<< " OFTasks BBTasks CNTasks LAST_FRAME ntask limitertasks initSched" << std::endl;
		return 1;
	} // deploy.prototxt network.caffemodel labels video 1/scale 

	int omp_OF = atoi(argv[1]);
	int opencv_BB = atoi(argv[2]);
	int openblas_CN = atoi(argv[3]);

	cv::setNumThreads(opencv_BB);
	omp_set_num_threads(omp_OF);
	openblas_set_num_threads(openblas_CN);
	
	Caffe::set_mode(Caffe::GPU);
	
	LAST_FRAME = (argc >= 5) ?  atoi(argv[4]) : LASTFRAME;
	// OF_SCALE =  atof(argv[5]);

	int ntasks = atoi(argv[5]); //4; // ¿?
	int lim_tasks = atoi(argv[6]); // 4; // ¿?
	
	int inintSched = atoi(argv[7]); // Frames OF CPU/GPU in a sample

#ifdef DEBUG
	// std::vector<string> labels;
	std::ifstream labelsf(F_LABEL);
	string line;
	while (std::getline(labelsf, line))
		labels.push_back(string(line));
#endif

	// INICIAR MEDIDAS BEAGLE
	server_t servidor;
	//counter_t pmlib_counter;
	line_t lineas;
	LINE_CLR_ALL(&lineas);
	LINE_SET( 7, &lineas ); // Solo uso esta
	pm_set_server("192.168.188.218", 6526, &servidor); // beagle2
	pm_create_counter("INA219Device", lineas, 1, 1000, servidor, &pmlib_counter);

	// Informacion para el LOG
	string line1;
    ifstream entrada("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    getline(entrada,line1); cout << "fCPU: " << line1 << " - "; entrada.close();
	entrada.open("/sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq");
    getline(entrada,line1); cout << "fGPU: " << line1 << " - "; entrada.close();	
    getline(entrada,line1); cout << "ncores_on: " << get_nprocs()  << endl; entrada.close();
	fprintf(stdout,"Graph_Sample * Threads OF: %d - BB: %d - CNN: %d\n",omp_OF,opencv_BB,openblas_CN);

	// Input Video file name
	string videoFileName = F_VIDEO;

	::google::InitGoogleLogging(argv[0]);
	FLAGS_minloglevel = 3; // Al log solo errores fatales

	/* Load the network. */
	// boost::shared_ptr<Net<float> > net;
	net.reset(new Net<float>(F_PROTO, TEST)); /* Load the network. */
	net->CopyTrainedLayersFrom(F_MODEL);

	// Blob<float>* input_layer = net->input_blobs()[0];
	// Blob<float>* output_layer = net->output_blobs()[0];
	input_layer = net->input_blobs()[0];
	output_layer = net->output_blobs()[0];
	
	bgmodel_CPU->setNMixtures(10); /* numMixtures */ 
////	bg_modelGPU->setNMixtures(10); /* numMixtures */ 

//--------------------------------

	// -- GRAFO --
	//////////////
	
	// Start task scheduler - Debe ser la primera llamada a TBB en el codigo
	tbb::task_scheduler_init init( ntasks /* 4 */ );
	
	graph g;

	// Nodos
	source_node<sampleOF> videoSampleInput(g, VideoRead(videoFileName,inintSched),false) ;
	limiter_node<sampleOF> limiter(g, lim_tasks); // 1 o 2, no lo tengo claro 0?
	function_node<sampleOF, sampleOF> computeOF_CPU( g, 1, OFComputationCPU() ); // Frames de la muestra en CPU
	function_node<sampleOF, sampleOF> computeOF_GPU( g, 1, OFComputationGPU() ); // Frames de la muestra en GPU
	function_node<sampleOF, sampleOF> computeBB_CPU( g, 1, BBComputationCPU() ); 
	join_node< tuple<sampleOF,sampleOF> > join(g);
	function_node< tuple<sampleOF,sampleOF>, continue_msg> output( g, 1, CNNInference() );
	
	// Aristas
	make_edge( videoSampleInput, limiter );
//	make_edge( videoSampleInput, computeOF_CPU );
//	make_edge( videoSampleInput, computeOF_GPU );
	make_edge( limiter, computeOF_CPU );
	make_edge( computeOF_CPU, computeBB_CPU );
	make_edge( limiter, computeOF_GPU );
//	make_edge( computeOF_CPU, input_port<0>(join) );
	make_edge( computeBB_CPU, input_port<0>(join) );
	make_edge( computeOF_GPU, input_port<1>(join) );
	make_edge( join, output);
	make_edge( output, limiter.decrement ); // limiter por frames terminados
 
// tbb::tick_count t0 = tbb::tick_count::now();

	videoSampleInput.activate();
	g.wait_for_all();

//--------------------------------

	// Mostrar RESULTADOS de rendimiento
	float time, potencia;

	pm_get_counter_data(&pmlib_counter);
	getMeasuresfromCounter(&time, &potencia, pmlib_counter);
	pm_finalize_counter(&pmlib_counter);

//---
int frame_num = LAST_FRAME;
sonda.nframes = frame_num;

cout << " Frames procesados " << frame_num << endl;

	sonda.muestras = true; // Para indicar que NO trabajamos a nivel de frame

	_printf_sonda(sonda, sonda.VIN.tini[26]);

	printf("Tiempo_medio_Infer: %f ms * Potencia_media: %f W * Energía_Infer: %f J\n",time/((frame_num-25)/5)*1000,potencia,time/((frame_num-25)/5)*potencia);

	fprintf(stdout,"----------\n");

	_printf_stats(sonda.VIN,"VIN",sonda.nframes);	cout << " ----------------------- " << endl;
	_printf_stats(sonda.OF,"OF",sonda.nframes); 	cout << " ----------------------- " << endl;
	_printf_stats(sonda.BB,"BB",sonda.nframes);		cout << " ----------------------- " << endl;
	_printf_stats(sonda.CNN,"CNN",sonda.nframes);	cout << " ----------------------- " << endl;
	_printf_stats(sonda.PRE,"PRE",sonda.nframes);	cout << " ----------------------- " << endl;
}
