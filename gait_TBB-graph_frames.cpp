#define F_PROTO "/home/jetson/gait/tumcaffe_deploy.prototxt"
#define F_MODEL "/home/jetson/gait/tumcaffe.caffemodel"
#define F_LABEL "/home/jetson/gait/tumgaidtestids.lst"
#define F_VIDEO "/home/jetson/gait/p003-n05.avi"
#define OF_SCALE 1.0
// float OF_SCALE; /* Para reajustar tamaño OF entrada (reducir complejidad) */ 
				   // hasta 2 funciona medio bien, con 4 y 8 mal
#define LASTFRAME 525 /* Para garantizar 100 Inferencias */
int LAST_FRAME; // Ojo, los arrays suponen maximo 1025 frames!
#define MAX_FRAMES 1025

// Ver permisos en cambiaHz

// ./gait_TBB-graph ../gait/tumcaffe_deploy.prototxt ../gait/tumcaffe.caffemodel ../gait/tumgaidtestids.lst ~/gait/p003-n05.avi 1 1 1 1 64
//
// Traduccion del pipeline del OF a version con grafo para completar la aplicacion
//
// Historia de Versiones. Mayo 2017 - Septiembre 2017 - Febrero 2019
//
// v1: En esta version solo usa CPU para el calculo del OF -> mysecond_pipeline-graph_v1.cpp
// v2: Calcula tambien OF en GPU. Planifico para que si GPU libre mande la tarea OF a GPU -> OptFlow_TBB-graph.cpp
// (faltaria incluir todo lo relativo al paso de parametros en linea, que esta en el original del pipeline)
// v3: Se han renombrado los nodos del OF y se van a incluir los de BB para crear un nuevo grafo, version solo CPU
// (por ahora solo vamos a usar GPU para OF, lo mas costoso con diferencia) -> Segmenta_TBB-graph.cpp
// v4: Grafo completo de la aplicacion, siguiendo la extructura del ejemplo basico test_split_join.cpp que me hice antes
// Cambiamos buffers por Mat opencv
// Trabajamos con atomics para bloquear el uso de la CPU a solo un proceso (OF/Caffe)
// v5: Renombrada a Tiempos: adaptada para generar cronograma de ejecución registrando dentro del código.

// Segun el codigo que desarrolle en su dia:
// El BB se calcula con el frame entero y luego se reescala. Finalmente se recorta el OF a un frame 60x60 centrado en BB.
// El valor del OF se escala x10 (habria que sumar tambien la media, pero es casi cero, se puede ignorar)

// TODO: El OF se calcula con el frame nomral y se reescala a 80x60 (/8) -> DONE

// Se aconseja usar Caffe compilado con OpenBlas y sin las cudNN (basado en pruebas de tiempos hechas)

#include <math.h>
#include <iostream>

// TBB
#include "tbb/flow_graph.h"
#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/atomic.h"

#include <caffe/caffe.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudabgsegm.hpp>

#include <opencv2/video/background_segm.hpp>

#include <opencv2/cudawarping.hpp> 	// resize
#include <opencv2/cudaarithm.hpp> 	// split
#include <opencv2/cudafilters.hpp> 

#include <cuda.h>
#include <cuda_runtime.h>

// Medir consumo con pmlib y Beaglebone UCM
#ifdef PM_LIB
#include "pmlib.h"
counter_t pmlib_counter;
#endif
// Medir consumo con la libreria que adapte de Andres para Odroid
#define ONLY_TIME
#include "energy_meter.h"
struct energy_sample *sample;

#include <sys/sysinfo.h>
#include <omp.h>

using namespace tbb::flow;
using namespace caffe;
using namespace cv;
using namespace cv::cuda;
using namespace std;

// Mis funciones de Medidas
// -----
#include "medidas.h"
t_sonda sonda;
// -----

//#define DEBUG

#ifdef DEBUG
	std::vector<string> labels;
#endif

// Decidir donde se reubica esto:
boost::shared_ptr<Net<float> > net;
Blob<float>* input_layer;
Blob<float>* output_layer; 

// Dice si las etapas OF, BB, PR intentaran hacer uso de la GPU
int OF_GPU, BB_GPU, PR_GPU;
//int OF_GPU_frm=0, BB_GPU_frm=0, PR_GPU_frm=0, inc_frm=0;
//int OF_CPU_frm=0, BB_CPU_frm=0, PR_CPU_frm=0;
tbb::atomic<int> OF_GPU_frm=0, BB_GPU_frm=0, PR_GPU_frm=0, inc_frm=0;
tbb::atomic<int> OF_CPU_frm=0, BB_CPU_frm=0, PR_CPU_frm=0;

// Si esta definido FUERZA_GPU se toman de los flags de entrada los que fuerza (Forced vs isFree)
#define FUERZA_GPU
// Fuerza el uso de la (0=GPU, 1=CPU)
// #define FUERZA_OF 0
// #define FUERZA_BB 1
// #define FUERZA_PR 0
int FUERZA_OF = 0;
int FUERZA_BB = 1;
int FUERZA_PR = 0;


//bool GPU_LIBRE = true;
tbb::atomic<int> nGPUs = 1; // GPUs libres, para controlar el acceso exclusivo
tbb::tick_count tfinP, tfinU;

struct framesOF { // Store all the data associated with a pair of frames (input frames and output flow)
	bool usaGPU; // Procesar en GPU/CPU -> No usado en este codigo, heredado de antiguo
	int index;
	Size dims;
	Rect BB;
	Mat prvs;
	Mat next;
	Mat flow;
};

// Dense GPU optical flow - lo declaro aqui para hacer el "calentamiento" de la GPU desde el main, antes de arrancar grafo
cv::Ptr<cv::cuda::FarnebackOpticalFlow> fbOF;

// Bgmodel Parameters -> incluir posteriormente en constructor de nodo que procesa ??
int minArea = 1000;
bool updateBGModel = true;
Ptr<cv::BackgroundSubtractorMOG2> bg_modelCPU = cv::createBackgroundSubtractorMOG2(40, 30, true);
Ptr<cuda::BackgroundSubtractorMOG2> bg_modelGPU = cuda::createBackgroundSubtractorMOG2(40, 30, true);
Mat strel = getStructuringElement(MORPH_RECT, Size(3,3));
Ptr<cuda::Filter> openFilter = cuda::createMorphologyFilter(MORPH_OPEN,  CV_8UC1, strel);

// Source node: devuelve false cuando no tiene mas que suministrar

struct VideoRead {
private:
	unsigned int index;
	VideoCapture stream;
	Mat last;
public:
    VideoRead(std::string file_name) : stream(file_name), index(0) {
		Mat Img;
		GpuMat d_img;
		
//		cout << "Opening " << file_name << endl;

		if(!(stream.read(Img))) //get one frame from video
		{ cout << "Error. Can't open " << file_name << endl; exit(-1);}
			
		cv::cvtColor(Img, Img, CV_BGR2GRAY);
		last = Img.clone();
/*		cout << "Frame size: " << last.cols/OF_SCALE << " x "
		    				   << last.rows/OF_SCALE << endl;
*/		
		// Inicializar modelo de fondo con el primer frame leido
		bg_modelCPU->apply(last, Img, updateBGModel ? -1 : 0);
		d_img.upload(last);
		bg_modelGPU->apply(d_img, d_img, updateBGModel ? -1 : 0);
	};
	
    ~VideoRead() { stream.release(); };
	
	bool operator()(framesOF& input) {
		if (index==(LAST_FRAME+1)) {
			// cout << "Last frame read" << index << endl;
			return false; // Solo proceso los LAST_FRAME primeros frames
		}

		Mat Img;

		if (index==25) { // El último frame YA procesado
#ifdef PM_LIB
			pm_start_counter(&pmlib_counter);
#endif
#ifdef ONLY_TIME
			tfinP = tbb::tick_count::now();
#else
			energy_meter_start(sample);  // starts sampling thread
#endif
		}

		index++;

_start_sonda(&(sonda.PRE), index, true);
// cout << "PRE empieza en " << index+5 << endl;
_start_sonda(&(sonda.VIN), index, false); 		
		if(!(stream.read(Img))) { //get one frame from video
			// cout << "Last frame read (" << index << ")" << endl;
			// Para que lea ciclicamente (AVI_RATIO no va)
			stream.set(CAP_PROP_POS_FRAMES, 0);// start (rewind)
			if (!(stream.read(Img))) { cout << "No pudo rebobinar " << endl; }
//			return false; 
		} // End - false finish de source node
		cv::cvtColor(Img, Img, CV_BGR2GRAY); // Usar un Mat intermedio para el color¿?
_stop_sonda(&(sonda.VIN),index);  

		input.index = index;
		input.dims = last.size();
		input.prvs = last.clone();
		last = Img.clone();
		input.next = last.clone();
		
// _stop_sonda(&(sonda.VIN),index);  
		
		return true;		
	}
	
};

struct OFComputation {
  framesOF operator()(framesOF input) // le he quitado el const &
	{	

		Mat prvs, next;
	
		// cv::resize(input.prvs, prvs, Size(input.dims.width/OF_SCALE, input.dims.height/OF_SCALE) );
		// cv::resize(input.next, next, Size(input.dims.width/OF_SCALE, input.dims.height/OF_SCALE) );

		if (!OF_GPU) { // Si NO permite a la etapa usar GPU
_start_sonda(&(sonda.OF), input.index, false);
			// calcOpticalFlowFarneback(prvs, next, input.flow, sqrt(2)/2.0, 5, 10, 2, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN );
			calcOpticalFlowFarneback(input.prvs, input.next, input.flow, sqrt(2)/2.0, 5, 10, 2, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN );
		}
		else{
		
		// Llamada CPU
#ifdef FUERZA_GPU
		if (FUERZA_OF) { // Fuerza el uso de la (0=GPU, 1=CPU)
#else
		if (--nGPUs <0) { // Si no quedan GPUs libres usa CPU
#endif
			nGPUs++;
_start_sonda(&(sonda.OF), input.index, false);
			calcOpticalFlowFarneback(input.prvs, input.next, input.flow, sqrt(2)/2.0, 5, 10, 2, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN );
		}// Llamada GPU
		else { // Si permite a la etapa usar GPU
			GpuMat prvs_gpu, next_gpu, flow_gpu;
			prvs_gpu.upload(input.prvs); //, stream);
			next_gpu.upload(input.next); //, stream);
_start_sonda(&(sonda.OF), input.index, true);
			fbOF->calc(prvs_gpu, next_gpu, flow_gpu);//, stream);
			flow_gpu.download(input.flow);//, stream);
			//stream.waitForCompletion();	
			nGPUs++;
		}
		// Fin GPU
		
		} // Permite a la etapa usar GPU?
		
		// Tras varias pruebas parece que lo mas rapido era hacer estas operaciones en CPU en ambos casos
	    cv::resize(input.flow, input.flow, Size(80,60)); // Usar Mat intermedia ¿?
        // cv::split(flow_x_y, flow_res);
_stop_sonda(&(sonda.OF),input.index);

		input.prvs.release(); // BB usa solo next

		return input; 
	}
};

struct BBComputation {
  framesOF operator()(framesOF input) // le he quitado el const &
	{	

	Mat img = input.next; // next, prvs para inicializar el primero solamente	
		
// 0- global, de BB; 1- local, cpu vs gpu 

	if (!BB_GPU) { // Si no permite a la etapa usar GPU
	    Mat fgmask, fgimg;
_start_sonda(&(sonda.BB), input.index, false);
		bg_modelCPU->apply(img, fgmask, updateBGModel ? -1 : 0); // -1: automatically select the learning rate
		morphologyEx(fgmask, fgimg, MORPH_OPEN, strel); // filter noise
		input.BB = boundingRect(fgimg); // Compute BB
	}
	else {
		// Llamada CPU
#ifdef FUERZA_GPU
		if (FUERZA_BB) { // Fuerza el uso de la (0=GPU, 1=CPU)
#else
		if (--nGPUs <0) { // Si no quedan GPUs libres usa CPU
#endif
		nGPUs++;
	    Mat fgmask, fgimg;
_start_sonda(&(sonda.BB), input.index, false);
		bg_modelCPU->apply(img, fgmask, updateBGModel ? -1 : 0); // -1: automatically select the learning rate
		morphologyEx(fgmask, fgimg, MORPH_OPEN, strel); // filter noise
		input.BB = boundingRect(fgimg); // Compute BB
	}
	else { // Usa GPU
		Mat fgimg;
		GpuMat d_img, d_fgmask, d_fgimg;
		d_img.upload(img);
_start_sonda(&(sonda.BB), input.index, true);
		bg_modelGPU->apply(d_img, d_fgmask, updateBGModel ? -1 : 0); 	
		openFilter->apply(d_fgmask, d_fgimg);
		d_fgimg.download(fgimg);
		input.BB = boundingRect(fgimg);
		nGPUs++;
	}
	}
_stop_sonda(&(sonda.BB),input.index);

// cout << "BB Termino frame " << input.index << endl;

	return input; 

	}
};

// Anadir a la clase CNNInference
vector<int> BBox_izq;
vector<Mat> flow_vec;

struct CNNInference {
	framesOF operator()(tuple<framesOF,framesOF> frames) { // le he quitado el const &tuple<int,float>

	int frame_num = (std::get<0>(frames).index)-1;
	// sampleOF inputOF = std::get<0>(samples); // Rama OF
	// sampleOF inputBB = std::get<1>(samples); // Rama BB
	
	Mat flow_x_y[2], flujo_crop, imgROI;
	Rect BB = std::get<1>(frames).BB;
	int izq = BB.x/8 + BB.width/(8*2); // - 30 + 30; // Para la escala de /8 (queda imagen de 60x80; sino, no sirve)

	BBox_izq.push_back(izq);
	cv::split(std::get<0>(frames).flow, flow_x_y); // split channels into two different images
	flow_vec.push_back(flow_x_y[0].clone());
	flow_vec.push_back(flow_x_y[1].clone());
		
	// A partir de tener llemos ya las 25 muestras (frames)
	if (frame_num>=25) {
		BBox_izq.erase(BBox_izq.begin()); 
		flow_vec.erase(flow_vec.begin());
		flow_vec.erase(flow_vec.begin());	
		
		if ((frame_num%5)==0) { // Predicciones solo cada 5 muestras
		int izquierda = BBox_izq[12]; // Central de los 25 (0-24)
		Rect ROI = Rect(izquierda, 0, 60, 60);

_start_sonda(&(sonda.CNN), frame_num, false); // TODO: no siempre sera TRUE	
		// Preparar entrada Red (25 muestras OFx OFy)
		// Antes de llamar a CAFE hay que recortar con BB centrado todos los OF
		float *bufferBB = input_layer->mutable_cpu_data();		
		for (int i=0; i<50; i++) { 
			cv::copyMakeBorder(flow_vec[i], flujo_crop,0,0,30,30,BORDER_CONSTANT, Scalar(0.0f)); // Se podria hacer en GPU!!!
			imgROI = 10*flujo_crop(ROI); // Escalar el OF x 10 (faltaria sumar media de la BD 0.00...) (necesario el x10?)
			memcpy(bufferBB+(60*60*(i)),imgROI.data,60*60*sizeof(float));
		}
		
			if (!PR_GPU) {
				Caffe::set_mode(Caffe::CPU);
				net->Forward(); // Hacer prediccion con Caffe 
			}
			else {
#ifdef FUERZA_GPU /* este if es al contrario de otros dos */
			if (!FUERZA_PR) { // Fuerza el uso de la (0=GPU, 1=CPU)
#else
			if (--nGPUs >=0) { // La GPU esta libre
#endif
sonda.CNN.usaGPU[frame_num] = true; // Unica vez que cambio a mano
				Caffe::set_mode(Caffe::GPU);
				net->Forward(); // Hacer prediccion con Caffe 
				nGPUs++; // libero
			}
			else {
				nGPUs++; // libero
				Caffe::set_mode(Caffe::CPU);
				net->Forward(); // Hacer prediccion con Caffe 
			}

			} // Fin Forwards
			
_stop_sonda(&(sonda.CNN),frame_num);
_stop_sonda(&(sonda.PRE),frame_num);
			
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

			if (frame_num==LAST_FRAME) { 		
#ifdef PM_LIB
	pm_stop_counter(&pmlib_counter);		
#endif
#ifdef ONLY_TIME
	tfinU = tbb::tick_count::now();
#else
	energy_meter_stop(sample);  	// stops sampling	
#endif	
			}

		} // each 5
	} // from 25
	
	std::get<0>(frames).next.release();
//	std::get<0>(frames).prvs.release();
	std::get<0>(frames).flow.release();
	
	return std::get<0>(frames);
	}
};

// Secuencer: dado objeto buffer devuelve su numero en la secuencia (MUY IMPORTANTE: comenzando por 0)
struct MySequencer {
	size_t operator()(const framesOF& frames) { 
		//std::cout << "Buff el numero " << frame.index << std::endl;
		return (frames.index)-1;
	}
};

int main(int argc, char **argv) {

	// Lectutra de PARAMETROS
	if (argc < 11) {
		std::cerr << "Usage: " << argv[0]
		<< " OFTasks BBTasks CNTasks LAST_FRAME ntask limitertasks useOF useBB useCN sr [ms]" << std::endl;
// << OF BB PR lim_tasks ***OFTasks BBTasks CNTasks LAST_FRAME" << std::endl;
		return 1;
	} // deploy.prototxt network.caffemodel labels video 1/scale 

	int omp_OF = atoi(argv[1]);
	int opencv_BB = atoi(argv[2]);
	int openblas_CN = atoi(argv[3]);

	cv::setNumThreads(opencv_BB);
	omp_set_num_threads(omp_OF);
	openblas_set_num_threads(openblas_CN);
		
	LAST_FRAME = (argc >= 5) ?  atoi(argv[4]) : LASTFRAME;
	// OF_SCALE =  atof(argv[5]);

	int ntasks = atoi(argv[5]); //4; // ¿?
	int lim_tasks = atoi(argv[6]); // 4; // ¿?
	
	// Dice si las etapas OF, BB, PR intentaran hacer uso de la GPU
	OF_GPU = atoi(argv[7]);
	BB_GPU = atoi(argv[8]);
	PR_GPU = atoi(argv[9]);
	
#ifdef FUERZA_GPU
FUERZA_OF = 1-OF_GPU;
FUERZA_BB = 1-BB_GPU;
FUERZA_PR = 1-PR_GPU;
#endif	

int OF_Paralelismo = 5;
	// int inintSched 	= atoi(argv[7]); // Frames OF CPU/GPU in a sample

#ifdef DEBUG
	// std::vector<string> labels;
	std::ifstream labelsf(F_LABEL);
	string line;
	while (std::getline(labelsf, line))
		labels.push_back(string(line));
#endif
	
	// INICIAR MEDIDAS BEAGLE
#ifdef PM_LIB
	server_t servidor;
	counter_t pmlib_counter;
	line_t lineas;
	LINE_CLR_ALL(&lineas);
	LINE_SET( 7, &lineas ); // Solo uso esta
	pm_set_server("192.168.188.218", 6526, &servidor); // beagle2
	pm_create_counter("INA219Device", lineas, 1, 1000, servidor, &pmlib_counter);
#endif

#ifndef ONLY_TIME
int sr = atoi(argv[10]); // 50 parace un buen valor
	// struct energy_sample *sample;
	sample=energy_meter_init(/*sr*/ sr, 0 /*debug*/);  // sr: sample rate in miliseconds (dificil bajar de 5)
#endif


	// Mostrar fCPU - fGPU
	string line1;
    ifstream entrada("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    getline(entrada,line1); cout << "fCPU: " << line1 << " - "; entrada.close();
	entrada.open("/sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq");
    getline(entrada,line1); cout << "fGPU: " << line1 << " - "; entrada.close();	
    getline(entrada,line1); cout << "ncores_on: " << get_nprocs()  << endl; entrada.close();
	fprintf(stdout,"TBB_Frames * Threads OF: %d - BB: %d - CNN: %d\n",omp_OF,opencv_BB,openblas_CN);

	// Extra de la version TBB Grafo personalizable
	char msg[256];
#ifdef FUERZA_GPU
// Fuerza el uso de la (0=GPU, 1=CPU) -> Va al reves que los flags
snprintf(msg,256,"GRAF_GPU_FORCED - OF-BB-PR: %d-%d-%d", 1-FUERZA_OF,1-FUERZA_BB,1-FUERZA_PR);
#else
snprintf(msg,256,"GRAF_GPU_isFREE - OF-BB-PR: %d-%d-%d", OF_GPU, BB_GPU, PR_GPU);
#endif
cout << msg ; //<< endl;
snprintf(msg,256," * ntasks: %d - lim_tasks: %d - OF_para: %d - Threads OF: %d - BB: %d - CNN: %d\n",
		 ntasks,lim_tasks,OF_Paralelismo,omp_OF,opencv_BB,openblas_CN);
cout << msg; // << endl;


	::google::InitGoogleLogging(argv[0]);
	FLAGS_minloglevel = 3; // Al log solo errores fatales
	// boost::shared_ptr<Net<float> > net;
	
	/* Load the network. */
	net.reset(new Net<float>(F_PROTO, TEST)); /* Load the network. */
	net->CopyTrainedLayersFrom(F_MODEL);

	// Blob<float>* input_layer = net->input_blobs()[0];
	// Blob<float>* output_layer = net->output_blobs()[0];
	input_layer = net->input_blobs()[0];
	output_layer = net->output_blobs()[0];

	// END -- Caffe --
	
	// Dense GPU optical flow creation and configuration. Same parameters as in w-Flow / libcudawFlow
	// FarnebackOpticalFlow::create	(numLevels = 5, pyrScale = 0.5, fastPyramids = false, winSize = 13,	numIters = 10,
	//								 polyN = 5, polySigma = 1.1, flags = 0)
	fbOF = cv::cuda::FarnebackOpticalFlow::create(5,sqrt(2)/2.0,false,13,2,7,1.5,cv::OPTFLOW_FARNEBACK_GAUSSIAN);

/*
	fbOF_CPU = 
		cv::FarnebackOpticalFlow::create(5,sqrt(2)/2.0,false,10,2,7,1.5,cv::OPTFLOW_FARNEBACK_GAUSSIAN);
	fbOF_GPU = 
		cv::cuda::FarnebackOpticalFlow::create(5,sqrt(2)/2.0,false,13,2,7,1.5,cv::OPTFLOW_FARNEBACK_GAUSSIAN);
*/
	
	// Calentamiento de la -- GPU -- para el OF
/*
	{
		Mat prvs(Mat::zeros(480,640,CV_8U)), next(Mat::zeros(480,640,CV_8U));
		GpuMat prvs_gpu, next_gpu, flow_gpu;
		prvs_gpu.upload(prvs);
		next_gpu.upload(next);
		fbOF->calc(prvs_gpu, next_gpu, flow_gpu);
	}
*/

	bg_modelCPU->setNMixtures(10); /* numMixtures */ 
	bg_modelGPU->setNMixtures(10); /* numMixtures */ 

	// -- GRAFO --
	
	// Start task scheduler - Debe ser la primera llamada a TBB en el codigo
	tbb::task_scheduler_init init( ntasks /* 4 */ );
	
	graph g;
	
	source_node<framesOF> videoInput(g, VideoRead(F_VIDEO),false) ;//"/home/ubuntu/src/p003-n05.avi"), false);
		
	// FALSO: No lo vamos a usar de momento...
	limiter_node<framesOF> limiter(g, lim_tasks); //16 // 50 //ntasks /*nthreads*4*///Cuidado con el limite, puede malfuncionar al grafo
	
	function_node<framesOF, framesOF> computeOF( g, OF_Paralelismo, OFComputation() ); // unlimited ?
	function_node<framesOF, framesOF> computeBB( g, 1, BBComputation() ); // unlimited no posible, es secuencial
	
	sequencer_node<framesOF> sequencer(g, MySequencer());
  
	join_node< tuple<framesOF,framesOF> > join(g);
  
	function_node< tuple<framesOF,framesOF>, continue_msg> output( g, 1, CNNInference() );
	
	// Con sequencer
//	make_edge( videoInput, computeOF );
//	make_edge( videoInput, computeBB );
	make_edge( videoInput, limiter ); // Conectar mejor el limiter con la salidad del OF?
	make_edge( limiter, computeBB );
	make_edge( limiter, computeOF ); // Conectar mejor el limiter con la salidad del OF?
	
	make_edge( computeOF, sequencer );
	make_edge( sequencer, input_port<0>(join) );
	make_edge( computeBB, input_port<1>(join) );
	make_edge( join, output);

	make_edge( output, limiter.decrement ); // limiter por frames terminados
	//make_edge( sequencer, limiter.decrement );
    	
	// tbb::tick_count t0 = tbb::tick_count::now();

	videoInput.activate();
	g.wait_for_all();

#ifdef PM_LIB
	float time, potencia;
	pm_get_counter_data(&pmlib_counter); //J
	getMeasuresfromCounter(&time, &potencia, pmlib_counter);
	pm_finalize_counter(&pmlib_counter);
#endif	
	
	int frame_num = LAST_FRAME;
	
sonda.nframes = LAST_FRAME;

	_printf_sonda(sonda, sonda.OF.tini[1]);

	// printf("Tiempo_medio_Infer: %f ms * Potencia_media: %f W * Energía_Infer: %f J\n",time/((frame_num-25)/5)*1000,potencia,time/((frame_num-25)/5)*potencia);

	fprintf(stdout,"----------\n");

	_printf_stats(sonda.VIN,"VIN",sonda.nframes);	cout << " ----------------------- " << endl;
	_printf_stats(sonda.OF,"OF",sonda.nframes); 	cout << " ----------------------- " << endl;
	_printf_stats(sonda.BB,"BB",sonda.nframes);		cout << " ----------------------- " << endl;
	_printf_stats(sonda.CNN,"CNN",sonda.nframes);	cout << " ----------------------- " << endl;
	_printf_stats(sonda.PRE,"PRE",sonda.nframes);	cout << " ----------------------- " << endl;	

	fprintf(stdout,"----------\n");

// energy_meter measurements:
	int n_infer = (frame_num-25)/5; // sonda.nframes (se quita la primera que es diferente)


#ifdef ONLY_TIME
	cout << "N_infer " << n_infer << endl;
	printf("Tiempo_medio_Infer: %f ms * Potencia_media: Unknown W\n", ((tfinU-tfinP).seconds()*1000)/n_infer);
#else
	struct timespec res = diff(sample->start_time, sample->stop_time);
	double total_time = (double)res.tv_sec+ (double)res.tv_nsec/1000000000.0;

	cout << "N_infer " << n_infer << " - n_samples_inf " << sample->samples/n_infer << " - sampling_rate ms " 
		 << total_time*1000/sample->samples << endl;

	cout << "ENERGIAxINFER mJ " << sample->TOT/n_infer << " = CPU " << sample->CPU/n_infer << " + GPU " << sample->GPU/n_infer
		 << " + Others " << (sample->TOT-(sample->CPU+sample->GPU))/n_infer << endl;

	printf("Tiempo_medio_Infer: %f ms * Potencia_media: %f W * Energía_Infer: %f J\n",
			total_time/n_infer*1000,
			(double)(sample->TOT)/(total_time*1000),
			sample->TOT/(n_infer*1000));
	energy_meter_destroy(sample);     // clean up everything
#endif
	
	return 0;
}
