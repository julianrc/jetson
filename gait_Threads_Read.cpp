#define F_PROTO "/home/jetson/gait/tumcaffe_deploy.prototxt"
#define F_MODEL "/home/jetson/gait/tumcaffe.caffemodel"
#define F_LABEL "/home/jetson/gait/tumgaidtestids.lst"
#define F_VIDEO "/home/jetson/gait/p003-n05.avi"
#define OF_SCALE 1.0
// float OF_SCALE; /* Para reajustar tamaño OF entrada (reducir complejidad) */
#define LASTFRAME 525 /* Para garantizar 100 Inferencias */
int LAST_FRAME; // Ojo, los arrays suponen maximo 1025 frames!
#define MAX_FRAMES 1025

// #define DEBUG

#include <opencv2/video/background_segm.hpp>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>

// Medir consumo con pmlib y Beaglebone UCM
#include <sys/sysinfo.h>
// #include "pmlib.h" // -> Ya está en medidas.h
#include <omp.h>
#include <thread>
#include "tbb/tick_count.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;
using namespace caffe;

#include "medidas.h" // Mis funciones de medidas, para aligerar los códigos
t_sonda sonda; // definicion del tipo en medidas.h

//
// Estrucutas de datos compartidas para almacenar los datos de una muestra
Mat frames[6]; // Incluye la previa, para el OF y BB
GpuMat Gframes[6];
vector<Mat> flow_vec; // Salida de OF

// pthread_barrier_t barrier;
// pthread_barrier_attr_t attr;
// unsigned count = 2;
#include "tbb/atomic.h"
tbb::atomic<int> finUSOFrames = 0;

cv::Ptr<cv::FarnebackOpticalFlow> fbOF_CPU; // = 		cv::cuda::FarnebackOpticalFlow::create(5,sqrt(2)/2.0,false,10,2,7,1.5,cv::OPTFLOW_FARNEBACK_GAUSSIAN);
cv::Ptr<cv::cuda::FarnebackOpticalFlow> fbOF_GPU; // = 		cv::cuda::FarnebackOpticalFlow::create(5,sqrt(2)/2.0,false,10,2,7,1.5,cv::OPTFLOW_FARNEBACK_GAUSSIAN);
	
void OF_in_CPU(vector <int> mis_frames, int frame_num)
{
	for(int i : mis_frames) {
		Mat flow_x_y, flow_res[2];
// cout << "OF-CPU " << i << " frames " << frame_num << endl;
_start_sonda(&(sonda.OF), frame_num+i, false);
		fbOF_CPU->calc(frames[i-1], frames[i], flow_x_y);
		cv::resize(flow_x_y, flow_x_y, Size(80,60));
_stop_sonda(&(sonda.OF),frame_num+i);

		cv::split(flow_x_y, flow_res);
		flow_vec[2*(i+19)]   = flow_res[0].clone(); // OFx
		flow_vec[2*(i+19)+1] = flow_res[1].clone(); // OFy
		//flow_res[0].release();
		//flow_res[1].release();
	}	
}

void OF_in_GPU(vector <int> mis_frames, int frame_num)
{
	for(int i : mis_frames) {
		
		GpuMat prvs_gpu, next_gpu, flow_gpu;
		Mat flow_x_y, flow_res, flow_split[2];
// cout << "OF-GPU " << i << " frames " << frame_num << endl;

//		prvs_gpu.upload(frames[i-1]);
//		next_gpu.upload(frames[i]);
		prvs_gpu= Gframes[i-1];
		next_gpu= Gframes[i];

if (i==5) {// Last frame of the sample
// cout << "Lei por ultima vez frame 5 de " << frame_num << endl;
		// pthread_barrier_wait(&barrier);
		finUSOFrames=1;
	// cout << "++ Barrera superada OF_GPU" << endl;
}
_start_sonda(&(sonda.OF), frame_num+i, true);
		fbOF_GPU->calc(prvs_gpu, next_gpu, flow_gpu);
		flow_gpu.download(flow_x_y);		
		cv::resize(flow_x_y, flow_res, Size(80,60)); // Hacer el resize antes que download gpu¿?
_stop_sonda(&(sonda.OF),frame_num+i);

		cv::split(flow_res, flow_split);
		flow_vec[2*(i+19)]   = flow_split[0].clone(); // OFx
		flow_vec[2*(i+19)+1] = flow_split[1].clone(); // OFy
	}	
}


int main(int argc, char **argv)
{
	thread th_OF_GPU, th_CNN_GPU;
	vector <int> frames_OF_CPU = {1}; // {1};// = {1,2,3,4,5};
	vector <int> frames_OF_GPU = {2,3,4,5};
	
	// Lectutra de PARAMETROS
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0]
		<< " OFTasks BBTasks CNTasks LAST_FRAME" << std::endl;
		return 1;
	} // deploy.prototxt network.caffemodel labels video 1/scale 
	
	int omp_OF = atoi(argv[1]);
	int opencv_BB = atoi(argv[2]);
	int openblas_CN = atoi(argv[3]);

	cv::setNumThreads(opencv_BB);
	omp_set_num_threads(omp_OF);
	openblas_set_num_threads(openblas_CN);
	Caffe::set_mode(Caffe::GPU);
		
	int frame_num = 0;
	LAST_FRAME = (argc == 5) ?  atoi(argv[4]) : LASTFRAME;
	// OF_SCALE =  atof(argv[5]);
	
	// INICIAR MEDIDAS BEAGLE
	
	server_t servidor;
	counter_t pmlib_counter;
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
	fprintf(stdout,"Threads_Sample_Read * Threads OF: %d - BB: %d - CNN: %d\n",omp_OF,opencv_BB,openblas_CN);

	// Input Video file name
	string videoFileName = F_VIDEO;
	
	/* Load labels */
#ifdef DEBUG
	std::vector<string> labels;
	std::ifstream labelsf(F_LABEL);
	string line;
	while (std::getline(labelsf, line))
		labels.push_back(string(line));
#endif

	// BB CPU variables
	int minArea = 1000; // No lo uso, pero debe eliminar BB pequeños
	bool updateBGModel = true;
    Mat GetImg, fgimg, fgmask;
	Ptr<cv::BackgroundSubtractorMOG2> bg_model = // (history, varThreshold, detectShadows)
		cv::createBackgroundSubtractorMOG2(40, 30, true);
	bg_model->setNMixtures( 10 /*numMixtures*/ );	
	Mat strel = getStructuringElement(MORPH_RECT, Size(3,3));
	
	// OF CPU/GPU variables
	Mat prvs, next;
	Mat flow_x_y, flujo_crop;	// Resultado downloaded de GPU -> para generar luego el vector ...
	Mat flow_res[2];
	fbOF_CPU = 
		cv::FarnebackOpticalFlow::create(5,sqrt(2)/2.0,false,10,2,7,1.5,cv::OPTFLOW_FARNEBACK_GAUSSIAN);
	fbOF_GPU = 
		cv::cuda::FarnebackOpticalFlow::create(5,sqrt(2)/2.0,false,13,2,7,1.5,cv::OPTFLOW_FARNEBACK_GAUSSIAN);
 	
	// BUFFERS
	float* bufferBB; // = input_layer->mutable_cpu_data(); // Post-corte BB (filas, columnas, canalx/y, muestras)
	vector<int> BBox_izq; // Centro BB, recorto hasta +60 -> BB (60x60)
	Rect ROI;

	// CNN CPU variables
	::google::InitGoogleLogging(argv[0]); // Caffe log (in /tmp)
	FLAGS_minloglevel = 3; // Al log solo errores fatales
	boost::shared_ptr<Net<float> > net;
	net.reset(new Net<float>(F_PROTO, TEST)); /* Load the network. */
	net->CopyTrainedLayersFrom(F_MODEL);
	Blob<float>* input_layer = net->input_blobs()[0];
	Blob<float>* output_layer = net->output_blobs()[0];

	// INICIALIZACION (llenado inicial del buffer) 
	/////////////////
// pthread_barrier_t barrier;
// pthread_barrier_attr_t attr;
// unsigned count = 2;
// Para evitar que lea nueva muestra antes de terminar de usar ultimo frame de la anterior
// pthread_barrier_init(&barrier, NULL /*&attr*/, 2 /*count*/);

	// Primer frame de la secuencia para OF (prvs - BN) y BB (GetImg - color)
	VideoCapture stream(F_VIDEO); // Video file open
	if(!(stream.read(GetImg))) // get one frame from video
		return -1;
 	cv::cvtColor(GetImg, prvs, COLOR_BGR2GRAY);

	//--- Inicializar modelo de fondo con primer frame leido
	bg_model->apply(prvs, fgmask, updateBGModel ? -1 : 0); // -1: automatically select the learning rate
	
	for (int i=1; i<=25; i++) { // Fill up the oflow buffer (25 x-, y- samples)

_start_sonda(&(sonda.PRE), 25, true);

_start_sonda(&(sonda.VIN), i, false);
		if(!(stream.read(GetImg))) // get one frame form video   
			return -1;
		cv::cvtColor(GetImg, next, COLOR_BGR2GRAY); //prev_res , next_res
_stop_sonda(&(sonda.VIN), i);
	
		frame_num++; 

_start_sonda(&(sonda.BB), i, false);
		bg_model->apply(next, fgmask, updateBGModel ? -1 : 0); // -1: automatically select the learning rate		
		morphologyEx(fgmask, fgimg, MORPH_OPEN, strel);
		Rect BB = boundingRect(fgimg);
_stop_sonda(&(sonda.BB),i);

		int izq = BB.x/8 + BB.width/(8*2); // - 30 (centro a izq) + 30 (borde) (ensancho luego por los bordes +60; // Para la escala de /8 (queda imagen de 60x80; sino, no sirve)
		BBox_izq.push_back(izq);

_start_sonda(&(sonda.OF), i, false);
		fbOF_CPU->calc(prvs, next, flow_x_y);
		cv::resize(flow_x_y, flow_x_y, Size(80,60));
_stop_sonda(&(sonda.OF),i);

		prvs = next.clone();
		//next.release();

        cv::split(flow_x_y, flow_res);
		flow_vec.push_back(flow_res[0].clone());
		flow_vec.push_back(flow_res[1].clone());
//		flow_res[0].release();
//		flow_res[1].release();

	}

	// Antes de llamar a CAFE hay que recortar con BB central todos los OF
	// para copiarlos en la capa de entrada de la Red
	int central = BBox_izq[12]; // 13º, el central de los 25 (0-24)
	ROI = Rect(central, 0, 60, 60); // Origen TOP-LEFT wx, wy

_start_sonda(&(sonda.CNN), 25, true);
	bufferBB = input_layer->mutable_cpu_data();
	for (int i=0; i<2*25; i++) {
		cv::copyMakeBorder(flow_vec[i], flujo_crop,0,0,30,30,BORDER_CONSTANT, Scalar(0.0f));
		Mat imgROI = 10*flujo_crop(ROI); // Escalar el OF x 10 (faltaria sumar media de la BD ~0.00...)
		memcpy(bufferBB+(60*60*(i)),imgROI.data,60*60*sizeof(float));
	}

	// Hacer primera prediccion con Caffe y mostrar resultados
	net->Forward();
_stop_sonda(&(sonda.CNN),25); // fin primer buffer completo
_stop_sonda(&(sonda.PRE),25); // fin primer buffer completo

// Eliminar los 5 del final para que los coloquen luego los threads

#ifdef DEBUG
	const float* salida = output_layer->cpu_data();
	for (int i=0; i<output_layer->channels(); i++) {
		if (salida[i]>0.01)
			std::cout << "Frame: " << frame_num << " Label: " << labels[i] 
					  << " prob = " << salida[i] << std::endl;
	}
#endif

{ // De calentamiento, siempre es mas lenta por inicilizar
		GpuMat prvs_gpu, next_gpu, flow_gpu;
		prvs_gpu.upload(prvs);
		next_gpu.upload(next);		
		fbOF_GPU->calc(prvs_gpu, next_gpu, flow_gpu);
}

	// Calcula nuevos OF (desplaza el buffer y encaja la nueva muestra al final)
	// ** PARTE QUE ES NUEVA PARA TRABAJAR CON MUESTRAS **
	// -> Ahora los Threads los meteran en su hueco correspondiente 20..24
	////////////////////
	
	 frames[0] = next.clone();
	//next.release();
	// if (frame_num==25) pm_start_counter(&pmlib_counter); // Estaba a 30
// _start_sonda(&(sonda.PRE), frame_num+5, true);
_start_sonda(&(sonda.VIN), frame_num+1, false);
	for (int i=1; i<=5; i++) {
		// Cargar los frames para calcular la siguiente muestra
		if(!(stream.read(GetImg))) {//get one frame form video   
			stream.set(CAP_PROP_POS_FRAMES, 0);// start (rewind)
#ifdef DEBUG
			cout << "[Rebobino << ] - Ultimo frame " << frame_num << endl;
#endif
				if (!(stream.read(GetImg))) { cout << "No pudo rebobinar " << endl; }
		}
		frame_num++;
		cv::cvtColor(GetImg, frames[i], COLOR_BGR2GRAY);
		Gframes[i].upload(frames[i]);
	}
_stop_sonda(&(sonda.VIN),frame_num-5+1);

	while (true) { // To the end of the video -> LAST_FRAME are processed

		// if (frame_num==LAST_FRAME) break; // Solo proceso los LAST_FRAME primeros frames
		if (frame_num-5==LAST_FRAME) break; // Solo proceso los LAST_FRAME primeros frames

		// if (frame_num==25) pm_start_counter(&pmlib_counter); // Estaba a 30
		if (frame_num==25+5) pm_start_counter(&pmlib_counter); // Estaba a 30

// _start_sonda(&(sonda.PRE), frame_num+5, true);
// _start_sonda(&(sonda.PRE), frame_num, true);

/*
_start_sonda(&(sonda.VIN), frame_num+1, false);
		for (int i=1; i<=5; i++) {
			// Cargar los frames para calcular la siguiente muestra
			if(!(stream.read(GetImg))) {//get one frame form video   
				stream.set(CAP_PROP_POS_FRAMES, 0);// start (rewind)
#ifdef DEBUG
				cout << "[Rebobino << ] - Ultimo frame " << frame_num << endl;
#endif
				if (!(stream.read(GetImg))) { cout << "No pudo rebobinar " << endl; }
			}
			frame_num++;
			cv::cvtColor(GetImg, frames[i], COLOR_BGR2GRAY);
		}
_stop_sonda(&(sonda.VIN),frame_num-5+1);
*/
		// Para sacarlo de los tiempos de leer frame:		
		for (int i=1; i<=5; i++) {
			BBox_izq.erase(BBox_izq.begin()); // BB
			flow_vec.erase(flow_vec.begin()); // OF x
			flow_vec.erase(flow_vec.begin()); // OF y
		}		
		BBox_izq.resize(25); //	¿?	
		flow_vec.resize(50); //
		
		// for (int i=0; i<=5; i++) Mat axu = frames[i].clone();
		
// _start_sonda(&(sonda.PRE), frame_num+5, true);
_start_sonda(&(sonda.PRE), frame_num, true);

		th_OF_GPU = thread(OF_in_GPU,frames_OF_GPU,frame_num-5);
		OF_in_CPU(frames_OF_CPU,frame_num-5); // Ponerle número de threads

		for (int i=1; i<=5; i++) {

// cout << "BB " << i << " frames " << frame_num << endl;			
			Mat fgmask, fgimg;
_start_sonda(&(sonda.BB), frame_num-5+i, false);
			bg_model->apply(frames[i], fgmask, updateBGModel ? -1 : 0); // -1: automatically select the learning rate
			morphologyEx(fgmask, fgimg, MORPH_OPEN, strel);
			Rect BB = boundingRect(fgimg); 
_stop_sonda(&(sonda.BB),frame_num-5+i);
			int izq = BB.x/8 + BB.width/(8*2);
			BBox_izq[i+19] = izq; // 20..24
		}

// Leer la siguiente muestra - Asegurarnos con un ¿Barrier? que OF4_GPU ya subio frames

// cout << "+ Esperando para leeer " << frame_num+1 << endl;
// pthread_barrier_wait(&barrier);
while (!finUSOFrames); // Espera activa
finUSOFrames=0;
// cout << "++ Barrera superada main" << endl;

//_start_sonda(&(sonda.PRE), frame_num+5, true);
		frames[0] = frames[5].clone(); // Prepara para la siguiente muestra
_start_sonda(&(sonda.VIN), frame_num+1, false);
		for (int i=1; i<=5; i++) {
			// Cargar los frames para calcular la siguiente muestra
			if(!(stream.read(GetImg))) {//get one frame form video   
				stream.set(CAP_PROP_POS_FRAMES, 0);// start (rewind)
#ifdef DEBUG
				cout << "[Rebobino << ] - Ultimo frame " << frame_num << endl;
#endif
				if (!(stream.read(GetImg))) { cout << "No pudo rebobinar " << endl; }
			}
			frame_num++;
			cv::cvtColor(GetImg, frames[i], COLOR_BGR2GRAY);
			Gframes[i].upload(frames[i]);
		}
_stop_sonda(&(sonda.VIN),frame_num-5+1);
// Fin leer siguiente muestra
// cout << "Fin leer siguiente muestra hasta frame" << frame_num << endl;
		
		th_OF_GPU.join(); // LA CPU espara a que se haya terminado el procesado de OF en GPU;

		// Antes de llamar a CAFE hay que recortar con BB central todos los OF
		int izq = BBox_izq[12]; // .x/8 + BBoxes[frame_num-13].width/(8*2) - 30 + 60;	// Central de los 25 primeros (0-24)
		ROI = Rect(izq, 0, 60, 60);

// _start_sonda(&(sonda.CNN), frame_num, true);		
_start_sonda(&(sonda.CNN), frame_num-5, true);		
		bufferBB = input_layer->mutable_cpu_data();
		for (int i=0; i<2*25; i++) {
				cv::copyMakeBorder(flow_vec[i], flujo_crop,0,0,30,30,BORDER_CONSTANT, Scalar(0.0f)); // Se podria hacer en GPU!!!
				Mat imgROI = 10*flujo_crop(ROI); // Escalar el OF x 10 (faltaria sumar media de la BD ~0.00...)
				memcpy(bufferBB+(60*60*(i)),imgROI.data,60*60*sizeof(float));
		}
		
		net->Forward();
_stop_sonda(&(sonda.CNN),frame_num-5);
// _stop_sonda(&(sonda.CNN),frame_num);
_stop_sonda(&(sonda.PRE),frame_num-5); // Se incremento en 5 

#ifdef DEBUG
		const float* salida = output_layer->cpu_data();
		for (int i=0; i<output_layer->channels(); i++) {
			if (salida[i]>0.01)
				std::cout << "Frame: " << frame_num << " Label: " << labels[i] 
						  << " prob = " << salida[i] << std::endl;
		}
#endif	

		// frames[0] = frames[5].clone(); // Prepara para la siguiente muestra
		// frames[5].release();
		
	} // while frames

	pm_stop_counter(&pmlib_counter);
		
	// Mostrar RESULTADOS de rendimiento
	float time, potencia;
	pm_get_counter_data(&pmlib_counter);
	getMeasuresfromCounter(&time, &potencia, pmlib_counter);
	pm_finalize_counter(&pmlib_counter);

//	printf("Tiempo_medio_Infer: %f ms * Potencia_media: %f W * Energía_Infer: %f J\n",time/((frame_num-25)/5)*1000,potencia,time/((frame_num-25)/5)*potencia);

frame_num-=5; // Leyo 5 mas de los que se procesaron

sonda.nframes = frame_num;
sonda.muestras = true; // Para indicar que NO trabajamos a nivel de frame

	_printf_sonda(sonda, sonda.BB.tini[1]);

	printf("Tiempo_medio_Infer: %f ms * Potencia_media: %f W * Energía_Infer: %f J\n",time/((frame_num-25)/5)*1000,potencia,time/((frame_num-25)/5)*potencia);

	fprintf(stdout,"----------\n");

	_printf_stats(sonda.VIN,"VIN",sonda.nframes);	cout << " ----------------------- " << endl;
	_printf_stats(sonda.OF,"OF",sonda.nframes); 	cout << " ----------------------- " << endl;
	_printf_stats(sonda.BB,"BB",sonda.nframes);		cout << " ----------------------- " << endl;
	_printf_stats(sonda.CNN,"CNN",sonda.nframes);	cout << " ----------------------- " << endl;
	_printf_stats(sonda.PRE,"PRE",sonda.nframes);	cout << " ----------------------- " << endl;
	
}
