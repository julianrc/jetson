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

#define VIDEO
#define DEBUG
// #define PM_LIB

// ./gait_Pipe ../gait/tumcaffe_deploy.prototxt ../gait/tumcaffe.caffemodel ../gait/tumgaidtestids.lst ../gait/p003-n05.avi 1 4 4 1 # Scale tokens ntasks OpenBlas_Threads

// Para usar la coma como separador decimales
// #include<locale>
// struct comma_separator : std::numpunct<char> {
    // virtual char do_decimal_point() const override { return ','; }
// };

// TBB Pipeline, ticks, scheduler
#include "tbb/pipeline.h"
#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"

//#include <cblas.h>

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

#include <sys/sysinfo.h>
#include "pmlib.h"
#include <omp.h>

// Medir consumo con pmlib y Beaglebone UCM
#include "pmlib.h"
counter_t pmlib_counter;

// tbb::tick_count tfinP, tfinU;

using namespace caffe;
using namespace cv;
using namespace cv::cuda;
using namespace std;

#ifdef DEBUG
	std::vector<string> labels;
#endif

// Mis funciones de Medidas
// -----
#include "medidas.h"
t_sonda sonda;
// ----

// float OF_SCALE=1; /* Para reajustar tamaño OF entrada (reducir complejidad) */
// hasta 2 funciona medio bien, con 4 y 8 mal
//#define LAST_FRAME 125 /* Para garantizar 21 muestras, medimos de fin 1 a fin 21 (20) */
// int LAST_FRAME; // Ojo, los arrays suponen maximo 1025 frames!

// ------------------------------------------------------------------

// Store all the data associated with a pair of frames (input frames and output flow)

typedef struct {
	bool usaGPU_OF; // Procesar en GPU/CPU
	bool usaGPU_BB; 
	bool usaGPU_CN; 
	int index;
	Size dims;
	Rect BB;
//	Mat prvs;
//	Mat next;
//	Mat flow;
	uchar* prvs;
	uchar* next;
	float* flow;
} t_framesOF;

// tbb::tick_count t0;

// ------------------------------------------------------------------

// OF

// Dense GPU optical flow - lo declaro aqui para hacer el "calentamiento" de la GPU desde el main, antes de arrancar grafo
//cv::Ptr<cv::cuda::FarnebackOpticalFlow> fbOF;

// BB

// Bgmodel Parameters -> incluir posteriormente en constructor del nodo que procesa ??
/*
int minArea = 1000;
bool updateBGModel = true;
Ptr<cv::BackgroundSubtractorMOG2> bg_modelCPU = cv::createBackgroundSubtractorMOG2(40, 30, true);
Ptr<cuda::BackgroundSubtractorMOG2> bg_modelGPU = cuda::createBackgroundSubtractorMOG2(40, 30, true);
Mat strel = getStructuringElement(MORPH_RECT, Size(3,3));
Ptr<cuda::Filter> openFilter = cuda::createMorphologyFilter(MORPH_OPEN,  CV_8UC1, strel);
*/

// CNN

// Decidir donde se reubica esto:
/*
boost::shared_ptr<Net<float> > net;
Blob<float>* input_layer;
Blob<float>* output_layer; 
//std::vector<string> labels;
*/

// ------------------------------------------------------------------

// Source node: devuelve false cuando no tiene mas que suministrar
// Asign a sequence number to each buffer as it is allocated at the input filter (start at 0).

class VideoRead: public tbb::filter {

private:

	unsigned int index;
	VideoCapture stream;
	Mat last;

public:
	
	// Constructor
    VideoRead(std::string file_name) : filter(serial_in_order), index(0), stream(file_name)
	{
		Mat Img;
		
	//	cout << "Opening " << file_name << endl;

		if(!(stream.read(Img))) //get one frame from video
		{ cout << "Error. Can't open " << file_name << endl; exit(-1);}
			
		cv::cvtColor(Img, Img, CV_BGR2GRAY);
		// cv::resize(Img, Img, Size(), 1/OF_SCALE, 1/OF_SCALE);
		// Size(Img.cols/OF_SCALE, Img.rows/OF_SCALE) ); // width,height
		last = Img.clone();
			
//		cout << "Frame size: " << last.size() << endl;
		
	};
	// Destructor
    ~VideoRead() { stream.release(); };
	
	// Operador
	void* operator()(void *) {
	
		Mat Img;
		t_framesOF* input = (t_framesOF*)malloc(sizeof(t_framesOF));
		
		input->dims = last.size();
		input->prvs = (uchar*)malloc(input->dims.height*input->dims.width*sizeof(uchar));
		input->next = (uchar*)malloc(input->dims.height*input->dims.width*sizeof(uchar));

		if (index==(LAST_FRAME+1)) {
			return NULL; // Solo proceso los LAST_FRAME primeros frames
		}		

#ifdef PM_LIB
		if (index==25) { // Vamos a comenzar el 26
			// tfinP = tbb::tick_count::now();
			// cout << "Comienza contador" << endl;
			pm_start_counter(&pmlib_counter);
		}
#endif
		index++;
		
_start_sonda(&(sonda.PRE), index, true);
_start_sonda(&(sonda.VIN), index, false); 		
		if(!(stream.read(Img))) { //get one frame from video
			stream.set(CAP_PROP_POS_FRAMES, 0); // start (rewind)
			if (!(stream.read(Img))) { cout << "No pudo rebobinar " << endl; }
		} // End - 'false' finish source node
	
		cv::cvtColor(Img, Img, CV_BGR2GRAY);
		// cv::resize(Img, Img, Size(), 1/OF_SCALE, 1/OF_SCALE);
		// Size(Img.cols/OF_SCALE, Img.rows/OF_SCALE) );
_stop_sonda(&(sonda.VIN),index);

		input->index = index;
		// input->dims = last.size();
		//input->prvs = last.clone();
		memcpy(input->prvs,last.data,input->dims.height*input->dims.width*sizeof(uchar));
		last = Img.clone();
		//input->next = last.clone();
		memcpy(input->next,last.data,input->dims.height*input->dims.width*sizeof(uchar));
		
		// Algoritmo de PLANIFICACION
		input->usaGPU_OF = true;// false;
		input->usaGPU_BB = false;
		input->usaGPU_CN = (index%5==0); //false
	
		return input;
	}
	
};

// ------------------------------------------------------------------
class OFComputation: public tbb::filter {

private:
  // Dense GPU optical flow
  cv::Ptr<cv::cuda::FarnebackOpticalFlow> fbOF;
	
public:

  /* Constructor: codigo para inicializar */
  OFComputation(): filter(serial_in_order) {  
	// Dense GPU optical flow creation and configuration
	// fbOF = cv::cuda::FarnebackOpticalFlow::create(5,sqrt(2)/2.0,false,10,2,7,1.5,cv::OPTFLOW_FARNEBACK_GAUSSIAN);
	fbOF = cv::cuda::FarnebackOpticalFlow::create(5,sqrt(2)/2.0,false,13,2,7,1.5,cv::OPTFLOW_FARNEBACK_GAUSSIAN);
  };

  void* operator()(void* item) // le he quitado el const &
	{		
		t_framesOF* input = (t_framesOF*) item;
	
		// sonda.OF.usaGPU[input->index] = input->usaGPU_OF;
		// sonda.OF.core[input->index] = sched_getcpu();
		// sonda.OF.tini[input->index] = tbb::tick_count::now();
		
		Mat prvs(input->dims, CV_8U, input->prvs);	//Mat prvs, next;
		Mat next(input->dims, CV_8U, input->next);
		Mat Matflow;
		input->flow = (float*) malloc(60*80*2*sizeof(float));		

		if (!(input->usaGPU_OF)) { // Si NO permite a la etapa usar CPU
_start_sonda(&(sonda.OF), input->index, false);
			calcOpticalFlowFarneback(prvs, next, Matflow, sqrt(2)/2.0, 5, 10, 2, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN );

//			cout << "OF CPU " << input->index << endl;
		}
		else { // Si permite a la etapa usar GPU
			GpuMat prvs_gpu, next_gpu, flow_gpu;
			prvs_gpu.upload(prvs);
			next_gpu.upload(next);
_start_sonda(&(sonda.OF), input->index, true);
			fbOF->calc(prvs_gpu, next_gpu, flow_gpu);
			flow_gpu.download(Matflow);
			
//			cout << "OF GPU " << input->index << endl;

		}
		
		// Parece mas rapido hacer esta operacion en CPU siempre
		cv::resize(Matflow, Matflow, Size(80,60)); // fx(Ancho) x fy(Alto)
_stop_sonda(&(sonda.OF), input->index);  

		memcpy(input->flow,(float*)(Matflow.data),60*80*2*sizeof(float));

		// sonda.OF.tfin[input->index] = tbb::tick_count::now();

		return input; 
	}
};

// ------------------------------------------------------------------
class BBComputation: public tbb::filter {

private:
// Bgmodel Parameters -> incluir posteriormente en constructor del nodo que procesa ??
	// int minArea;// = 1000;
	bool updateBGModel;// = true;
	Ptr<cv::BackgroundSubtractorMOG2> bg_modelCPU;// = cv::createBackgroundSubtractorMOG2(40, 30, true);
	Ptr<cuda::BackgroundSubtractorMOG2> bg_modelGPU; // = cuda::createBackgroundSubtractorMOG2(40, 30, true);
	Mat strel; // = getStructuringElement(MORPH_RECT, Size(3,3));
	Ptr<cuda::Filter> openFilter; // = cuda::createMorphologyFilter(MORPH_OPEN,  CV_8UC1, strel);

public:
	/* Constructor: codigo para inicializar */
	BBComputation(): filter(serial_in_order) 
	{
		// minArea = 1000;
		updateBGModel = true;
		bg_modelCPU = cv::createBackgroundSubtractorMOG2(40, 30, true);
		bg_modelGPU = cuda::createBackgroundSubtractorMOG2(40, 30, true);
		strel = getStructuringElement(MORPH_RECT, Size(3,3));
		openFilter = cuda::createMorphologyFilter(MORPH_OPEN,  CV_8UC1, strel);
		
		bg_modelCPU->setNMixtures(10); /* numMixtures */ 
		bg_modelGPU->setNMixtures(10); /* numMixtures */ 
	};
	
	void* operator()(void* item) // le he quitado el const &
	{	
		t_framesOF* input = (t_framesOF*)item;

		// if (input->index==26) {
			// // tfinP = tbb::tick_count::now();
			// pm_start_counter(&pmlib_counter);
		// }
		
		// sonda.BB.usaGPU[input->index] = input->usaGPU_BB;
		// sonda.BB.core[input->index] = sched_getcpu();
		// sonda.BB.tini[input->index] = tbb::tick_count::now();
		
		Mat fgimg;
		// Mat img = input->next; // next; prvs para inicializar el primero solamente
		Mat img(input->dims, CV_8U, input->next);

		// Inicializar modelo de fondo con el primer frame leido
		if (input->index==1)
		{
			GpuMat d_img;
			Mat last(input->dims, CV_8U, input->prvs);
_start_sonda(&(sonda.BB), input->index, false);
			bg_modelCPU->apply(last, last, updateBGModel ? -1 : 0);
			d_img.upload(last);
			bg_modelGPU->apply(d_img, d_img, updateBGModel ? -1 : 0);
		}

		if (!(input->usaGPU_BB)) { // Si permite a la etapa usar CPU
			Mat fgmask;
_start_sonda(&(sonda.BB), input->index, false);
			bg_modelCPU->apply(img, fgmask, updateBGModel ? -1 : 0); // -1: automatically select the learning rate
			morphologyEx(fgmask, fgimg, MORPH_OPEN, strel); // filter noise

//			cout << "BB CPU " << input->index << endl;
		}
		else { // Usa GPU
			GpuMat d_img, d_fgmask, d_fgimg;
			
			d_img.upload(img);
_start_sonda(&(sonda.BB), input->index, true);
			bg_modelGPU->apply(d_img, d_fgmask, updateBGModel ? -1 : 0); 	
			openFilter->apply(d_fgmask, d_fgimg);
			d_fgimg.download(fgimg);
			
//			cout << "BB GPU " << input->index << endl;
		}
		
		input->BB = boundingRect(fgimg);
_stop_sonda(&(sonda.BB),input->index);

		free(input->prvs); // No lo usare mas, en realidad solo para inicializar modeloBG la primera vez

		// sonda.BB.tfin[input->index] = tbb::tick_count::now();
		
/*		cout << "BB " << input->index << " " <<
			(sonda.BB.tini[input->index]-t0).seconds() << " " <<
			(sonda.BB.tfin[input->index]-sonda.BB.tini[input->index]).seconds()*1000
			<< endl;
*/
		return input; 
	}
};

// ------------------------------------------------------------------
class CNNInference: public tbb::filter {

private:
	
	vector<int> BBox_izq;
	vector<Mat> flow_vec;
	
	boost::shared_ptr<Net<float> > net;
	Blob<float>* input_layer;
	Blob<float>* output_layer; 
//std::vector<string> labels;

public:

	/* Constructor: codigo para inicializar */
  CNNInference(/*string prog_name,*/ string net_file, string weights_file) : filter(serial_in_order)
	{
		//::google::InitGoogleLogging(prog_name);
		/*
		std::ifstream labelsf(argv[3]);
		string line;
		while (std::getline(labelsf, line))
			labels.push_back(string(line));
		*/
		
		/* Load the network. */
		net.reset(new Net<float>(net_file, TEST));
		net->CopyTrainedLayersFrom(weights_file);

		input_layer = net->input_blobs()[0];
		output_layer = net->output_blobs()[0];
	};

  void* operator()(void* item) {

	t_framesOF* input = (t_framesOF*)item;
	
	// sonda.CNN.usaGPU[input->index] = input->usaGPU_CN;
	// sonda.CNN.core[input->index] = sched_getcpu();
	// sonda.CNN.tini[input->index] = tbb::tick_count::now();

	int frame_num = input->index;// - 1;
	// Mat flow_x_y[2], 
	Mat flujo_crop, imgROI;
	Rect BB = input->BB;
	int izq = BB.x/8 + BB.width/(8*2); // - 30 + 30; // Para la escala de /8 (queda imagen de 80x60; sino, no sirve)

	BBox_izq.push_back(izq);
	// cv::split(input->flow, flow_x_y); // flow aqui es cv::Mat
	// flow_vec.push_back(flow_x_y[0].clone());
	// flow_vec.push_back(flow_x_y[1].clone());
	Mat flow_x(Size(80,60), CV_32F, &(input->flow[0]));
	Mat flow_y(Size(80,60), CV_32F, &(input->flow[80*60])); //Donde comienza fy, tras fx
	
#ifdef VIDEO
                {
                  drawOpticalFlow(/*const Mat&*/ input->flow, /*const Mat&*/ /*GetImg*/ input->next);
                  rectangle(input->next,BB,Scalar(0, 0, 255));
                  videoOF.write(input->next);
                }
#endif	
	
	
	
	
	flow_vec.push_back(flow_x.clone()); 
	flow_vec.push_back(flow_y.clone()); 
		
	// A partir de tener llemas ya las 25 muestras (frames)
	if (frame_num>=25) {
/*		BBox_izq.erase(BBox_izq.begin()); 
		flow_vec.erase(flow_vec.begin());
		flow_vec.erase(flow_vec.begin());
*/
		
		if ((frame_num%5)==0) { // Predicciones nuevas solo cada ventana de 5 muestras
			int izquierda = BBox_izq[12]; // Central de los 25 (0-24)
			Rect ROI = Rect(izquierda, 0, 60, 60);
			// Preparar entrada Red (25 muestras OFx OFy)
			// Antes de llamar a CAFE hay que recortar con BB centrado todos los OF
			
_start_sonda(&(sonda.CNN), frame_num, false); // TODO: no siempre sera TRUE	
			float *bufferBB = input_layer->mutable_cpu_data();		
			for (int i=0; i<50; i++) { 
				cv::copyMakeBorder(flow_vec[i], flujo_crop,0,0,30,30,BORDER_CONSTANT, Scalar(0.0f)); // Copia rellenado bordes. Se podria hacer en GPU!!!
				imgROI = 10*flujo_crop(ROI); // Escalar el OF x 10 (faltaria sumar media de la BD, proxima a 0.00...)
				memcpy(bufferBB+(60*60*(i)),imgROI.data,60*60*sizeof(float));
			}
		
			if (!(input->usaGPU_CN)) {
				Caffe::set_mode(Caffe::CPU);
				net->Forward(); // Hacer prediccion

//				cout << "CN CPU " << input->index << endl;
			}
			else {
sonda.CNN.usaGPU[frame_num] = true; // Unica vez que cambio a mano
				Caffe::set_mode(Caffe::GPU);
				net->Forward(); // Hacer prediccion 
				
//				cout << "BB GPU " << input->index << endl;
			}
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

			// if (frame_num==25) { // fin Pred_1
				// //t20 = tbb::tick_count::now();
				// //pm_start_counter(&contadorS2);
			// }
#ifdef PM_LIB
			if (frame_num==LAST_FRAME) { // fin Pred_21
				pm_stop_counter(&pmlib_counter);
				// cout << "Acaba contador" << endl;
				// tfinU = tbb::tick_count::now();
			}
#endif
		} // each 5
	
		// A partir de 25 muestras vamos liberando por la cabeza la cola circular
		BBox_izq.erase(BBox_izq.begin()); // BB
		flow_vec.erase(flow_vec.begin()); // fx 
		flow_vec.erase(flow_vec.begin()); // fy
		
	} // from 25
	
	// input->next.release();
	// input->flow.release();
	free(input->next);
	free(input->flow);

	// sonda.CNN.tfin[input->index] = tbb::tick_count::now();
	
	return NULL;
  }
};

// ------------------------------------------------------------------

 #ifdef VIDEO
        // Videos Salida
        int frame_width  = 640; //stream.get(CV_CAP_PROP_FRAME_WIDTH);
        int frame_height = 480; //stream.get(CV_CAP_PROP_FRAME_HEIGHT);
        //int fps = stream.get(CV_CAP_PROP_FPS);

        VideoWriter videoOF("out_of1.avi",CV_FOURCC('M','J','P','G'),5/*fps*/,
                                                  Size(frame_width,frame_height));
#endif
 
int main(int argc, char **argv)
{
  
	if (argc != 7) {
		std::cerr << "Usage: " << argv[0]
		<< " ntokens ntasks OFTasks BBTasks CNTasks LAST_FRAME" << std::endl;
		// << " deploy.prototxt network.caffemodel labels video 1/scale ntokens ntasks OFTasks BBTasks CNTasks LAST_FRAME" << std::endl;
		return 1;
	}
	
  ::google::InitGoogleLogging(argv[0]);
  FLAGS_minloglevel = 3; // Al log solo errores fatales

  // LAST_FRAME = atoi(argv[11]);
  LAST_FRAME = (argc >= 7) ?  atoi(argv[6]) : LASTFRAME;
// cout << "Proceso " << LAST_FRAME << " frames" << endl;
//  NTHREADS = opts.ncores;
 
  int omp_OF = atoi(argv[3]);
  int opencv_BB = atoi(argv[4]);
  int openblas_CN = atoi(argv[5]);
  
  int n_tokens = atoi(argv[1]);
  int ntasks = atoi(argv[2]);
  
  	// Init 
	cv::setNumThreads(opencv_BB);
	omp_set_num_threads(omp_OF);
	openblas_set_num_threads(openblas_CN);
  
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
	line_t lineas;
	LINE_CLR_ALL(&lineas);
	LINE_SET( 7, &lineas );
	pm_set_server("192.168.188.218", 6526, &servidor);
	pm_create_counter("INA219Device", lineas, 1, 1000, servidor, &pmlib_counter);
#endif

	// Mostrar fCPU - fGPU
	string line1;
    ifstream entrada("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    getline(entrada,line1); cout << "fCPU: " << line1 << " - "; entrada.close();
	entrada.open("/sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq");
    getline(entrada,line1); cout << "fGPU: " << line1 << " - "; entrada.close();	
//	entrada.open("/sys/devices/system/cpu/online");
    getline(entrada,line1); cout << "ncores_on: " << get_nprocs()  << endl; // entrada.close();
	
	char msg[256];
snprintf(msg,256,"PIPE - OF-BB-PR: %s-%s-%s", "1","0","0");
cout << msg ; //<< endl;
snprintf(msg,256," * ntasks: %d - tokens: %d - Threads OF: %d - BB: %d - CNN: %d\n",ntasks,n_tokens,omp_OF,opencv_BB,openblas_CN);
cout << msg; // << endl;

  //int ntasks = opts.ncores;
  
  // Start task scheduler
  tbb::task_scheduler_init init( ntasks );
 
  // Create the pipeline
  tbb::pipeline pipeline;

  // Create FRAME READING stage and add it to the pipeline
  VideoRead myVideoRead(F_VIDEO); //argv[4]); // Constructor
  pipeline.add_filter( myVideoRead );
 
  // Create BB+CNN stage and add it to the pipeline
  BBComputation myBBComputation; 
  pipeline.add_filter( myBBComputation );

  // Create OF stage and add it to the pipeline
  OFComputation myOFComputation; 
  pipeline.add_filter( myOFComputation ); 
  
  // Create INFERENCE stage and add it to the pipeline
  CNNInference myCNNInference(F_PROTO, F_MODEL); //(/*argv[0],*/ argv[1], argv[2]);
  pipeline.add_filter( myCNNInference );

// tbb::tick_count 
// t0 = tbb::tick_count::now();
  // Primera ejecucion especial ¿?
 // if (uno_GPU)
//  pipeline.run( 4*8 ); // pipeline.run( n_tokens );
	
  // Run the pipeline
  // // tbb::tick_count tini = tbb::tick_count::now();
  // Need more than one token in flight per thread to keep all threads busy; 2-4 works
    pipeline.run( n_tokens ); // pipeline.run( n_tokens );
  // // tbb::tick_count tfin = tbb::tick_count::now();

 // if (opts.use_gpu) GPU_frms--;
/*  
  cout << "Threads " << ntasks << endl;
  cout << "Frames: Total " << CPU_frms+GPU_frms << " CPU " << CPU_frms << " GPU " << GPU_frms << endl;
  cout << "Time: Total " << (t1-t0).seconds() << " frame " << (t1-t0).seconds()/(CPU_frms+GPU_frms) << " fps " << (CPU_frms+GPU_frms)/(t1-t0).seconds() << endl;
*/

  // // Remove filters from pipeline before they are implicitly destroyed.
  // pipeline.clear(); 

// Volcado medidas
// cout.precision(3);

// Cabecera:
//std::cout.imbue(std::locale(std::cout.getloc(), new comma_separator));

	float time, potencia;
#ifdef PM_LIB
	pm_get_counter_data(&pmlib_counter); //J
	getMeasuresfromCounter(&time, &potencia, pmlib_counter);
	pm_finalize_counter(&pmlib_counter);
#endif

  // Remove filters from pipeline before they are implicitly destroyed.
  pipeline.clear(); 
	
	int frame_num = LAST_FRAME;
	
sonda.nframes = LAST_FRAME;

	_printf_sonda(sonda, sonda.BB.tini[1]);

	printf("Tiempo_medio_Infer: %f ms * Potencia_media: %f W * Energía_Infer: %f J\n",time/((frame_num-25)/5)*1000,potencia,time/((frame_num-25)/5)*potencia);

	fprintf(stdout,"----------\n");

	_printf_stats(sonda.VIN,"VIN",sonda.nframes);	cout << " ----------------------- " << endl;
	_printf_stats(sonda.OF,"OF",sonda.nframes); 	cout << " ----------------------- " << endl;
	_printf_stats(sonda.BB,"BB",sonda.nframes);		cout << " ----------------------- " << endl;
	_printf_stats(sonda.CNN,"CNN",sonda.nframes);	cout << " ----------------------- " << endl;
	_printf_stats(sonda.PRE,"PRE",sonda.nframes);	cout << " ----------------------- " << endl;	

}
