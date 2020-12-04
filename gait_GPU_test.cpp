// ./gait_GPU_test -w=13 -i=2 -N=7 -s=1.5 -g=256 -p=0.707106781 3

// Adaptacion de gait_GPU_video para hacer tests de GPU para los parametros del OF

#define F_PROTO "/home/jetson/gait/tumcaffe_deploy.prototxt"
#define F_MODEL "/home/jetson/gait/tumcaffe.caffemodel"
#define F_LABEL "/home/jetson/gait/tumgaidtestids.lst"
#define F_VIDEO "/home/jetson/gait/p%03d-n05.avi"
#define OF_SCALE 1.0
// float OF_SCALE; /* Para reajustar tamaño OF entrada (reducir complejidad) */
#define LASTFRAME 85 /* Para garantizar 100 Inferencias */
int LAST_FRAME; // Ojo, los arrays suponen maximo 1025 frames!
#define MAX_FRAMES 1025

// #define DEBUG

#include "tbb/tick_count.h"

#include <caffe/caffe.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudawarping.hpp> 	// resize
#include <opencv2/cudaarithm.hpp> 	// split
#include <opencv2/cudafilters.hpp> 

#include <cuda.h>
#include <cuda_runtime.h>


// #define PM_LIB
// Medir consumo con pmlib y Beaglebone UCM
#ifdef PM_LIB
	#include "pmlib.h"
	#include <sys/sysinfo.h>
#endif
// Medir consumo con la libreria que adapte de Andres para Odroid
#include "energy_meter.h"

using namespace caffe;
using namespace cv;
using namespace cv::cuda;
using namespace std;

#include "medidas.h"

t_sonda sonda;

tbb::tick_count t0;

int main(int argc, char **argv)
{	
/*
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0]
		<< "LAST_FRAME" << std::endl;
		return 1; // deploy.prototxt network.caffemodel labels 1/scale openBlasTasks openCVTasks video
	}
*/
	// LAST_FRAME = (argc == 2) ?  atoi(argv[1]) : LASTFRAME;
	LAST_FRAME = LASTFRAME;
	
    const String keys = // <none> -> no default
        "{ help h usage ? |      | print this message   }"
        "{ @index         | 3    | index of tumgait img }"
        "{ L numlevels    | 5    | pyr num of levels    }"
        "{ p pyrscale     | 0.5  | pyr scale            }"
        "{ f fast         |      | fastPyramids        }"
        "{ w winsize      | 10   | win size             }"
        "{ i numiter      | 10   | num iters            }"
        "{ N polyN        | 5    | polyN                }"
        "{ s polysigma    | 1.1  | poly sigma           }"
        "{ g flags        | 0    | OF flags ( 256 => OPTFLOW_FARNEBACK_GAUSSIAN) }";

	CommandLineParser cmd(argc, argv, keys);
	cmd.about("OF GPU tests   v1.0.0");
	if (cmd.has("help"))
	{
		cmd.printMessage();
		return 0;
	}
	
	// create (int numLevels=5, double pyrScale=0.5, bool fastPyramids=false, int winSize=13, int numIters=10, int polyN=5, double polySigma=1.1, int flags=0)
	
	int index = cmd.get<int>(0); // image index, first positional parameter
	int numLevels = cmd.get<int>("L");
	float pyrScale = cmd.get<float>("pyrscale");
	bool fastPyramids = cmd.has("fast");
	int winSize = cmd.get<int>("winsize");
	int numIters = cmd.get<int>("numiter");
	int polyN = cmd.get<int>("polyN");
	float polySigma = cmd.get<float>("s");
	int flags = cmd.get<int>("g");
	
	if (!cmd.check())
	{
		cout << "CHECK " << endl;
		cmd.printErrors();
		return 0;
	}

	// Imprimir parametros del OF
	cout << "Subject ID : " << index << endl;
	cout << "numLevels: " << numLevels << " - pyrScale: " << pyrScale << " - fastPyramids: " << fastPyramids << endl;
	cout << "winSize: " << winSize << " - numIters: " << numIters << " - polyN: " 
		 << polyN << " - polySigma " << polySigma << " - flags: " << flags << endl;

	int frame_num = 0;
	vector<int> aciertos;
	int npred = 0;

	// Init Caffe	

	Caffe::set_mode(Caffe::GPU);

	//OF_SCALE =  atof(argv[5]);

	// INICIAR MEDIDAS BEAGLE
	// Some variables to use pmlib
	tbb::tick_count tfinP, tfinU;

#ifdef PM_LIB
	server_t servidor;
	counter_t pmlib_counter;
	line_t lineas;
	LINE_CLR_ALL(&lineas);
	LINE_SET( 7, &lineas ); // Solo uso esta
	pm_set_server("192.168.188.218", 6526, &servidor); // beagle2
	pm_create_counter("INA219Device", lineas, 1, 1000, servidor, &pmlib_counter);
#endif

	struct energy_sample *sample;
	sample=energy_meter_init(/*sr*/ 1, 0 /*debug*/);  // sr: sample rate in miliseconds (dificil bajar de 10)
	// Al no usar apenas más de una CPU podemos apurar la medida sin sobrecargar

	// Mostrar fCPU - fGPU
/*
	string line1;
    ifstream entrada("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    getline(entrada,line1); cout << "fCPU: " << line1 << " - "; entrada.close();
	entrada.open("/sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq");
    getline(entrada,line1); cout << "fGPU: " << line1 << " - "; entrada.close();	
    getline(entrada,line1); cout << "ncores_on: " << get_nprocs()  << endl; // entrada.close();

	fprintf(stdout,"----------\n");
*/

	::google::InitGoogleLogging(argv[0]);
	// ::google::SetCommandLineOption("GLOG_minloglevel", "2"); // 2-error, 3-fatal
	FLAGS_minloglevel = 3; // Al log solo errores fatales
	boost::shared_ptr<Net<float> > net;
	
	/* Load labels */
	std::vector<string> labels;
	std::ifstream labelsf(F_LABEL);
	string line;
	while (std::getline(labelsf, line))
		labels.push_back(string(line));
	
	/* Load the network. */
	net.reset(new Net<float>(F_PROTO, TEST));
	net->CopyTrainedLayersFrom(F_MODEL);

	Blob<float>* input_layer = net->input_blobs()[0];
	Blob<float>* output_layer = net->output_blobs()[0];

	int minArea = 1000; // No lo uso, pero debe eliminar BB pequeños
	bool updateBGModel = true;
    Mat GetImg, fgmask, fgimg;
	GpuMat d_fgmask, d_fgimg;
	
	// (history, varThreshold, detectShadows)
	Ptr<cuda::BackgroundSubtractorMOG2> bg_model = 
		cuda::createBackgroundSubtractorMOG2(40, 30, true);
	bg_model->setNMixtures(10); /* numMixtures */ 
	Mat strel = getStructuringElement(MORPH_RECT, Size(3,3));
	Ptr<cuda::Filter> openFilter = 
		cuda::createMorphologyFilter(MORPH_OPEN, d_fgmask.type(), strel);
	
	// CPU variables
	Mat flow_res;
	Mat flow_x_y; // Resultado downloaded de GPU -> para generar luego el vector ...
	Mat flows[2], flujo_crop, imgROI; // subir y renombrar ¿?

	// GPU variables
	GpuMat prvs_gpu, next_gpu, flow_gpu;
 	
	// Dense GPU optical flow creation and configuration. Same parameters as in w-Flow / libcudawFlow
	// FarnebackOpticalFlow::create	(numLevels = 5, pyrScale = 0.5, fastPyramids = false, winSize = 13,	numIters = 10,
	//								 polyN = 5, polySigma = 1.1, flags = 0)
	//	cv::cuda::FarnebackOpticalFlow::create(); //(5,sqrt(2)/2.0,false,10,2,7,1.5,cv::OPTFLOW_FARNEBACK_GAUSSIAN);
	// SE RECOMIENDAN OTROS PARAMETROS LIGERAMENTE DIFERENTES !!
	cv::Ptr<cv::cuda::FarnebackOpticalFlow> fbOF = 
		cv::cuda::FarnebackOpticalFlow::create 
			(numLevels, pyrScale, fastPyramids, winSize, numIters, polyN, polySigma, flags);
			// cv::OPTFLOW_FARNEBACK_GAUSSIAN); //-> 256 // | cv::OPTFLOW_USE_INITIAL_FLOW); // -> 1

	// BUFFERS
	float* bufferBB; // = input_layer->mutable_cpu_data(); // Post-corte BB (filas, columnas, canalx/y, muestras)
	vector<int> BBox_izq;
	vector<Mat> flow_vec;
	Rect ROI;
	
	// File name
	char videoFileName[256];
	snprintf(videoFileName, 256, F_VIDEO, index);

	Mat GetImgC;

	// Video file open
	VideoCapture stream(videoFileName);
	if(!(stream.read(GetImgC))) //get one frame from video
		return -1;
		
    cv::cvtColor(GetImgC, GetImg, COLOR_BGR2GRAY);	
	prvs_gpu.upload(GetImg);

	//--- Inicializar modelo de fondo con primer frame leido
	bg_model->apply(prvs_gpu, d_fgmask, updateBGModel ? -1 : 0); // -1: automatically select the learning rate
	
	//
	// INICIALIZACION (llenado inicial del buffer) 
	/////////////////

	sonda.muestras = false; // Para indicar que trabajamos a nivel de frame
	for (int i=1; i<=25; i++) { // Fill up the flow buffer (25 x-, y- samples)

_start_sonda(&(sonda.VIN), i, false);
		if(!(stream.read(GetImgC))) //get one frame form video   
			return -1; // No se ha podido iniciar el buffer con los 25 primeros 
		cv::cvtColor(GetImgC, GetImg, COLOR_BGR2GRAY);
_stop_sonda(&(sonda.VIN), i);
			
		frame_num++;
		
// BG-GPU
		next_gpu.upload(GetImg);
_start_sonda(&(sonda.BB), i, true);
		bg_model->apply(next_gpu, d_fgmask, updateBGModel ? -1 : 0);
		openFilter->apply(d_fgmask, d_fgimg);
		d_fgimg.download(fgimg);
		Rect BB = boundingRect(fgimg);
_stop_sonda(&(sonda.BB),i);

		int izq = BB.x/8 + BB.width/(8*2);
		BBox_izq.push_back(izq);

_start_sonda(&(sonda.OF), i, true);
		fbOF->calc(prvs_gpu, next_gpu, flow_gpu); // versiones sin escalar
		flow_gpu.download(flow_res);	
		cv::resize(flow_res, flow_x_y, Size(80,60));
_stop_sonda(&(sonda.OF),i);

		cv::split(flow_x_y, flows); // split channels into two different images
		prvs_gpu = next_gpu.clone();// !!! CREO QUE SI ES POSIBLE CLONAR EN GPU....

		flow_vec.push_back(flows[0].clone());
		flow_vec.push_back(flows[1].clone());
	}

	// Antes de llamar a CAFE hay que recortar con BB central todos los OF
	int izquierda = BBox_izq[12]; // Central de los 25 (0-24)
	ROI = Rect(izquierda, 0, 60, 60);

_start_sonda(&(sonda.CNN), 25, false);
	bufferBB = input_layer->mutable_cpu_data();
	for (int i=0; i<50; i++) { 
		cv::copyMakeBorder(flow_vec[i], flujo_crop,0,0,30,30,BORDER_CONSTANT, Scalar(0.0f)); // Se podria hacer en GPU!!!
		imgROI = 10*flujo_crop(ROI); // Escalar el OF x 10 (faltaria sumar media de la BD 0.00...) (necesario el x10?)
		memcpy(bufferBB+(60*60*(i)),imgROI.data,60*60*sizeof(float));
	}
	
	// Hacer primera prediccion con Caffe y mostrar resultados
	net->Forward();
_stop_sonda(&(sonda.CNN),25); // fin primer buffer completo

	npred++;
	
#ifdef DEBUG	
{	
	vector<float> mi_salida;
	mi_salida.insert (mi_salida.begin(), output_layer->cpu_data(), output_layer->cpu_data()+output_layer->channels());
	int max_idx = arg_max(mi_salida);
	if (mi_salida[max_idx]>0.01 && (stoi(labels[max_idx])==index) ) 
		aciertos.push_back(npred);
}
#endif

	//
	// Calcula nuevos OF (desplaza el buffer y encaja la nueva muestra al final)
	////////////////////
	
	while (true) { // To the end of the video (LAST_FRAME)
	
		// OJO frame_num hace referencia a anterior iteracion (ya leido y procesado)
		if (frame_num==LAST_FRAME) break; // Solo proceso los LAST_FRAME primeros frames
		
		if (frame_num==25) {
			tfinP = tbb::tick_count::now();
#ifdef PM_LIB
			pm_start_counter(&pmlib_counter);
#endif
			energy_meter_start(sample);  // starts sampling thread
		}
		
_start_sonda(&(sonda.PRE), frame_num+1 , true);
		
_start_sonda(&(sonda.VIN), frame_num+1, false);
		if(!(stream.read(GetImgC))) { //get one frame form video
			break;
			stream.set(CAP_PROP_POS_FRAMES, 0);// start (rewind)
			if (!(stream.read(GetImgC))) { cout << "No pudo rebobinar " << endl; }
		}
		cv::cvtColor(GetImgC, GetImg, COLOR_BGR2GRAY);
_stop_sonda(&(sonda.VIN),frame_num+1);

		frame_num++;

		next_gpu.upload(GetImg);	
_start_sonda(&(sonda.BB), frame_num, true);
		bg_model->apply(next_gpu, d_fgmask, updateBGModel ? -1 : 0); 
		openFilter->apply(d_fgmask, d_fgimg);
		d_fgimg.download(fgimg);
		Rect BB = boundingRect(fgimg); 
_stop_sonda(&(sonda.BB),frame_num);

		int izq = BB.x/8 + BB.width/(8*2);
		BBox_izq.push_back(izq);

_start_sonda(&(sonda.OF), frame_num, true);
		fbOF->calc(prvs_gpu, next_gpu, flow_gpu);
		flow_gpu.download(flow_res);
		cv::resize(flow_res, flow_x_y, Size(80,60));
_stop_sonda(&(sonda.OF),frame_num);

		prvs_gpu = next_gpu.clone();

		cv::split(flow_x_y, flows); // split channels into two different images
		flow_vec.push_back(flows[0].clone());
		flow_vec.push_back(flows[1].clone());	

		// A partir de tener llemos ya las 25 muestras (frames)
		// (para implementar buffer circular con las ultimos 25 OFx,y)
		BBox_izq.erase(BBox_izq.begin()); // BB
		flow_vec.erase(flow_vec.begin()); // OF x
		flow_vec.erase(flow_vec.begin()); // OF y
					
		if (frame_num%5==0) { // Cada 5 muestras

			// Antes de llamar a CAFE hay que recortar con BB central todos los OF
			izquierda = BBox_izq[12]; //frame_num-13];//.x/8 + BBoxes[frame_num-13].width/(8*2) - 30 + 60;	// Central de los 25 primeros (0-24)
			ROI = Rect(izquierda, 0, 60, 60);

_start_sonda(&(sonda.CNN), frame_num, true);		
			bufferBB = input_layer->mutable_cpu_data();
			for (int i=0; i<50; i++) { 
				cv::copyMakeBorder(flow_vec[i], flujo_crop,0,0,30,30,BORDER_CONSTANT, Scalar(0.0f)); // Se podria hacer en GPU!!!
				imgROI = 10*flujo_crop(ROI); // Escalar el OF x 10 (faltaria sumar media de la BD 0.00...) (necesario el x10?)
				memcpy(bufferBB+(60*60*(i)),imgROI.data,60*60*sizeof(float));
			}

			net->Forward();
_stop_sonda(&(sonda.CNN),frame_num);
			npred++;
	// Show results (for debugging only)
#ifdef DEBUG
{	
	vector<float> mi_salida;
	mi_salida.insert (mi_salida.begin(), output_layer->cpu_data(), output_layer->cpu_data()+output_layer->channels());
	int max_idx = arg_max(mi_salida);
	if (mi_salida[max_idx]>0.01 && (stoi(labels[max_idx])==index) )
		aciertos.push_back(npred);
}
#endif

		} // cada 5 frames
_stop_sonda(&(sonda.PRE),frame_num);

	} // frames
	
#ifdef PM_LIB
	pm_stop_counter(&pmlib_counter);		
	float time, potencia;
	pm_get_counter_data(&pmlib_counter); //J
	getMeasuresfromCounter(&time, &potencia, pmlib_counter);
	pm_finalize_counter(&pmlib_counter);
#endif
	energy_meter_stop(sample);  	// stops sampling	
	// energy_meter_printf(sample, stderr);  // print total results
tfinU = tbb::tick_count::now();

#ifdef DEBUG
	fprintf(stdout,"----------\n");
	cout << "Predicciones: " << aciertos.size() << " - " << npred << " * ";
	for (unsigned i=0; i<aciertos.size(); i++)
		std::cout << ' ' << aciertos[i];
	cout << endl;
#endif

	fprintf(stdout,"----------\n");

sonda.nframes = frame_num;

	_printf_sonda(sonda, sonda.BB.tini[1]);

#ifdef PM_LIB
	printf("Tiempo_medio_Infer: %f ms * Potencia_media: %f W * Energía_Infer: %f J\n",time/((frame_num-25)/5)*1000,potencia,time/((frame_num-25)/5)*potencia);
#endif

	fprintf(stdout,"----------\n");
 
	_printf_stats(sonda.VIN,"VIN",sonda.nframes);	cout << " ----------------------- " << endl;
	_printf_stats(sonda.OF,"OF",sonda.nframes); 	cout << " ----------------------- " << endl;
	_printf_stats(sonda.BB,"BB",sonda.nframes);		cout << " ----------------------- " << endl;
	_printf_stats(sonda.CNN,"CNN",sonda.nframes);	cout << " ----------------------- " << endl;
	_printf_stats(sonda.PRE,"PRE",sonda.nframes);	cout << " ----------------------- " << endl;

	fprintf(stdout,"----------\n");

// energy_meter measurements:
	struct timespec res;
	res=diff(sample->start_time, sample->stop_time);
	double total_time = (double)res.tv_sec+ (double)res.tv_nsec/1000000000.0;
	// printf("Tiempo TBB: %lf\n", (tfinU-tfinP).seconds()*1000);

	int n_infer = (frame_num-25)/5; // sonda.nframes (se quita la primera que es diferente)

cout << "N_infer " << n_infer << " - n_samples_inf " << sample->samples/n_infer << " - sampling_rate ms " 
     << total_time*1000/sample->samples << endl;

cout << "ENERGIAxINFER mJ " << sample->TOT/n_infer << " = CPU " << sample->CPU/n_infer << " + GPU " << sample->GPU/n_infer
	 << " + Others " << (sample->TOT-(sample->CPU+sample->GPU))/n_infer << endl;
	 
// cout << "ENERGY TOTAL " << sample->TOT << " = CPU " << sample->CPU << " + GPU " << sample->GPU
	 // << " + Others " << sample->TOT-(sample->CPU+sample->GPU) << endl;
// printf("CPU total energy measured= %lf Joules\n", sample->CPU );  // energy is in Joules
// printf("GPU total energy measured= %lf Joules\n", sample->GPU );  // energy is in Joules
// printf("TOTAL energy measured= %lf Joules\n", sample->TOT );  // energy is in Joules
// printf("Energy without CPU & GPU = %lf\n", sample->TOT-(sample->CPU+sample->GPU));
// printf("Tiempo_medio_Infer: %lf ms - %lf - n_infer %d \n", total_time/n_infer*1000,total_time,n_infer);

printf("Tiempo_medio_Infer: %f ms * Potencia_media: %f W * Energía_Infer: %f J\n",
		total_time/n_infer*1000,
		(double)(sample->TOT)/(total_time*1000),
		sample->TOT/(n_infer*1000));

	// printf("Watios medios %0.4f \n", (double)(sample->TOT)/(total_time*1000));
    // printf("CLOCK_REALTIME = %lf sec\n",(double)res.tv_sec+ (double)res.tv_nsec/1000000000.0);
    // printf("# of samples: %ld\n", sample->samples);
    // printf("sample every (real) = %lf sec\n",((double)res.tv_sec+ (double)res.tv_nsec/1000000000.0)/sample->samples);

energy_meter_destroy(sample);     // clean up everything

	stream.release();	
	
	return 0;

}
