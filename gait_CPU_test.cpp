#include <numeric> // accumulate

// Adaptacion de gait_CPU para hacer tests de GPU para los parametros del OF. Trabaja por secuecias

#define F_PROTO "/home/jetson/gait/tumcaffe_deploy.prototxt"
#define F_MODEL "/home/jetson/gait/tumcaffe.caffemodel"
#define F_LABEL "/home/jetson/gait/tumgaidtestids.lst"
#define F_VIDEO "/home/jetson/gait/p%03d-n05.avi"

#define OF_SCALE 1.0
// float OF_SCALE; /* Para reajustar tamaño OF entrada (reducir complejidad) */
#define LASTFRAME 525 /* Para garantizar 100 Inferencias */
int LAST_FRAME; // Ojo, los arrays suponen maximo 1025 frames!
#define MAX_FRAMES 1025

// #define DEBUG

#include "tbb/tick_count.h"
#include <omp.h>
#include <sys/sysinfo.h>

#include <opencv2/video/background_segm.hpp>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>

// Medir consumo con pmlib y Beaglebone UCM
// #include <omp.h>
// #define PM_LIB
// Medir consumo con pmlib y Beaglebone UCM
#ifdef PM_LIB
	#include "pmlib.h"
#endif
// Medir consumo con la libreria que adapte de Andres para Odroid
#include "energy_meter.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;
using namespace caffe;

#include "medidas.h"

t_sonda sonda;

int main(int argc, char **argv)
{
	// Lectutra de PARAMETROS
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0]
		// << " OFTasks BBTasks CNTasks LAST_FRAME" << std::endl;
		<< " OFTasks BBTasks CNTasks index" << std::endl;
		return 1;
	} // deploy.prototxt network.caffemodel labels video 1/scale 

	int omp_OF = atoi(argv[1]);
	int opencv_BB = atoi(argv[2]);
	int openblas_CN = atoi(argv[3]);
	int index = atoi(argv[4]);

	cv::setNumThreads(opencv_BB);
	omp_set_num_threads(omp_OF);
	openblas_set_num_threads(openblas_CN);
	Caffe::set_mode(Caffe::CPU);
		
	int frame_num = 0;
	LAST_FRAME = (argc == 5) ?  atoi(argv[4]) : LASTFRAME;
	// OF_SCALE =  atof(argv[5]);
	
	vector<int> aciertos;
	int npred = 0;

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

	struct energy_sample *sample;
	sample=energy_meter_init(/*sr*/ 50, 0 /*debug*/);  // sr: sample rate in miliseconds (dificil bajar de 5)
	// 50: bien los tiempos con CPU y consumos:
	// Tiempo_medio_Infer: 1305.880831 ms * Potencia_media: 4.473857 W * Energa_Infer: 5.842325 J
	// Con 4 cores, casos los extremos
	// - 0 ms : mucho más tiempo, media de energia bastante menor que lo esperado
	// Ej: Tiempo_medio_Infer: 1740.965343 ms * Potencia_media: 4.145585 W * Energa_Infer: 7.217319 J
	// - 1 seg: tiempo correcto, pero se le escapa mucha energia
	// Tiempo_medio_Infer: 1292.756455 ms * Potencia_media: 0.152943 W * Energa_Infer: 0.197718 J
	// Resultados con Beagle (de referencia, con consumo ligeramente superior al medir con Rshunt)
	// Tiempo_medio_Infer: 1275.343384 ms * Potencia_media: 4.884373 W * Energa_Infer: 6.229253 J
	
	// Informacion para el LOG
	string line1;
    ifstream entrada("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    getline(entrada,line1); cout << "fCPU: " << line1 << " - "; entrada.close();
	entrada.open("/sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq");
    getline(entrada,line1); cout << "fGPU: " << line1 << " - "; entrada.close();	
    getline(entrada,line1); cout << "ncores_on: " << get_nprocs()  << endl; entrada.close();
	fprintf(stdout,"CPU_Sample * Threads OF: %d - BB: %d - CNN: %d\n",omp_OF,opencv_BB,openblas_CN);

// fprintf(stdout,"----------\n");

	// Input Video file name
	// string videoFileName = F_VIDEO;
	
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
	
	// OF CPU variables
	Mat prvs, next, flow;
	Mat flow_x_y, flujo_crop;	// Resultado downloaded de GPU -> para generar luego el vector ...
	Mat flow_res[2];
	vector<Mat> flow_vec;
	Ptr<cv::FarnebackOpticalFlow> fbOF = // Parametros del articulo original del GAIT
		cv::FarnebackOpticalFlow::create(5,sqrt(2)/2.0,false,10,2,7,1.5,
										 cv::OPTFLOW_FARNEBACK_GAUSSIAN); // cv::OPTFLOW_USE_INITIAL_FLOW);

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

	// Primer frame de la secuencia para OF (prvs - BN) y BB (GetImg - color)
	char videoFileName[256];
	snprintf(videoFileName, 256, F_VIDEO, index);
	VideoCapture stream(videoFileName); // Video file open
	if(!(stream.read(GetImg))) // get one frame from video
		return -1;
 	cv::cvtColor(GetImg, prvs, COLOR_BGR2GRAY);

	//--- Inicializar modelo de fondo con primer frame leido
	bg_model->apply(prvs, fgmask, updateBGModel ? -1 : 0); // -1: automatically select the learning rate
	
	sonda.muestras = false; // Para indicar que trabajamos a nivel de frame

	for (int i=1; i<=25; i++) { // Fill up the oflow buffer (25 x-, y- samples)

_start_sonda(&(sonda.VIN), i, false);
		if(!(stream.read(GetImg))) // get one frame form video   
			return -1;
		cv::cvtColor(GetImg, next, COLOR_BGR2GRAY); //prev_res , next_res
_stop_sonda(&(sonda.VIN), i);

		// flow = Mat::zeros(GetImg.size(), CV_32FC2);                                           

		frame_num++; 

_start_sonda(&(sonda.BB), i, false);
		bg_model->apply(next, fgmask, updateBGModel ? -1 : 0); // -1: automatically select the learning rate		
		morphologyEx(fgmask, fgimg, MORPH_OPEN, strel);
		Rect BB = boundingRect(fgimg);
_stop_sonda(&(sonda.BB),i);

		int izq = BB.x/8 + BB.width/(8*2); // - 30 (centro a izq) + 30 (borde) (ensancho luego por los bordes +60; // Para la escala de /8 (queda imagen de 60x80; sino, no sirve)
		BBox_izq.push_back(izq);

_start_sonda(&(sonda.OF), i, false);
        fbOF->calc(prvs, next, flow);
		cv::resize(flow, flow_x_y, Size(80,60));
_stop_sonda(&(sonda.OF),i);

		prvs = next.clone();

        cv::split(flow_x_y, flow_res);
		flow_vec.push_back(flow_res[0].clone());
		flow_vec.push_back(flow_res[1].clone());

	}

	// Antes de llamar a CAFE hay que recortar con BB central todos los OF
	// para copiarlos en la capa de entrada de la Red
	int central = BBox_izq[12]; // 13º, el central de los 25 (0-24)
	ROI = Rect(central, 0, 60, 60); // Origen TOP-LEFT wx, wy

_start_sonda(&(sonda.CNN), 25, false);
	bufferBB = input_layer->mutable_cpu_data();
	for (int i=0; i<2*25; i++) {
		cv::copyMakeBorder(flow_vec[i], flujo_crop,0,0,30,30,BORDER_CONSTANT, Scalar(0.0f));
		Mat imgROI = 10*flujo_crop(ROI); // Escalar el OF x 10 (faltaria sumar media de la BD ~0.00...)
		memcpy(bufferBB+(60*60*(i)),imgROI.data,60*60*sizeof(float));
	}

	// Hacer primera prediccion con Caffe y mostrar resultados
	net->Forward();
_stop_sonda(&(sonda.CNN),25); // fin primer buffer completo

// Eliminar los 5 del final para que los coloquen luego los threads
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

	// Calcula nuevos OF (desplaza el buffer y encaja la nueva muestra al final)
	// -> Ahora los Threads los meteran en su hueco correspondiente 20..24
	////////////////////
	
	while (true) { // To the end of the video -> LAST_FRAME are processed
	
		// if (frame_num==LAST_FRAME) break; // Solo proceso los LAST_FRAME primeros frames

		if (frame_num==25)  {
#ifdef PM_LIB
			pm_start_counter(&pmlib_counter);
#endif
			energy_meter_start(sample);  // starts sampling thread
		}
		
_start_sonda(&(sonda.PRE), frame_num+1 , false);
		
_start_sonda(&(sonda.VIN), frame_num+1, false);
		if(!(stream.read(GetImg))) {//get one frame form video
			break;
			stream.set(CAP_PROP_POS_FRAMES, 0);// start (rewind)
			if (!(stream.read(GetImg))) { cout << "No pudo rebobinar " << endl; }
		}
		cv::cvtColor(GetImg, next, COLOR_BGR2GRAY);
_stop_sonda(&(sonda.VIN),frame_num+1);

		frame_num++;

_start_sonda(&(sonda.BB), frame_num, false);
		bg_model->apply(next, fgmask, updateBGModel ? -1 : 0); // -1: automatically select the learning rate
        morphologyEx(fgmask, fgimg, MORPH_OPEN, strel);
		Rect BB = boundingRect(fgimg); 
_stop_sonda(&(sonda.BB),frame_num);
		
		int izq = BB.x/8 + BB.width/(8*2);
		BBox_izq.push_back(izq); // Cambiar a BBox_izq(20+n_muestra)

_start_sonda(&(sonda.OF), frame_num, false);
        fbOF->calc(prvs, next, flow);
		cv::resize(flow, flow_x_y, Size(80,60));
_stop_sonda(&(sonda.OF),frame_num);

 		prvs = next.clone();

		cv::split(flow_x_y, flow_res);
		flow_vec.push_back(flow_res[0].clone());
		flow_vec.push_back(flow_res[1].clone());	

		// A partir de tener llemos ya las 25 muestras (frames) -> se hara de 5 en 5
		BBox_izq.erase(BBox_izq.begin()); 
		flow_vec.erase(flow_vec.begin()); // OF x
		flow_vec.erase(flow_vec.begin()); // OF y	
				
		if (frame_num%5==0) { // Cada 5 muestras
		
			// Antes de llamar a CAFE hay que recortar con BB central todos los OF
			izq = BBox_izq[12]; // .x/8 + BBoxes[frame_num-13].width/(8*2) - 30 + 60;	// Central de los 25 primeros (0-24)
			ROI = Rect(izq, 0, 60, 60);

_start_sonda(&(sonda.CNN), frame_num, false);
			bufferBB = input_layer->mutable_cpu_data();
			for (int i=0; i<2*25; i++) {
				cv::copyMakeBorder(flow_vec[i], flujo_crop,0,0,30,30,BORDER_CONSTANT, Scalar(0.0f)); // Se podria hacer en GPU!!!
				Mat imgROI = 10*flujo_crop(ROI); // Escalar el OF x 10 (faltaria sumar media de la BD ~0.00...)
				memcpy(bufferBB+(60*60*(i)),imgROI.data,60*60*sizeof(float));
			}
		
			net->Forward();
_stop_sonda(&(sonda.CNN),frame_num);

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
	
		}  // cada 5 frames
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

	struct timespec res = diff(sample->start_time, sample->stop_time);
	double total_time = (double)res.tv_sec+ (double)res.tv_nsec/1000000000.0;
	int n_infer = (frame_num-25)/5; // sonda.nframes (se quita la primera que es diferente)

	cout << "N_infer " << n_infer << " - n_samples_inf " << sample->samples/n_infer << " - sampling_rate ms " 
		 << total_time*1000/sample->samples << endl;

	cout << "ENERGIAxINFER mJ " << sample->TOT/n_infer << " = CPU " << sample->CPU/n_infer << " + GPU " << sample->GPU/n_infer
		 << " + Others " << (sample->TOT-(sample->CPU+sample->GPU))/n_infer << endl;

	printf("Tiempo_medio_Infer: %f ms * Potencia_media: %f W * Energía_Infer: %f J\n",
			total_time/n_infer*1000,
			(double)(sample->TOT)/(total_time*1000),
			sample->TOT/(n_infer*1000));

	energy_meter_destroy(sample);     // clean up everything
	
	
	stream.release();
	
	return 0;

}
