# jetson

Code examples for jetson nano. The same gait identification algorithm can be runned using different heterogeneous configurations, resulting in a variety of execition time and energy consmputions

Prerrequisites
- OpenCV 3.x
- Caffe
- For the three last versions also TBB is required

Folders
- gait: model and test video
- include, lib: caffe and energy measuring libraries: pmlib (UCM) and energy_meter (UMA)

Files
- compila_opencv.sh: command line to compile the code
- optflowgf.cpp: replace this file with the original one in opencv3.X to compute the Optical Flow using multiple threads. Recompile with:
`make opencv_video` and copy the updated lib to the corresponding opencv folder

Code versions:
* gait_CPU_test.cpp: all the computation is performed in the CPU
* gait_GPU_test.cpp: the main kernels are executed in the GPU
* gait_Threads_Read.cpp: computation is distributed between CPU and GPU. Best results in time and energy
* gait_Pipe.cpp: TBB pipeline. Kernel nodes can be executed by CPU or GPU, depending on the parameters given
* gait_graph_sample.cpp: TBB graph. Kernel nodes can be executed by CPU or GPU, depending on the parameters given. Each kernel process a sample (5 consecutive frames)
* gait_TBB-graph_frames.cpp: TBB graph. Kernel nodes can be executed by CPU or GPU, depending on the parameters given. Each kernel process a single frame
