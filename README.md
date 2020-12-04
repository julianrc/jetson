# jetson
Code examples for jetson nano

Prerrequisites:
- OpenCV 3.x
- Caffe

Folders
- gait: model and test video
- include, lib: no description required

Files
- compila_opencv.sh: command line to compile the code
- optflowgf.cpp: replace this file with the original one in opencv3.X to compute the Optical Flow using multiple threads. Recompile with:
`make opencv_video` and copy the updated lib to the corresponding opencv folder

Code versions:
* gait_CPU_test.cpp: all the computation is performed in the CPU
* gait_GPU_test.cpp: the main kernels are executed in the GPU
* gait_Threads_Read.cpp: computation is distributed between CPU and GPU
