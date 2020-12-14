# jetson nano

Code examples for jetson nano. The same gait identification algorithm [1] can be runned using different heterogeneous configurations, resulting in a variety of execition time and energy consmputions

Prerrequisites
- OpenCV 3.x
- Caffe
- For the three last versions also TBB is required

Folders
- gait: model and test video
- include, lib: caffe and energy measuring libraries: pmlib (UCM) and energy_meter (UMA)

Files
- compila_opencv.sh: command line to compile the code
- optflowgf.cpp: replace this file with the original one in opencv3.X to compute the Optical Flow using multiple thr`eads. Recompile with:
`make opencv_video` and copy the updated lib to the corresponding opencv folder

Code versions (execute without paramaters to get help). Example:
* gait_Threads_Read.cpp: computation is distributed between CPU and GPU. Best results in time and energy

`gait_Threads_Read OFTasks BBTasks CNTasks LAST_FRAME`

-`OFTasks`: number of Optical Flow threads,

-`BBTasks`: number of Bounding Box computation threads,

-`CNTasks`: number of CNN inference threads,

-`LAST_FRAME`: last frame to process (to get measures and statistics). If there are less frames in the video it is rewinded after reaching the end.

The rest of parameters can be configured editing the source code.

* gait_CPU_test.cpp: all the computation is performed in the CPU
* gait_GPU_test.cpp: the main kernels are executed in the GPU
* gait_Pipe.cpp: TBB pipeline. Kernel nodes can be executed by CPU or GPU, depending on the parameters given
* gait_graph_sample.cpp: TBB graph. Kernel nodes can be executed by CPU or GPU, depending on the parameters given. Each kernel process a sample (5 consecutive frames)
* gait_TBB-graph_frames.cpp: TBB graph. Kernel nodes can be executed by CPU or GPU, depending on the parameters given. Each kernel process a single frame

[1] _Automatic learning of gait signatures for people identification_, FM Castro, MJ Marín-Jiménez, N Guil, NP de la Blanca. International Work-Conference on Artificial Neural Networks, pp. 257-270.
