# Jeston NANO
# Dinamico, sin debug * Ademas quito: -fno-inline -fno-omit-frame-pointer -> incluyo pmlib
g++ $1.cpp -O3 -o $1 -lcaffe -lpm -lenergy_meter_arm -Wno-unused-result -fopenmp -ltbb -lpthread -lrt -lm -ldl -lz `pkg-config --cflags --libs opencv` -I ~/include -I /usr/local/opencv3/include -I ~/src/caffe-master/include/ -I /usr/local/cuda/include/ -L ~/lib -L /usr/local/cuda/lib/ -L /usr/local/opencv3/lib -std=c++11 -lglog -lboost_system -Wno-write-strings -lopenblas # -rpath-link /usr/local/opencv3/lib

# Warning Beaglebone

# No reconocidas/utiles -marm -mfpu=neo -march=armv7-a n 


# Ultimo usado en TK1
# Dinamico, sin debug * Ademas quito: -fno-inline -fno-omit-frame-pointer -> incluyo pmlib
#g++ $1.cpp -O3 -marm -mfpu=neon -march=armv7-a -o $1 libcaffe.so streamline_annotate.o pmlib.a -Wno-unused-result -ltbb -lpthread -lrt -lm -ldl -lz `pkg-config --cflags --libs opencv_so` -I ~/caffe/include/ -I /usr/local/cuda/include/ -L /usr/local/cuda/lib/ -lcudart -std=c++11 -lglog -Wno-write-strings # Warning Beaglebone

