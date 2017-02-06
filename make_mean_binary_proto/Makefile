#!/bin/sh

#Refer to
#"caffe/build/examples/CMakeFiles/classification.dir/link.txt"
#for lingking option
#"caffe/build/examples/CMakeFiles/classification.dir/flags.make"
#for compiler option
#"caffe/build/examples/CMakeFiles/classification.dir/build.make"
#for compile command

DEBUG_SS = n
TIME_SS = y

SRCS = convert_image_to_binaryproto.cpp
OBJS = $(addsuffix .o, $(basename $(SRCS)))

CAFFE_INC_DIR = -I/home/mg6/caffe/build/include -I/home/mg6/caffe/include -I/home/mg6/caffe/build
CAFFE_LIB_DIR = -L/home/mg6/caffe/build/lib/
CAFFE_LIB = -lcaffe

OPENCV_INC_DIR = `pkg-config --cflags-only-I opencv`
OPENCV_LIB_DIR = `pkg-config --libs-only-L opencv`
OPENCV_LIBS = `pkg-config --libs-only-l opencv`

GLFW_INC_DIR = `pkg-config --cflags-only-I glfw3`
GLFW_LIB_DIR = `pkg-config --libs-only-L glfw3`
GLFW_LIBS = `pkg-config --libs-only-l glfw3`
CXX = /usr/bin/c++

CXX_FLAGS =  -fPIC -Wall -Wno-sign-compare -Wno-uninitialized -O3 -DNDEBUG  
CXX_DEFINES = -DCAFFE_VERSION=1.0.0-rc3 -DGTEST_USE_OWN_TR1_TUPLE -DUSE_LEVELDB -DUSE_LMDB -DUSE_OPENCV -DWITH_PYTHON_LAYER 

CXX_INC_DIR = -isystem /usr/include/hdf5/serial -isystem /usr/local/cuda-8.0/include -isystem /usr/local/include -isystem /usr/local/include/opencv -isystem /usr/include/python2.7 -isystem /home/ryota/.local/lib/python2.7/site-packages/numpy/core/include -I/home/ryota/caffe/build/include -I/home/ryota/caffe/include -I/home/ryota/caffe/build $(OPENCV_INC_DIR) $(GLFW_INC_DIR) $(CAFFE_INC_DIR)
LDLIBS = -lcaffe -lproto -lboost_system -lboost_thread -lboost_filesystem -lboost_chrono -lboost_date_time -lboost_atomic -lpthread -lpthread -lglog -lgflags -lprotobuf /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so -lsz -lz -ldl -lm -lpthread -lpthread -lglog -lgflags -lprotobuf /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so -lsz -lz -ldl -lm -llmdb -lleveldb -lsnappy /usr/local/cuda-8.0/lib64/libcudart.so /usr/local/cuda-8.0/lib64/libcurand.so /usr/local/cuda-8.0/lib64/libcublas.so -lpython2.7 -lboost_python $(OPENCV_LIBS) $(GLFW_LIBS)

LDFLAGS = -rdynamic -L/usr/local/cuda-8.0/lib64 -Wl,-rpath,/usr/local/cuda-8.0/lib64:/home/ryota/caffe/build/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/usr/local/lib:::::::: -Wl,-rpath-link,/usr/local/lib $(OPENCV_LIB_DIR) $(GLFW_LIB_DIR) $(CAFFE_LIB_DIR) $(CAFFE_LIB)

all: $(OBJS)
	$(CXX) $(CXX_DEFINES) $(CXX_INC_DIR) $(CXX_FLAGS) $(OBJS) $(LDFLAGS) $(LDLIBS) -o convert_image_to_binaryproto

%.o: %.cpp
	$(CXX) $(CXX_DEFINES) $(CXX_INC_DIR) $(CXX_FLAGS) -o $@ -c $^

.PHONY: clean
clean:
	rm -f *.o
	rm -f ./convert_image_to_binaryproto
