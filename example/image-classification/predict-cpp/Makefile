# Special thanks to https://github.com/pertusa for the Makefile
CFLAGS=-std=c++11 -Wno-unknown-pragmas -Wall

# Added for openblas
# export OPENBLAS_ROOT=/usr/local/opt/openblas

# CFLAGS+= -I${OPENBLAS_ROOT}/include
# LDFLAGS=-L${OPENBLAS_ROOT}/lib -lopenblas

# Added for opencv
CFLAGS+= `pkg-config --cflags opencv`
LDFLAGS+=`pkg-config --libs opencv`

# Added for mxnet
export MXNET_ROOT=`pwd`/../../../

CFLAGS+=-Wall -I$(MXNET_ROOT)/include
LDFLAGS+=-L$(MXNET_ROOT)/lib -lmxnet -lopenblas
ifeq ($(USE_ACL), 1)
LDFLAGS+=-L$(MXNET_ROOT)/../ComputeLibrary/build/arm_compute -L$(MXNET_ROOT)/../ComputeLibrary/build -larm_compute
endif

all:image-classification-predict image-classification-predict-forCaffeMode

image-classification-predict: image-classification-predict.o
	$(CXX) -O3 -o image-classification-predict image-classification-predict.o $(LDFLAGS)

image-classification-predict.o: image-classification-predict.cc
	$(CXX) -O3 -c image-classification-predict.cc ${CFLAGS}

image-classification-predict-forCaffeMode: image-classification-predict-forCaffeMode.o
	$(CXX) -O3 -o image-classification-predict-forCaffeMode image-classification-predict-forCaffeMode.o $(LDFLAGS)

image-classification-predict-forCaffeMode.o: image-classification-predict-forCaffeMode.cc
	$(CXX) -O3 -c image-classification-predict-forCaffeMode.cc ${CFLAGS}
	
clean: 
	rm -f *.d *.o
	rm -f image-classification-predict image-classification-predict-forCaffeMode

lint:
	python ../../../dmlc-core/scripts/lint.py mxnet "cpp" ./
