# 1. User Quick Guide
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

This User Quick Guide will help you get started to setup MXNetOnACL on RK3399 quickly.

# 2. Preparation
## 2.1 General dependencies installation
	sudo apt-get update -y
	sudo apt-get upgrade -y
	sudo apt-get install build-essential git libatlas-base-dev libopencv-dev -y 
	sudo apt-get install python-pip python-dev -y
	sudo apt-get install -y python-numpy python-scipy
	sudo pip install --upgrade pip
	sudo apt-get install scons –y
	sudo apt-get install git –y


## 2.2 Download source code
	cd ~

#### Download "ACL" (arm_compute : [v17.06](https://github.com/ARM-software/ComputeLibrary/tree/dbdab85d6e0f96d3361a9e30310367d89953466c)):
	git clone https://github.com/ARM-software/ComputeLibrary.git
#### Download "MXNetOnACL" :
	git clone --recursive https://github.com/OAID/MXNetOnACL.git

# 3. Build MXNetOnACL
## 3.1 Build ACL :
	cd ~/ComputeLibrary
	scons Werror=1 -j8 debug=0 asserts=1 neon=1 opencl=1 embed_kernels=1 os=linux arch=arm64-v8a

## 3.2 Build MXNet :
	export ACL_ROOT=~/ComputeLibrary
	cd ~/MXNetOnACL
	cp config.mk.acl config.mk
	make 

## 3.3 Build Classification Sample
	cd example/image-classification/predict-cpp
	export CXX=aarch64-linux-gnu-g++
	export USE_ACL=1
	make

## 3.4 To Configure The Libraries

	sudo cp ~/ComputeLibrary/build/libArm_compute.so /usr/lib 
	sudo cp ~/MXNetOnACL/lib/libmxnet.so  /usr/lib

# 4. Run Caffenet Classification

## 4.1 Download MXNet Model
	cd ~/MXNetOnACL
	mkdir model
	cd model
	   You can download MXNet pretrained model from  http://data.mxnet.io/mxnet/models/imagenet/caffenet/

## 4.2 Run MXNet Classification 
	cd ~/MXNetOnACL
	example/image-classification/predict-cpp/image-classification-predict models/bvlc_alexnet/caffenet-symbol.json models/bvlc_alexnet/caffenet-0000.params mean_224.nd synset_words.txt cat.jpg
   The output message:<br>
	Best Result: [ tabby, tabby cat] id = 281, accuracy = 0.23338266

