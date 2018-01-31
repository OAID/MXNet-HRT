# 1. User Quick Guide
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

This User Quick Guide will help you get started to setup MXNetOnACL on RK3399 quickly.

# 2. Preparation
## 2.1 General dependencies installation
	sudo apt-get update -y
	sudo apt-get upgrade -y
	sudo apt-get install build-essential git libatlas-base-dev  libblas-dev libopencv-dev -y 
	sudo apt-get install python-pip python-dev -y
	sudo apt-get install -y python-numpy python-scipy
	sudo pip install --upgrade pip
	sudo apt-get install scons –y
	sudo apt-get install git –y

## 2.2 Download source code
	cd ~

#### Download "AID-tools" (AID-tools : [v1.0](ftp://ftp.openailab.net/tools/package)):
	wget ftp://ftp.openailab.net/tools/package/AID-tools.tar.gz
	
#### Download "MXNetOnACL" :
	git clone --recursive https://github.com/OAID/MXNetOnACL.git

# 3. Build MXNetOnACL
## 3.1 install AID-tools :
	sudo tar -xvf AID-tools.tar.gz -C /usr/local
	sudo /usr/local/AID/gen-pkg-config-pc.sh /usr/local/AID

## 3.2 Build MXNet :
	cd ~/MXNetOnACL
	make
	sudo make install
	sudo /usr/local/AID/gen-pkg-config-pc.sh /usr/local/AID

## 3.3 Build Classification Sample
	cd example/image-classification/predict-cpp
	export CXX=aarch64-linux-gnu-g++
	export USE_ACL=1
	make

# 4. Run Caffenet Classification

## 4.1 Download MXNet Model
	cd ~/MXNetOnACL
	cd model
	   You can download MXNet pretrained model and synset text from  http://data.mxnet.io/mxnet/models/imagenet/
	   or download model from ftp://ftp.openailab.net/tools/CaffeOnACL_test_model/models.tar.gz

## 4.2 Run MXNet Classification 
	cd ~/MXNetOnACL
	example/image-classification/predict-cpp/image-classification-predict-forCaffeMode cpu model/caffenet/caffenet-symbol.json model/caffenet/caffenet-0000.params model/Inception/mean_224.nd model/synset.txt model/pictures/cat.jpg
    The output message:
    Best Result: [ tabby, tabby cat] id = 281, accuracy = 0.27792722

