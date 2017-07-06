![OPEN AI LAB](https://github.com/OAID/mxnetOnACL/blob/master/acl_openailab/pics/openailab.png)

# 1. Release Notes
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

Please refer to [mxnetOnACL Release NOTE](https://github.com/OAID/mxnetOnACL/tree/master/acl_openailab/Reversion.md) for details

# 2. Preparation
## 2.1 General dependencies installation
	sudo apt-get -y update
	sodo apt-get -y upgrade
	sudo apt-get install build-essential libatlas-base-dev libopencv-dev -y 
	sudo apt-get install -y python-numpy python-scipy
	pip install --upgrade pip
	sudo apt-get install scons –y
	sudo apt-get install git –y
	sudo apt-get install axel -y

## 2.2 Download source code
Assume the directory structure of the code on firefly3399 is:

	ACL：/home/firefly/ComputeLibrary (git clone https://github.com/ARM-software/ComputeLibrary.git) (arm_compute v17.06)
	Mxnet：/home/firefly/mxnetOnACL  (git clone https://github.com/OAID/mxnetOnACL.git --recursive)

#### Download "ACL" (arm_compute : v17.06):
	git clone https://github.com/ARM-software/ComputeLibrary.git
#### Download "mxnetOnACL" :
	git clone https://github.com/OAID/mxnetOnACL.git --recursive

# 3. Build mxnetOnACL
## 3.1 Build ACL :
	cd /home/firefly/ComputeLibrary
	scons Werror=1 -j8 debug=0 asserts=1 neon=1 opencl=1 embed_kernels=1 os=linux arch=arm64-v8a

## 3.2 Build mxnet :
	cd /home/firefly/mxnetOnACL
	cp config.mk.acl config.mk
	make
	cd python
	sudo python setup.py install

## 3.3 build classification sample :
	export CXX=aarch64-linux-gnu-g++
	export USE_ACL=1
	cd /home/firefly/mxnetOnACL/example/image-classification/predict-cpp
	make

## 4 Run tests
If the output message of the following test is same as the examples, it means Mxnet poring is success.

#### Prepare model:
	mkdir /home/firefly/mxnetOnACL/model
	mkdir /home/firefly/mxnetOnACL/model/Inception
	cd  /home/firefly/mxnetOnACL/model/Inception
	axel -n 10 http://data.mxnet.io/mxnet/models/imagenet/inception-bn/Inception-BN-0126.params
	axel http://data.mxnet.io/mxnet/models/imagenet/inception-bn/Inception-BN-symbol.json
#### Run classification
	export LD_LIBRARY_PATH=/home/firefly/mxnetOnACL/lib:/home/firefly/ComputeLibrary/build
	./example/image-classification/predict-cpp/image-classification-predict ./model/pictures/cat.jpg
####
	output message --
	  model/Inception/Inception-BN-symbol.json ... 116922 bytes
	  model/Inception/Inception-BN-0126.params ... 45284780 bytes
	  [22:55:08] src/nnvm/legacy_json_util.cc:190: Loading symbol saved by previous version v0.8.0. Attempting to upgrade...
	  [22:55:08] src/nnvm/legacy_json_util.cc:198: Symbol successfully upgraded!
	  model/Inception/mean_224.nd ... 602188 bytes
	  Accuracy[0] = 0.00000030
	  ......
	  Accuracy[999] = 0.00007677
	  Best Result: [ Egyptian cat] id = 285, accuracy = 0.35542053
####

