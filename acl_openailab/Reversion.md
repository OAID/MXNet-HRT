# Release Note
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

The release version is 0.1.0. You can download the source code from [OAID/mxnetOnACL](https://github.com/OAID/mxnetOnACL)

## Verified Platform :

The release is verified on 64bits ARMv8 processor<br>
* Hardware platform : Rockchip RK3399 (firefly RK3399 board)<br>
* Software platform : Ubuntu 16.04<br>

## 6 operators accelerated by ACL layers :
* 	activation
*   convolution
*   fully_connected
*   lrn
*   pooling
*   softmax_output

## ACL compatibility issues :
There are some compatibility issues between ACL and mxnet's operators, we bypass it to mxnet's original operators as the workaround solution for the below issues
* Normalization in-channel issue
* Tanh issue
* Even Kernel size
* Softmax supporting multi-dimension issue
* Group issue
* Performance need be fine turned in the future

# Changelist
The caffe based version is `793bd96351749cb8df16f1581baf3e7d8036ac37`.
## New Files :
*	acl_openailab\README.md
*	acl_openailab\Reversion.md
*	acl_openailab\pics\openailab.png
*	amalgamation\std_string_func.h
*	config.mk.acl
*	example\image-classification\predict-cpp\image-classification-predict-forCaffeMode.cc
*	model\Inception\mean_224.nd
*	model\pictures\cat.jpg
*	src\operator\acl\acl_activation-inl.h
*	src\operator\acl\acl_convolution-inl.h
*	src\operator\acl\acl_fully_connected-inl.h
*	src\operator\acl\acl_layer.cc
*	src\operator\acl\acl_layer.h
*	src\operator\acl\acl_lrn-inl.h
*	src\operator\acl\acl_pooling-inl.h
*	src\operator\acl\acl_softmax_output-inl.h

## Change Files :
*	Makefile
*	README.md
*	amalgamation\Makefile
*	amalgamation\amalgamation.py
*	amalgamation\mxnet_predict0.cc
*	example\image-classification\predict-cpp\Makefile
*	include\mxnet\base.h
*	src\ndarray\ndarray.cc
*	src\operator\activation-inl.h
*	src\operator\activation.cc
*	src\operator\convolution.cc
*	src\operator\fully_connected.cc
*	src\operator\lrn-inl.h
*	src\operator\lrn.cc
*	src\operator\pooling-inl.h
*	src\operator\pooling.cc
*	src\operator\softmax_output-inl.h
*	src\operator\softmax_output.cc

# Issue report
Encounter any issue, please report on [issue report](https://github.com/OAID/mxnetOnACL/issues). Issue report should contain the following information :
* The exact description of the steps that are needed to reproduce the issue 
* The exact description of what happens and what you think is wrong 
