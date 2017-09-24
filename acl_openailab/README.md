# MXNetOnACL
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

MXNet is a project that is maintained by **OPEN** AI LAB, it uses Arm Compute Library (NEON+GPU) to speed up [MXNet](https://github.com/apache/incubator-mxnet) and provide utilities to debug, profile and tune application performance. 

The release version is 0.3.0, is based on [Rockchip RK3399](http://www.rock-chips.com/plus/3399.html) Platform, target OS is Ubuntu 16.04. Can download the source code from [OAID/MXNet](https://github.com/OAID/MXNetOnACL)

* The ARM Computer Vision and Machine Learning library is a set of functions optimised for both ARM CPUs and GPUs using SIMD technologies. See also [Arm Compute Library](https://github.com/ARM-software/ComputeLibrary).
* Caffe is a fast open framework for deep learning. See also [MXNet](https://github.com/apache/incubator-mxnet).

### Documents
* [Installation instructions](https://github.com/OAID/MXNetOnACL/blob/master/acl_openailab/installation.md)
* [User Manuals PDF](https://github.com/OAID/MXNetOnACL/blob/master/acl_openailab/user_manual.pdf)
* [Performance Report PDF](https://github.com/OAID/MXNetOnACL/blob/master/acl_openailab/performance_report.pdf)

### Arm Compute Library Compatibility Issues :
There are some compatibility issues between ACL and Caffe Layers, we bypass it to Caffe's original layer class as the workaround solution for the below issues

* Normalization in-channel issue
* Tanh issue
* Softmax supporting multi-dimension issue
* Group issue

Performance need be fine turned in the future

# Release History
The MXNet based version is [26b1cb9ad0bcde9206863a6f847455ff3ec3c266](https://github.com/apache/incubator-mxnet/tree/26b1cb9ad0bcde9206863a6f847455ff3ec3c266).
## Version 0.2.0 - Aug 27, 2017

Support Arm Compute Library version 17.06 with 4 new layers added

* Batch Normalization Layer
* Direct convolution Layer
* Concatenate layer


## Version 0.1.0 - Jul 6, 2017 
   
  Initial version supports 10 Layers accelerated by Arm Compute Library version 17.05 : 

* Convolution Layer
* Pooling Layer
* LRN Layer
* ReLU Layer
* Sigmoid Layer
* Softmax Layer
* TanH Layer
* AbsVal Layer
* BNLL Layer
* InnerProduct Layer


# Issue Report
Encounter any issue, please report on [issue report](https://github.com/OAID/MXNetOnACL/issues). Issue report should contain the following information :

*  The exact description of the steps that are needed to reproduce the issue 
* The exact description of what happens and what you think is wrong 
