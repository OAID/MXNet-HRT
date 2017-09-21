# Release Note
[![Build](https://img.shields.io/teamcity/codebetter/bt428.svg)](build) [![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

The release version is 0.2.0. You can download the source code from [OAID/MXNetOnACL](https://github.com/OAID/MXNetOnACL)

## Verified Platform :

The release is verified on 64bits ARMv8 processor

- Hardware platform : Rockchip RK3399 ([Firefly-RK3399 board](http://wiki.t-firefly.com/index.php/Firefly-RK3399))
- Software platform : Ubuntu 16.04<br>

## ACL Compatibility Issues :
There are some compatibility issues between ACL and MXNet Operators, we bypass it to MXNet's original Operator class as the workaround solution for the below issues

* Normalization in-channel issue
* Tanh issue
* Softmax supporting multi-dimension issue
* Group issue

Performance need be fine turned in the future

# Issue Report
Encounter any issue, please report on [issue report](https://github.com/OAID/MXNetOnACL/issues). Issue report should contain the following information :

*  The exact description of the steps that are needed to reproduce the issue 
* The exact description of what happens and what you think is wrong 


## Release History
The MXNet based version is [26b1cb9ad0bcde9206863a6f847455ff3ec3c266](https://github.com/apache/incubator-mxnet/tree/26b1cb9ad0bcde9206863a6f847455ff3ec3c266).


## MXNetOnACL Version 0.2.0 - Aug 27, 2017

Support Arm Compute Library version 17.06 with 4 new layers added

* Batch Normalization Layer
* Direct convolution Layer
* Concatenate layer


## MXNetOnACL Version 0.1.0 - Jul 6, 2017 
   
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
