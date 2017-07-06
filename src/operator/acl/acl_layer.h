#ifndef CAFFE_ACL_LAYER_HPP_
#define CAFFE_ACL_LAYER_HPP_

#if USE_ACL == 1
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NENormalizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLNormalizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "arm_compute/runtime/CL/functions/CLPoolingLayer.h"
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "arm_compute/runtime/CL/functions/CLSoftmaxLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace mxnet {
namespace op {

enum TensorType{
    tensor_input,
    tensor_output,
    tensor_weights,
    tensor_biases,
};
template <typename ACLTensor>
class BaseTensor:public ACLTensor{
public:
    BaseTensor(bool share)
       :share_(share),type_(tensor_input),allocate_(false){
    }
    virtual void bindmem(void *mem,bool share){
        mem_=mem;
        share_=share;
    }
    virtual void settensortype(TensorType type){
        type_=type;
    };
    virtual void map(bool blocking = true){}
    virtual void unmap(){}
    virtual void commit();
    int tensor_copy(void * mem, bool toTensor=true);
protected:
    void* mem_;
    bool share_;
    TensorType type_;
    bool allocate_;
};
class GPUTensor:public BaseTensor<arm_compute::CLTensor>{
public:
    explicit GPUTensor(bool share)
       :BaseTensor(share){}
    virtual void map(bool blocking = true){
        if (!allocate_){
            arm_compute::CLTensor::allocator()->allocate();
            allocate_=true;
        }
        arm_compute::CLTensor::map(blocking);
     }
     virtual void unmap(){
        arm_compute::CLTensor::unmap();
     }
};
class CPUTensor:public BaseTensor<arm_compute::Tensor>{
public:
    explicit CPUTensor(bool share)
        :BaseTensor(share){}
    virtual void map(bool blocking = true){
        if (!allocate_){
            arm_compute::Tensor::allocator()->allocate();
            allocate_=true;
        }
    }
    virtual void unmap(){
    }
};
template <typename ACLLayer,typename ACLTensor>
class ACLXPUBaseLayer{
public:
    virtual void commit(){
        if (input) {
            input->settensortype(tensor_input);
            input->commit();
        }
        if (output){
            output->settensortype(tensor_output);
            output->commit();
        }
        if (weights){
            weights->settensortype(tensor_weights);
            weights->commit();
        }
        if (biases){
            biases->settensortype(tensor_biases);
            biases->commit();
        }
    }
    virtual void run(bool gpu){
        commit();
        layer->run();
        if (gpu) {
            // Make sure all the OpenCL jobs are done executing:
            arm_compute::CLScheduler::get().sync();
        }
    }
    virtual bool reshape(arm_compute::TensorShape &shape,TensorType type);
    explicit ACLXPUBaseLayer(){
        layer=nullptr;
        input=nullptr;
        output=nullptr;
        weights=nullptr;
        biases=nullptr;
    }
    virtual void freelayer(){
        if (layer) delete layer;
        if (input) delete input;
        if (output) delete output;
        if (weights) delete weights;
        if (biases) delete biases;
        layer=nullptr;
        input=nullptr;
        output=nullptr;
        weights=nullptr;
        biases=nullptr;
    }
    virtual ~ACLXPUBaseLayer(){
        freelayer();
    }
    ACLLayer *layer;
    ACLTensor *input;
    ACLTensor *output;
    ACLTensor *weights;
    ACLTensor *biases;
};
template <typename GPULayer, typename CPULayer>
class ACLBaseLayer {
public:
    explicit ACLBaseLayer();
    virtual void gpu_run();
    virtual void cpu_run();
    virtual ~ACLBaseLayer();
    virtual GPULayer * new_gpulayer();
    virtual CPULayer * new_cpulayer();
    ACLXPUBaseLayer<GPULayer,GPUTensor>& gpu(){
        return gpu_;
    }
    ACLXPUBaseLayer<CPULayer,CPUTensor>& cpu(){
        return cpu_;
    }
    bool checkreshape(arm_compute::TensorShape shape,bool gpu=false, TensorType type=tensor_input);
    void acl_run(void *input_data, void *output_data,bool gpu=false);
    template <typename ACLTensor> bool tensor_mem(ACLTensor *tensor,void *mem,bool share=false);
    template <typename ACLTensor> bool tensor_mem(void *mem,ACLTensor *tensor,bool share=false);
    template <typename ACLTensor> ACLTensor * new_tensor(arm_compute::TensorShape shape,void *mem=nullptr,bool share=false);
protected:
    ACLXPUBaseLayer<GPULayer,GPUTensor> gpu_;
    ACLXPUBaseLayer<CPULayer,CPUTensor> cpu_;
    bool init_layer_;
    bool force_bypass_acl_path_;
};

}  // namespace op
}  // namespace mxnet
#define INSTANTIATE_ACLBASECLASS(GPULayer,CPULayer) \
  template class ACLBaseLayer<GPULayer,CPULayer>; 

#define INSTANTIATE_ACLBASE_FUNCTION(GPULayer,CPULayer,ACLTensor) \
    template bool ACLBaseLayer<GPULayer,CPULayer>::tensor_mem<ACLTensor>(ACLTensor *tensor,void *mem,bool share); \
    template bool ACLBaseLayer<GPULayer,CPULayer>::tensor_mem(void *mem,ACLTensor *tensor,bool share); \
    template ACLTensor * ACLBaseLayer<GPULayer,CPULayer>::new_tensor(arm_compute::TensorShape shape,void *mem,bool share); \

#endif
#endif
