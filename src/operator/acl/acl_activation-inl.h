/*!
 * Copyright (c) 2016 by Contributors
 * \file acl_activation-inl.h
 * \brief
 * \author Joey
*/
#ifndef MXNET_OPERATOR_ACL_ACTIVATION_INL_H_
#define MXNET_OPERATOR_ACL_ACTIVATION_INL_H_

#if USE_ACL == 1
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../activation-inl.h"
#include "acl_layer.h"

namespace mxnet {
namespace op {

template <typename xpu, typename ForwardOp, typename BackwardOp, typename DType>
class ACLActivationOp : public ActivationOp<xpu, ForwardOp,BackwardOp,DType>,public ACLBaseLayer<arm_compute::CLActivationLayer,arm_compute::NEActivationLayer> {
 private:
  ActivationParam param_;
  Context ctx_;
  bool is_gpu_;

  void SetupACLLayer(const OpContext &ctx, const std::vector<TBlob> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &out_data,
                     const std::vector<TBlob> &aux_args){
        const unsigned int count  = in_data[activation::kData].shape_.Size();
        const unsigned int count_ = out_data[activation::kOut].shape_.Size();
        arm_compute::TensorShape input_shape(count);
        arm_compute::TensorShape output_shape(count_);
        checkreshape(input_shape,is_gpu_);
        if (!this->init_layer_) return;
        this->init_layer_=false;
        // Initialize ACL.
        if (is_gpu_) {
            new_gpulayer();
        }else{
            new_cpulayer();
        }

        this->force_bypass_acl_path_=false;
        arm_compute::ActivationLayerInfo::ActivationFunction type;
        switch(param_.act_type){
            case activation::kReLU:
                type=arm_compute::ActivationLayerInfo::ActivationFunction::RELU;
                break;
            case activation::kSigmoid:
                type=arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC;
                break;
            case activation::kTanh:
                type=arm_compute::ActivationLayerInfo::ActivationFunction::TANH;
                break;
            case activation::kSoftReLU:
                type=arm_compute::ActivationLayerInfo::ActivationFunction::SOFT_RELU;
                break;
        }
        arm_compute::ActivationLayerInfo act_info(type);
        DType * input_data =in_data[activation::kData].dptr<DType>();
        DType * output_data =out_data[activation::kOut].dptr<DType>();
         
        if(type== arm_compute::ActivationLayerInfo::ActivationFunction::TANH)
          act_info=arm_compute::ActivationLayerInfo(type,1.0,1.0);

        if (is_gpu_) {
            new_tensor(this->gpu().input,input_shape,(void*)input_data);
            new_tensor(this->gpu().output,output_shape,(void*)output_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
            this->gpu().layer->configure(this->gpu().input,this->gpu().output,act_info);
        }else{
            new_tensor(this->cpu().input,input_shape,(void*)input_data);
            new_tensor(this->cpu().output,output_shape,(void*)output_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
            this->cpu().layer->configure(this->cpu().input,this->cpu().output,act_info);
        }
    }


 public:
  explicit ACLActivationOp(Context & ctx,ActivationParam p)
      : ActivationOp<xpu, ForwardOp,BackwardOp,DType>() {
    this->param_ = p;
    this->ctx_ = ctx;
    this->is_gpu_ = ctx_.arm_gpu_mode();
    switch(param_.act_type){
        case activation::kReLU:
            this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_RELU;
            break;
        case activation::kSigmoid:
            this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_SIGMOID;
            break;
        case activation::kTanh:
            this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_TANH;
            break;
        case activation::kSoftReLU:
            this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_BNLL;
            break;
    }
  }

 public:
  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
#ifdef USE_PROFILING
      logtime_util log_time;
      switch(param_.act_type){
          case activation::kReLU:
              log_time.setlogtime_info(ACL_RELU_INFO);
              break;
          case activation::kSigmoid:
              log_time.setlogtime_info(ACL_SIGMOID_INFO);
              break;
          case activation::kTanh:
              log_time.setlogtime_info(ACL_TANH_INFO);
              break;
          case activation::kSoftReLU:
              log_time.setlogtime_info(ACL_BNLL_INFO);
              break;
      }
#endif //USE_PROFILING
      if (this->force_bypass_acl_path_){
         ActivationOp<xpu, ForwardOp,BackwardOp,DType>::Forward(ctx,in_data,req,out_data,aux_args);
         return;
      }
      DType * input_data =in_data[activation::kData].dptr<DType>();
      DType * output_data =out_data[activation::kOut].dptr<DType>();
      SetupACLLayer(ctx,in_data,req,out_data,aux_args);
      acl_run((void*)input_data,(void*)output_data,is_gpu_);
  }
};  // class ACLActivationOp
}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_ACL_ACTIVATION_INL_H_
