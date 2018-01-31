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
#include "acl_operator.h"

namespace mxnet {
namespace op {

template <typename xpu, typename ForwardOp, typename BackwardOp, typename DType>
class ACLActivationOp : public ActivationOp<xpu, ForwardOp,BackwardOp,DType>,public ACLOperator {
 private:
  ActivationParam param_;
  Context ctx_;

  void SetupACLLayer(const OpContext &ctx, const std::vector<TBlob> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &out_data,
                     const std::vector<TBlob> &aux_args){
        const unsigned int count  = in_data[activation::kData].shape_.Size();
        const unsigned int count_ = out_data[activation::kOut].shape_.Size();
        arm_compute::TensorShape input_shape(count);
        arm_compute::TensorShape output_shape(count_);
        if (is_operator_init_done(input_shape)) return;
        set_operator_init_done();

        // Initialize ACL.
        this->force_bypass_acl_path_=false;
        arm_compute::ActivationLayerInfo::ActivationFunction type;
        switch(param_.act_type){
            default:
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
     
        if(type== arm_compute::ActivationLayerInfo::ActivationFunction::TANH)
          act_info=arm_compute::ActivationLayerInfo(type,1.0,1.0);

      new_tensor(input(),input_shape,(void*)InputdataPtr<DType>(ACLOp_Ptr(this),in_data,activation::kData));
      new_tensor(output(),output_shape,(void*)OutputdataPtr<DType>(ACLOp_Ptr(this),out_data,activation::kOut));
      acl_configure(activation,this,act_info);
    }
  bool Bypass_acl() {
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_){
        bypass_acl=true;
    }
    return bypass_acl;
  }


 public:
  explicit ACLActivationOp(Context & ctx,ActivationParam p)
      : ActivationOp<xpu, ForwardOp,BackwardOp,DType>() , ACLOperator(ctx.arm_gpu_mode()){
    this->param_ = p;
    this->ctx_ = ctx;
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
      if (Bypass_acl()){
         ActivationOp<xpu, ForwardOp,BackwardOp,DType>::Forward(ctx,in_data,req,out_data,aux_args);
         return;
      }
      SetupACLLayer(ctx,in_data,req,out_data,aux_args);
      mxnet::op::acl_run<DType>(ACLOp_Ptr(this),in_data,out_data);
  }
};  // class ACLActivationOp
}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_ACL_ACTIVATION_INL_H_
