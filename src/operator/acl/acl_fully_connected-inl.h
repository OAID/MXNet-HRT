/*!
 * Copyright (c) 2016 by Contributors
 * \file acl_fully_connected-inl.h
 * \brief
 * \author Joey
*/
#ifndef MXNET_OPERATOR_ACL_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_ACL_FULLY_CONNECTED_INL_H_

#if USE_ACL == 1
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../fully_connected-inl.h"
#include "acl_layer.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class ACLFullyConnectedOp : public FullyConnectedOp<xpu, DType>,ACLBaseLayer<arm_compute::CLFullyConnectedLayer,arm_compute::NEFullyConnectedLayer> {
 private:
  FullyConnectedParam param_;
  Context ctx_;
  bool is_gpu_;
  void SetupACLLayer(const OpContext &ctx, const std::vector<TBlob> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &out_data,
                     const std::vector<TBlob> &aux_args){

      mshadow::Shape<4> ishape= mshadow::Shape4(1,1,1,1);
      mshadow::Shape<4> oshape= mshadow::Shape4(1,1,1,1);

      //ishape
      for (unsigned int i=0;i<in_data[fullc::kData].shape_.ndim();++i) {
          ishape[i]=in_data[fullc::kData].shape_[i];
      }
      for (unsigned int i=0;i<out_data[fullc::kOut].shape_.ndim();++i) {
          oshape[i]=out_data[fullc::kOut].shape_[i];
      }
      unsigned int M = ishape[0]*oshape[2] * oshape[3]; 
      unsigned int N = oshape[1];
      unsigned int K = ishape[1]*ishape[2]*ishape[3];

      arm_compute::TensorShape weights_shape_t(K, N);
      arm_compute::TensorShape weights_shape(N, K);
      arm_compute::TensorShape biases_shape(N);
      arm_compute::TensorShape input_shape(K, M);
      arm_compute::TensorShape output_shape(N, M);
      checkreshape(input_shape,is_gpu_);
      if (!this->init_layer_) return;
      this->init_layer_=false;
      // Initialize ACL.
      if (is_gpu_) {
          new_gpulayer();
      }else{
          new_cpulayer();
      }

      DType * input_data =in_data[fullc::kData].dptr<DType>();
      DType * output_data =out_data[fullc::kOut].dptr<DType>();
      DType * weithts_data=in_data[fullc::kWeight].dptr<DType>();
      DType * bias_data;
      if (!param_.no_bias) 
          bias_data=in_data[fullc::kBias].dptr<DType>();

      bool transpose = true;
      this->force_bypass_acl_path_ = false; 
      if (is_gpu_) {
          if (transpose) {
              new_tensor(this->gpu().weights,weights_shape_t,(void*)weithts_data);
          }else{
              new_tensor(this->gpu().weights,weights_shape,(void*)weithts_data);
          }
          tensor_mem(this->gpu().weights,(void*)weithts_data);
          if (!param_.no_bias) {
              new_tensor(this->gpu().biases,biases_shape,(void*)bias_data);
              tensor_mem(this->gpu().biases,(void*)bias_data);
          }
          new_tensor(this->gpu().input,input_shape,(void*)input_data);
          new_tensor(this->gpu().output,output_shape,(void*)output_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
          this->gpu().layer->configure(this->gpu().input,this->gpu().weights,this->gpu().biases,this->gpu().output,transpose);
      }else{
          if (transpose) {
              new_tensor(this->cpu().weights,weights_shape_t,(void*)weithts_data);
          }else{
              new_tensor(this->cpu().weights,weights_shape,(void*)weithts_data);
          }
          tensor_mem(this->cpu().weights,(void*)weithts_data);
          if (!param_.no_bias) {
              new_tensor(this->cpu().biases,biases_shape,(void*)bias_data);
              tensor_mem(this->cpu().biases,(void*)bias_data);
          }
          new_tensor(this->cpu().input,input_shape,(void*)input_data);
          new_tensor(this->cpu().output,output_shape,(void*)output_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
          this->cpu().layer->configure(this->cpu().input,this->cpu().weights,this->cpu().biases,this->cpu().output,transpose);

      }
  }

 public:
  explicit ACLFullyConnectedOp(Context & ctx,FullyConnectedParam p)
      : FullyConnectedOp<xpu, DType>(p) {
    this->param_ = p;
    this->ctx_ = ctx;
    this->is_gpu_ = ctx_.arm_gpu_mode();
    this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_FC;
  }

 public:
  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_FC_INFO);
#endif //USE_PROFILING
      if (this->force_bypass_acl_path_){
         FullyConnectedOp<xpu, DType>::Forward(ctx,in_data,req,out_data,aux_args);
         return;
      }

      DType * input_data =in_data[fullc::kData].dptr<DType>();
      DType * output_data =out_data[fullc::kOut].dptr<DType>();
      SetupACLLayer(ctx,in_data,req,out_data,aux_args);
      acl_run((void*)input_data,(void*)output_data,is_gpu_);
  }
};  // class ACLFullyConnectedOp
}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_ACL_FULLY_CONNECTED_INL_H_
