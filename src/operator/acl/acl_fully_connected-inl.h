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
#include "acl_operator.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class ACLFullyConnectedOp : public FullyConnectedOp<xpu, DType>,ACLOperator {
 private:
  FullyConnectedParam param_;
  Context ctx_;
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
      if (is_operator_init_done(input_shape)) return;
      set_operator_init_done();

      // Initialize ACL.
      bool transpose = true;
      this->force_bypass_acl_path_ = false; 

      if (transpose) {
          new_tensor(weights(),weights_shape_t,GetDataPtr<DType>(ACLOp_Ptr(this),in_data,fullc::kWeight));
      }else{
          new_tensor(weights(),weights_shape,GetDataPtr<DType>(ACLOp_Ptr(this),in_data,fullc::kWeight));
      }
      if (!param_.no_bias) {
          new_tensor(biases(),biases_shape,GetDataPtr<DType>(ACLOp_Ptr(this),in_data,fullc::kBias));
      }
      new_tensor(input(),input_shape,InputdataPtr<DType>(ACLOp_Ptr(this),in_data,fullc::kData));
      new_tensor(output(),output_shape,OutputdataPtr<DType>(ACLOp_Ptr(this),out_data,fullc::kOut));
      acl_configure(fc,this,transpose);
  }
  bool Bypass_acl() {
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_){
        bypass_acl=true;
    }
    return bypass_acl;
  }

 public:
  explicit ACLFullyConnectedOp(Context & ctx,FullyConnectedParam p)
      : FullyConnectedOp<xpu, DType>(p) , ACLOperator(ctx.arm_gpu_mode()){
    this->param_ = p;
    this->ctx_ = ctx;
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
      if (Bypass_acl()){
         FullyConnectedOp<xpu, DType>::Forward(ctx,in_data,req,out_data,aux_args);
         return;
      }

      SetupACLLayer(ctx,in_data,req,out_data,aux_args);
      mxnet::op::acl_run<DType>(ACLOp_Ptr(this),in_data,out_data);
  }
};  // class ACLFullyConnectedOp
}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_ACL_FULLY_CONNECTED_INL_H_
