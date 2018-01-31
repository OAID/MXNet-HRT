/*!
 * Copyright (c) 2016 by Contributors
 * \file acl_batch_norm-inl.h
 * \brief
 * \author Joey
*/
#ifndef MXNET_OPERATOR_ACL_BATCHNORM_INL_H_
#define MXNET_OPERATOR_ACL_BATCHNORM_INL_H_

#if USE_ACL == 1
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../batch_norm-inl.h"
#include "acl_operator.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class ACLBatchNormOp : public BatchNormOp<xpu, DType, DType>,ACLOperator  {
 private:
  BatchNormParam param_;
  Context ctx_;

  void PrepareHelpSetupACLLayer(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args);
  void SetupACLLayer(const OpContext &ctx, const std::vector<TBlob> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &out_data,
                     const std::vector<TBlob> &aux_args){

      const TShape& ishape=in_data[batchnorm::kData].shape_;
      const TShape& oshape=out_data[batchnorm::kOut].shape_;
      unsigned int in_num=ishape[0];
      unsigned int in_channels=ishape[1];
      unsigned int in_width=ishape[2];
      unsigned int in_height=ishape[3];
      unsigned int out_num=oshape[0];
      unsigned int out_channels=oshape[1];
      unsigned int out_width=oshape[2];
      unsigned int out_height=oshape[3];
      arm_compute::TensorShape in_shape (in_width, in_height,in_channels,in_num);
      arm_compute::TensorShape out_shape(out_width, out_height,out_channels,out_num);
      if (is_operator_init_done(in_shape)) return;
      set_operator_init_done();
      PrepareHelpSetupACLLayer(ctx,in_data,req,out_data,aux_args);

      // Initialize ACL.
      this->force_bypass_acl_path_=false;
      arm_compute::TensorShape mean_shape(in_channels);
      arm_compute::TensorShape var_shape=mean_shape;
      arm_compute::TensorShape beta_shape=mean_shape;
      arm_compute::TensorShape gamma_shape=mean_shape;

      new_tensor(input(),in_shape,InputdataPtr<DType>(ACLOp_Ptr(this),in_data,batchnorm::kData));
      new_tensor(output(),out_shape,OutputdataPtr<DType>(ACLOp_Ptr(this),out_data,batchnorm::kOut));
      new_tensor(mean(),mean_shape,GetDataPtr<DType>(ACLOp_Ptr(this),out_data,batchnorm::kMean));
      new_tensor(var(),var_shape,GetDataPtr<DType>(ACLOp_Ptr(this),out_data,batchnorm::kVar));
      new_tensor(beta(),beta_shape,GetDataPtr<DType>(ACLOp_Ptr(this),in_data,batchnorm::kBeta));
      new_tensor(gamma(),gamma_shape,GetDataPtr<DType>(ACLOp_Ptr(this),in_data,batchnorm::kGamma));
      acl_configure(bn,this,param_.eps);

  }
  bool Bypass_acl() {
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_){
        bypass_acl=true;
    }
    return bypass_acl;
  }

 public:
  explicit ACLBatchNormOp(Context & ctx,BatchNormParam p)
      : BatchNormOp<xpu, DType, DType>(p) , ACLOperator(ctx.arm_gpu_mode()){
    this->param_ = p;
    this->ctx_ = ctx;
    this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_BN;
  }

 public:
  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_BN_INFO);
#endif //USE_PROFILING
      if (Bypass_acl()){
          BatchNormOp<xpu, DType, DType>::Forward(ctx,in_data,req,out_data,aux_args);
          return;
      }
      SetupACLLayer(ctx,in_data,req,out_data,aux_args);
      mxnet::op::acl_run<DType>(ACLOp_Ptr(this),in_data,out_data);
  }
};  // class ACLBatchNormOp
}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_ACL_BATCHNORM_INL_H_
