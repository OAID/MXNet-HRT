/*!
 * Copyright (c) 2016 by Contributors
 * \file acl_pooling-inl.h
 * \brief
 * \author Joey
*/
#ifndef MXNET_OPERATOR_ACL_POOLING_INL_H_
#define MXNET_OPERATOR_ACL_POOLING_INL_H_

#if USE_ACL == 1
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../pooling-inl.h"
#include "acl_operator.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class ACLPoolingOp : public PoolingOp<xpu, DType>, public ACLOperator {
 private:
  PoolingParam param_;
  Context ctx_;
  unsigned int stride_w_;
  unsigned int stride_h_;
  unsigned int pad_w_;
  unsigned int pad_h_;
  unsigned int num_;
  unsigned int channels_;
  unsigned int width_;
  unsigned int height_;
  unsigned int kernel_w_;
  unsigned int kernel_h_;
  unsigned int pooled_height_;
  unsigned int pooled_width_;

  void SetupACLLayer(const OpContext &ctx, const std::vector<TBlob> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &out_data,
                     const std::vector<TBlob> &aux_args){

      arm_compute::TensorShape in_shape ((unsigned int)this->width_, (unsigned int)this->height_,(unsigned int)this->channels_);
      arm_compute::TensorShape out_shape((unsigned int)this->pooled_width_, (unsigned int)this->pooled_height_,(unsigned int)this->channels_);
      if (is_operator_init_done(in_shape)) return;
      set_operator_init_done();

      // Initialize ACL.
      this->force_bypass_acl_path_=false;
      arm_compute::PoolingLayerInfo pool_info;
      if(param_.pool_type==pool_enum::kMaxPooling)
         pool_info=arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, this->kernel_w_, arm_compute::PadStrideInfo(this->stride_w_,this->stride_h_,this->pad_w_,this->pad_h_,arm_compute::DimensionRoundingType::CEIL));
      else
         pool_info=arm_compute::PoolingLayerInfo(arm_compute::PoolingType::AVG, this->kernel_w_, arm_compute::PadStrideInfo(this->stride_w_,this->stride_h_,this->pad_w_,this->pad_h_,arm_compute::DimensionRoundingType::CEIL));

      new_tensor(input(),in_shape,InputdataPtr<DType>(ACLOp_Ptr(this),in_data,pool_enum::kData));
      new_tensor(output(),out_shape,OutputdataPtr<DType>(ACLOp_Ptr(this),out_data,pool_enum::kOut));
      acl_configure(pooling,this,pool_info);
  }
  bool Bypass_acl() {
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_||this->param_.global_pool){
        bypass_acl=true;
    }
    if (param_.pool_type!=pool_enum::kMaxPooling && 
        param_.pool_type!=pool_enum::kAvgPooling) {
        bypass_acl=true;
    }
    if (this->kernel_h_!=this->kernel_w_ ) {
        bypass_acl=true;
    }
    if (this->kernel_h_!=2 && this->kernel_h_!=3) {
        bypass_acl=true;
    }
    return bypass_acl;
  }

 public:
  explicit ACLPoolingOp(Context & ctx,PoolingParam p)
      : PoolingOp<xpu, DType>(p) , ACLOperator(ctx.arm_gpu_mode()){
    this->param_ = p;
    this->ctx_ = ctx;
    this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_POOLING;
  }

 public:
  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_POOLING_INFO);
#endif //USE_PROFILING
      const TShape& ishape=in_data[pool_enum::kData].shape_;
      const TShape& oshape=out_data[pool_enum::kOut].shape_;
      this->pad_w_=param_.pad[0];
      this->pad_h_=param_.pad[1];
      this->num_ = ishape[0];
      this->channels_= ishape[1];
      this->width_ = ishape[2];
      this->height_ = ishape[3];
      this->kernel_w_=param_.kernel[0];
      this->kernel_h_=param_.kernel[1];
      TShape stride=param_.global_pool?
        TShape(ishape.data()+ishape.ndim()-param_.kernel.ndim(), ishape.data()+ishape.ndim())
        : param_.stride;
      this->stride_w_=stride[0];
      this->stride_h_=stride[1];
      this->pooled_height_ = static_cast<int>(ceil(static_cast<float>(
          height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
      this->pooled_width_ = static_cast<int>(ceil(static_cast<float>(
          width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;

      if (Bypass_acl()) {
          PoolingOp<xpu, DType>::Forward(ctx,in_data,req,out_data,aux_args);
          return ;
      }
      SetupACLLayer(ctx,in_data,req,out_data,aux_args);
      DType * input_data =in_data[pool_enum::kData].dptr<DType>();
      DType * output_data =out_data[pool_enum::kOut].dptr<DType>();
      for (unsigned int n = 0; n < this->num_; ++n) {
            acl_run(input_data,output_data);
            input_data += ishape.ProdShape(1, 3);
            output_data += oshape.ProdShape(1, 3);
      }
  }
};  // class ACLPoolingOp
}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_ACL_POOLING_INL_H_
