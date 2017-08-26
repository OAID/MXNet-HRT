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
#include "acl_layer.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class ACLPoolingOp : public PoolingOp<xpu, DType>,public ACLBaseLayer<arm_compute::CLPoolingLayer,arm_compute::NEPoolingLayer> {
 private:
  PoolingParam param_;
  Context ctx_;
  bool is_gpu_;
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

      arm_compute::TensorShape in_shape ((unsigned int)this->width_, (unsigned int)this->height_);
      arm_compute::TensorShape out_shape((unsigned int)this->pooled_width_, (unsigned int)this->pooled_height_);
      checkreshape(in_shape,is_gpu_);
      if (!this->init_layer_) return;
      this->init_layer_=false;
      // Initialize ACL.
      if (is_gpu_) {
          new_gpulayer();
      }else{
          new_cpulayer();
      }

      this->force_bypass_acl_path_=false;
      arm_compute::PoolingLayerInfo *pool_info;
      DType * input_data =in_data[pool_enum::kData].dptr<DType>();
      DType * output_data =out_data[pool_enum::kOut].dptr<DType>();
      if(param_.pool_type!=pool_enum::kMaxPooling)
         pool_info=new arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, this->kernel_w_, arm_compute::PadStrideInfo(this->stride_w_,this->stride_h_,this->pad_w_,this->pad_h_,arm_compute::DimensionRoundingType::CEIL));
      else
         pool_info=new arm_compute::PoolingLayerInfo(arm_compute::PoolingType::AVG, this->kernel_w_, arm_compute::PadStrideInfo(this->stride_w_,this->stride_h_,this->pad_w_,this->pad_h_,arm_compute::DimensionRoundingType::CEIL));

      if (is_gpu_) {
          new_tensor(this->gpu().input,in_shape,(void*)input_data);
          new_tensor(this->gpu().output,out_shape,(void*)output_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
          this->gpu().layer->configure(this->gpu().input,this->gpu().output,*pool_info);
      }else{
          new_tensor(this->cpu().input,in_shape,(void*)input_data);
          new_tensor(this->cpu().output,out_shape,(void*)output_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
          this->cpu().layer->configure(this->cpu().input,this->cpu().output,*pool_info);
      }
      delete pool_info;
  }

 public:
  explicit ACLPoolingOp(Context & ctx,PoolingParam p)
      : PoolingOp<xpu, DType>(p) {
    this->param_ = p;
    this->ctx_ = ctx;
    this->is_gpu_ = ctx_.arm_gpu_mode();
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
      if (this->force_bypass_acl_path_||this->param_.global_pool){
          PoolingOp<xpu, DType>::Forward(ctx,in_data,req,out_data,aux_args);
          return;
      }
      DType * input_data =in_data[pool_enum::kData].dptr<DType>();
      DType * output_data =out_data[pool_enum::kOut].dptr<DType>();
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
      this->pooled_height_ = static_cast<int>(ceil(static_cast<float>(
          height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
      this->pooled_width_ = static_cast<int>(ceil(static_cast<float>(
          width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;

      TShape kernel=param_.global_pool?
        TShape(ishape.data()+ishape.ndim()-param_.kernel.ndim(), ishape.data()+ishape.ndim())
        : param_.kernel;
      this->stride_w_=kernel[0];
      this->stride_h_=kernel[1];
      if (param_.pool_type!=pool_enum::kMaxPooling && 
          param_.pool_type!=pool_enum::kAvgPooling) {
          PoolingOp<xpu, DType>::Forward(ctx,in_data,req,out_data,aux_args);
          return ;
      }
      if (this->kernel_h_!=this->kernel_w_ || oshape.Size()>1) {
          PoolingOp<xpu, DType>::Forward(ctx,in_data,req,out_data,aux_args);
          return ;
      }
      if (this->kernel_h_!=2 && this->kernel_h_!=3) {
          PoolingOp<xpu, DType>::Forward(ctx,in_data,req,out_data,aux_args);
          return ;
      }
      SetupACLLayer(ctx,in_data,req,out_data,aux_args);
      for (unsigned int n = 0; n < this->num_; ++n) {
        for (unsigned int c = 0; c < this->channels_; ++c) {
            acl_run((void*)input_data,(void*)output_data,is_gpu_);
            input_data += ishape.ProdShape(2, 3);
            output_data += oshape.ProdShape(2, 3);
        }
      }
  }
};  // class ACLPoolingOp
}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_ACL_POOLING_INL_H_
