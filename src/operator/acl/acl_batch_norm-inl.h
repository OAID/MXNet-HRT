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
#include "acl_layer.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class ACLBatchNormOp : public BatchNormOp<xpu, DType, DType>,public ACLBaseLayer<arm_compute::CLBatchNormalizationLayer,arm_compute::NEBatchNormalizationLayer> {
 private:
  BatchNormParam param_;
  Context ctx_;
  bool is_gpu_;

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
      checkreshape(in_shape,is_gpu_);
      if (!this->init_layer_) return;
      this->init_layer_=false;
      PrepareHelpSetupACLLayer(ctx,in_data,req,out_data,aux_args);
      // Initialize ACL.
      if (is_gpu_) {
          new_gpulayer();
      }else{
          new_cpulayer();
      }

      this->force_bypass_acl_path_=false;
      DType * input_data =in_data[batchnorm::kData].dptr<DType>();
      DType * output_data =out_data[batchnorm::kOut].dptr<DType>();
      const TBlob &weights         = in_data[batchnorm::kGamma];
      const TBlob &bias            = in_data[batchnorm::kBeta];
      const TBlob &meanVector      = out_data[batchnorm::kMean];
      const TBlob &varianceVector  = out_data[batchnorm::kVar];
      DType *mean = meanVector.dptr<DType>();
      DType  *var = varianceVector.dptr<DType>();
      DType        *gamma_val = weights.dptr<DType>();
      const DType  *beta_val = bias.dptr<DType>();
      arm_compute::TensorShape mean_shape(in_channels);
      arm_compute::TensorShape var_shape=mean_shape;
      arm_compute::TensorShape beta_shape=mean_shape;
      arm_compute::TensorShape gamma_shape=mean_shape;

      if (is_gpu_) {
          new_tensor(this->gpu().input,in_shape,(void*)input_data);
          new_tensor(this->gpu().output,out_shape,(void*)output_data);
          new_tensor(this->gpu().mean,mean_shape);
          new_tensor(this->gpu().var,var_shape);
          new_tensor(this->gpu().beta,beta_shape);
          new_tensor(this->gpu().gamma,gamma_shape);
          tensor_mem(this->gpu().mean,(void*)mean);
          tensor_mem(this->gpu().var,(void*)var);
          tensor_mem(this->gpu().beta,(void*)beta_val);
          tensor_mem(this->gpu().gamma,(void*)gamma_val);
          this->gpu().mean->commit();
          this->gpu().var->commit();
          this->gpu().beta->commit();
          this->gpu().gamma->commit();
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->gpu().layer->configure(this->gpu().input,this->gpu().output,this->gpu().mean,this->gpu().var,this->gpu().beta,this->gpu().gamma,param_.eps);
      }else{
          new_tensor(this->cpu().input,in_shape,(void*)input_data);
          new_tensor(this->cpu().output,out_shape,(void*)output_data);
          new_tensor(this->cpu().mean,mean_shape);
          new_tensor(this->cpu().var,var_shape);
          new_tensor(this->cpu().beta,beta_shape);
          new_tensor(this->cpu().gamma,gamma_shape);
          tensor_mem(this->cpu().mean,(void*)mean);
          tensor_mem(this->cpu().var,(void*)var);
          tensor_mem(this->cpu().beta,(void*)beta_val);
          tensor_mem(this->cpu().gamma,(void*)gamma_val);
          this->cpu().mean->commit();
          this->cpu().var->commit();
          this->cpu().beta->commit();
          this->cpu().gamma->commit();
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->cpu().layer->configure(this->cpu().input,this->cpu().output,this->cpu().mean,this->cpu().var,this->cpu().beta,this->cpu().gamma,param_.eps);
      }
  }

 public:
  explicit ACLBatchNormOp(Context & ctx,BatchNormParam p)
      : BatchNormOp<xpu, DType, DType>(p) {
    this->param_ = p;
    this->ctx_ = ctx;
    this->is_gpu_ = ctx_.arm_gpu_mode();
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
      if (this->force_bypass_acl_path_){
          BatchNormOp<xpu, DType, DType>::Forward(ctx,in_data,req,out_data,aux_args);
          return;
      }
      DType * input_data =in_data[batchnorm::kData].dptr<DType>();
      DType * output_data =out_data[batchnorm::kOut].dptr<DType>();
      SetupACLLayer(ctx,in_data,req,out_data,aux_args);
      acl_run((void*)input_data,(void*)output_data,is_gpu_);
  }
};  // class ACLBatchNormOp
}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_ACL_BATCHNORM_INL_H_
