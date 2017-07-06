/*!
 * Copyright (c) 2016 by Contributors
 * \file acl_convolution-inl.h
 * \brief
 * \author Joey
*/
#ifndef MXNET_OPERATOR_ACL_LRN_INL_H_
#define MXNET_OPERATOR_ACL_LRN_INL_H_

#if USE_ACL == 1
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../lrn-inl.h"
#include "acl_layer.h"

namespace mxnet {
namespace op {

const arm_compute::NormType IN_MAP=(arm_compute::NormType)0;
template <typename xpu, typename DType>
class ACLLocalResponseNormOp : public LocalResponseNormOp<xpu>,public ACLBaseLayer<arm_compute::CLNormalizationLayer,arm_compute::NENormalizationLayer> {
 private:
  LRNParam param_;
  Context ctx_;
  bool is_gpu_;
  arm_compute::NormType type_;

  void SetupACLLayer(const OpContext &ctx, const std::vector<TBlob> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &out_data,
                     const std::vector<TBlob> &aux_args){
      const TShape& ishape=in_data[lrn_enum::kData].shape_;
      unsigned int channels=ishape[1];
      unsigned int width=ishape[2];
      unsigned int height=ishape[3];
      arm_compute::TensorShape shape(width,height,channels);
      checkreshape(shape,is_gpu_);
      if (!this->init_layer_) return;
      // Initialize ACL.
      if (is_gpu_) {
          new_gpulayer();
      }else{
          new_cpulayer();
      }

      this->force_bypass_acl_path_=false;
      arm_compute::NormalizationLayerInfo *norm_info;
      DType * input_data =in_data[lrn_enum::kData].dptr<DType>();
      DType * output_data =out_data[lrn_enum::kOut].dptr<DType>();

      const float nsize = param_.nsize;
      const float alpha = param_.alpha;
      const float beta = param_.beta;
      const float knorm = param_.knorm;

      if(this->type_ == IN_MAP)
         norm_info=new arm_compute::NormalizationLayerInfo(IN_MAP, nsize, alpha, beta, knorm);
      else
         norm_info=new arm_compute::NormalizationLayerInfo(arm_compute::NormType::CROSS_MAP, nsize, alpha, beta, knorm);

      if (is_gpu_) {
          this->gpu().input=new_tensor<GPUTensor>(shape,(void*)input_data);
          this->gpu().output=new_tensor<GPUTensor>(shape,(void*)output_data);
          this->gpu().layer->configure(this->gpu().input,this->gpu().output,*norm_info);
      }else{
          this->cpu().input=new_tensor<CPUTensor>(shape,(void*)input_data);
          this->cpu().output=new_tensor<CPUTensor>(shape,(void*)output_data);
          this->cpu().layer->configure(this->cpu().input,this->cpu().output,*norm_info);
      }
      delete norm_info;
    }


 public:
  explicit ACLLocalResponseNormOp(Context & ctx,LRNParam p)
      : LocalResponseNormOp<xpu>(p) {
    this->param_ = p;
    this->ctx_ = ctx;
    this->is_gpu_ = ctx_.arm_gpu_mode();
    this->type_=arm_compute::NormType::CROSS_MAP;
  }

 public:
  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
      if (this->force_bypass_acl_path_||this->type_ == IN_MAP){
         LocalResponseNormOp<xpu>::Forward(ctx,in_data,req,out_data,aux_args);
         return;
      }

      DType * input_data =in_data[lrn_enum::kData].dptr<DType>();
      DType * output_data =out_data[lrn_enum::kOut].dptr<DType>();
      const TShape& ishape=in_data[lrn_enum::kData].shape_;
      const TShape& oshape=out_data[lrn_enum::kOut].shape_;
      SetupACLLayer(ctx,in_data,req,out_data,aux_args);
      if (this->type_==arm_compute::NormType::CROSS_MAP) {
          for (unsigned int n = 0; n < ishape[0]; ++n) {
              acl_run((void*)input_data,(void*)output_data,is_gpu_);
              input_data+=ishape.ProdShape(1,ishape.ndim());
              output_data+=oshape.ProdShape(1,oshape.ndim());
          }
      }else if(this->type_==IN_MAP){
          acl_run((void*)input_data,(void*)output_data,is_gpu_);
      }
  }
};  // class ACLLocalResponseNormOp
}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_ACL_LRN_INL_H_
