/*!
 * Copyright (c) 2016 by Contributors
 * \file acl_lrn-inl.h
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
#include "acl_operator.h"

namespace mxnet {
namespace op {

const arm_compute::NormType IN_MAP=(arm_compute::NormType)0;
template <typename xpu, typename DType>
class ACLLocalResponseNormOp : public LocalResponseNormOp<xpu>,ACLOperator {
 private:
  LRNParam param_;
  Context ctx_;
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
      if (is_operator_init_done(shape)) return;
      set_operator_init_done();

      // Initialize ACL.
      this->force_bypass_acl_path_=false;

      const float nsize = param_.nsize;
      const float alpha = param_.alpha;
      const float beta = param_.beta;
      const float knorm = param_.knorm;

      arm_compute::NormalizationLayerInfo norm_info(IN_MAP, nsize, alpha, beta, knorm);
      if(this->type_ == IN_MAP)
         norm_info=arm_compute::NormalizationLayerInfo(IN_MAP, nsize, alpha, beta, knorm);
      else
         norm_info=arm_compute::NormalizationLayerInfo(arm_compute::NormType::CROSS_MAP, nsize, alpha, beta, knorm);

      new_tensor(input(),shape,InputdataPtr<DType>(ACLOp_Ptr(this),in_data,lrn_enum::kData));
      new_tensor(output(),shape,OutputdataPtr<DType>(ACLOp_Ptr(this),out_data,lrn_enum::kOut));
      acl_configure(lrn,this,norm_info);
    }
  bool Bypass_acl() {
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_||this->type_ == IN_MAP){
        bypass_acl=true;
    }
    return bypass_acl;
  }


 public:
  explicit ACLLocalResponseNormOp(Context & ctx,LRNParam p)
      : LocalResponseNormOp<xpu>(p), ACLOperator(ctx.arm_gpu_mode()){
    this->param_ = p;
    this->ctx_ = ctx;
    this->type_=arm_compute::NormType::CROSS_MAP;
    this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_LRN;
  }

 public:
  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_LRN_INFO);
#endif //USE_PROFILING
      if (Bypass_acl()){
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
              acl_run(input_data,output_data);
              input_data+=ishape.ProdShape(1,ishape.ndim());
              output_data+=oshape.ProdShape(1,oshape.ndim());
          }
      }else if(this->type_==IN_MAP){
          acl_run(input_data,output_data);
      }
  }
};  // class ACLLocalResponseNormOp
}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_ACL_LRN_INL_H_
