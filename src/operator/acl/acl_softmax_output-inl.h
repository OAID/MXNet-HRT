/*!
 * Copyright (c) 2016 by Contributors
 * \file acl_softmax_output-inl.h
 * \brief
 * \author Joey
*/
#ifndef MXNET_OPERATOR_ACL_SOFTMAX_OUTPUT_INL_H_
#define MXNET_OPERATOR_ACL_SOFTMAX_OUTPUT_INL_H_

#if USE_ACL == 1
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../softmax_output-inl.h"
#include "acl_operator.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class ACLSoftmaxOutputOp : public SoftmaxOutputOp<xpu, DType>,ACLOperator  {
 private:
  SoftmaxOutputParam param_;
  Context ctx_;
  unsigned int channels_;
  unsigned int inner_num_;
  unsigned int outer_num_;

  void SetupACLLayer(const OpContext &ctx, const std::vector<TBlob> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &out_data,
                     const std::vector<TBlob> &aux_args){

      arm_compute::TensorShape shape(this->channels_*this->inner_num_);
      if (is_operator_init_done(shape)) return;
      set_operator_init_done();

      // Initialize ACL.
      this->force_bypass_acl_path_=false;
      new_tensor(input(),shape,InputdataPtr<DType>(ACLOp_Ptr(this),in_data,softmaxout_enum::kData));
      new_tensor(output(),shape,OutputdataPtr<DType>(ACLOp_Ptr(this),out_data,softmaxout_enum::kOut));
      acl_configure(softmax,this,NULL);

  }
  bool Bypass_acl() {
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_|| this->inner_num_>1){
        bypass_acl=true;
    }
    return bypass_acl;
  }
 public:
  explicit ACLSoftmaxOutputOp(Context & ctx,SoftmaxOutputParam p)
      : SoftmaxOutputOp<xpu, DType>(p) , ACLOperator(ctx.arm_gpu_mode()){
    this->param_ = p;
    this->ctx_ = ctx;
    this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_SOFTMAX;
  }

 public:
  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
      const TShape& ishape=in_data[softmaxout_enum::kData].shape_;
      const TShape& oshape=out_data[softmaxout_enum::kOut].shape_;
      this->channels_ = ishape[1]; 
      this->inner_num_= ishape[0];
      this->outer_num_= oshape[0];
#ifdef USE_PROFILING
    logtime_util log_time(ACL_SOFTMAX_INFO);
#endif //USE_PROFILING
      if (Bypass_acl()){
         SoftmaxOutputOp<xpu, DType>::Forward(ctx,in_data,req,out_data,aux_args);
         return ;
      }
      SetupACLLayer(ctx,in_data,req,out_data,aux_args);

       DType * input_data =in_data[softmaxout_enum::kData].dptr<DType>();
       DType * output_data =out_data[softmaxout_enum::kOut].dptr<DType>();
       for (unsigned int i = 0; i < this->outer_num_; ++i) {
          acl_run(input_data,output_data);
          output_data += channels_;
          input_data += channels_;
      }
  }
};  // class ACLSoftmaxOutputOp
}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_ACL_SOFTMAX_OUTPUT_INL_H_
