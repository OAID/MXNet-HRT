/*!
 * Copyright (c) 2016 by Contributors
 * \file acl_convolution-inl.h
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
#include "acl_layer.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class ACLSoftmaxOutputOp : public SoftmaxOutputOp<xpu, DType>,public ACLBaseLayer<arm_compute::CLSoftmaxLayer,arm_compute::NESoftmaxLayer> {
 private:
  SoftmaxOutputParam param_;
  Context ctx_;
  bool is_gpu_;
  unsigned int channels_;
  unsigned int inner_num_;
  unsigned int outer_num_;

  void SetupACLLayer(const OpContext &ctx, const std::vector<TBlob> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &out_data,
                     const std::vector<TBlob> &aux_args){

      arm_compute::TensorShape shape(this->channels_*this->inner_num_);
      checkreshape(shape,is_gpu_);
      if (!this->init_layer_) return;
      this->init_layer_=false;

      // Initialize ACL.
      if (is_gpu_) {
          new_gpulayer();
      }else{
          new_cpulayer();
      }

      this->force_bypass_acl_path_=false;
      DType * input_data =in_data[softmaxout_enum::kData].dptr<DType>();
      DType * output_data =out_data[softmaxout_enum::kOut].dptr<DType>();
      if (is_gpu_) {
          this->gpu().input=new_tensor<GPUTensor>(shape,(void*)input_data);
          this->gpu().output=new_tensor<GPUTensor>(shape,(void*)output_data);
          this->gpu().layer->configure(this->gpu().input,this->gpu().output);
      }else{
          this->cpu().input=new_tensor<CPUTensor>(shape,(void*)input_data);
          this->cpu().output=new_tensor<CPUTensor>(shape,(void*)output_data);
          this->cpu().layer->configure(this->cpu().input,this->cpu().output);
      }
  }

 public:
  explicit ACLSoftmaxOutputOp(Context & ctx,SoftmaxOutputParam p)
      : SoftmaxOutputOp<xpu, DType>(p) {
    this->param_ = p;
    this->ctx_ = ctx;
    this->is_gpu_ = ctx_.arm_gpu_mode();
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
      if (this->force_bypass_acl_path_|| this->inner_num_>1){
         SoftmaxOutputOp<xpu, DType>::Forward(ctx,in_data,req,out_data,aux_args);
         return ;
      }
      DType * input_data =in_data[softmaxout_enum::kData].dptr<DType>();
      DType * output_data =out_data[softmaxout_enum::kOut].dptr<DType>();
      SetupACLLayer(ctx,in_data,req,out_data,aux_args);

      for (unsigned int i = 0; i < this->outer_num_; ++i) {
          acl_run((void*)input_data,(void*)output_data,is_gpu_);
          output_data += channels_;
          input_data += channels_;
      }
  }
};  // class ACLSoftmaxOutputOp
}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_ACL_SOFTMAX_OUTPUT_INL_H_
