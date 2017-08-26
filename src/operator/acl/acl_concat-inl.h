/*!
 * Copyright (c) 2016 by Contributors
 * \file acl_concat-inl.h
 * \brief
 * \author Joey
*/
#ifndef MXNET_OPERATOR_ACL_CONCAT_INL_H_
#define MXNET_OPERATOR_ACL_CONCAT_INL_H_

#if USE_ACL == 1
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../concat-inl.h"
#include "acl_layer.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class ACLConcatOp : public ConcatOp<xpu, DType>,public ACLBaseLayer<arm_compute::CLDepthConcatenate,arm_compute::NEDepthConcatenate> {
 private:
  ConcatParam param_;
  Context ctx_;
  bool is_gpu_;
  std::vector<arm_compute::ITensor *> cpu_vectors;
  std::vector<arm_compute::ICLTensor *> gpu_vectors;

  void SetupACLLayer(const OpContext &ctx, const std::vector<TBlob> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &out_data,
                     const std::vector<TBlob> &aux_args){

      unsigned int channels=0;
      for (int i = 0; i < in_data.size(); ++i) {
          channels+=in_data[i].shape_[1];
      }
      const TShape& oshape=out_data[concat_enum::kOut].shape_;
      unsigned int out_num=oshape[0];
      unsigned int out_channels=oshape[1];
      unsigned int out_width=oshape[2];
      unsigned int out_height=oshape[3];
      arm_compute::TensorShape out_shape(out_width, out_height,out_channels,out_num);
      if (!this->init_layer_) return;
      this->init_layer_=false;
      // Initialize ACL.
      if (is_gpu_) {
          new_gpulayer();
      }else{
          new_cpulayer();
      }

      this->force_bypass_acl_path_=false;
      DType * output_data =out_data[concat_enum::kOut].dptr<DType>();

      if (is_gpu_) {
          for (int i = 0; i < in_data.size(); ++i) {
            const TShape& ishape=in_data[i].shape_;
            DType * input_data =in_data[i].dptr<DType>();
            unsigned int in_num=ishape[0];
            unsigned int in_channels=ishape[1];
            unsigned int in_width=ishape[2];
            unsigned int in_height=ishape[3];
            arm_compute::TensorShape vec_shape(in_width, in_height,in_channels);
            GPUTensor *vector;
            new_tensor(vector,vec_shape,(void*)input_data);
            tensor_mem(vector,(void*)input_data);
            vector->commit();
            gpu_vectors.push_back(vector);
          }
          new_tensor(this->gpu().output,out_shape,(void*)output_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->gpu().layer->configure(gpu_vectors,this->gpu().output);
      }else{
          for (int i = 0; i < in_data.size(); ++i) {
            const TShape& ishape=in_data[i].shape_;
            DType * input_data =in_data[i].dptr<DType>();
            unsigned int in_num=ishape[0];
            unsigned int in_channels=ishape[1];
            unsigned int in_width=ishape[2];
            unsigned int in_height=ishape[3];
            arm_compute::TensorShape vec_shape(in_width, in_height,in_channels);
            CPUTensor *vector;
            new_tensor(vector,vec_shape,(void*)input_data);
            tensor_mem(vector,(void*)input_data);
            vector->commit();
            cpu_vectors.push_back(vector);
          }
          new_tensor(this->cpu().output,out_shape,(void*)output_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->cpu().layer->configure(cpu_vectors,this->cpu().output);
      }
  }

 public:
  explicit ACLConcatOp(Context & ctx,ConcatParam p)
      : ConcatOp<xpu, DType>(p) {
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
      const TShape& oshape=out_data[concat_enum::kOut].shape_;
      const TShape& ishape=out_data[concat_enum::kData0].shape_;
      if (this->force_bypass_acl_path_||oshape[0]!=ishape[0]){
          ConcatOp<xpu, DType>::Forward(ctx,in_data,req,out_data,aux_args);
          return;
      }
      DType * output_data =out_data[concat_enum::kOut].dptr<DType>();
      SetupACLLayer(ctx,in_data,req,out_data,aux_args);
      acl_run(nullptr,(void*)output_data,is_gpu_);
  }
};  // class ACLConcatOp
}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_ACL_CONCAT_INL_H_
