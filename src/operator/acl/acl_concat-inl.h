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
#include "acl_operator.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class ACLConcatOp : public ConcatOp<xpu, DType>,public ACLOperator {
 private:
  ConcatParam param_;
  Context ctx_;

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
      if (is_operator_init_done(out_shape,tensor_output)) return;
      set_operator_init_done();

      // Initialize ACL.
      this->force_bypass_acl_path_=false;

      std::vector<arm_compute::TensorShape> shapes;
      for (int i = 0; i < in_data.size(); ++i) {
          const TShape& ishape=in_data[i].shape_;
          unsigned int in_num=ishape[0];
          unsigned int in_channels=ishape[1];
          unsigned int in_width=ishape[2];
          unsigned int in_height=ishape[3];
          arm_compute::TensorShape in_shape(in_width, in_height,in_channels);
          new_tensor(cinput(i),in_shape,InputdataPtr<DType>(ACLOp_Ptr(this),in_data,i));
      }
      new_tensor(output(),out_shape,OutputdataPtr<DType>(ACLOp_Ptr(this),out_data,concat_enum::kOut));
      acl_configure(concat,this,in_data.size());
  }

 public:
  explicit ACLConcatOp(Context & ctx,ConcatParam p)
      : ConcatOp<xpu, DType>(p) , ACLOperator(ctx.arm_gpu_mode()){
    this->param_ = p;
    this->ctx_ = ctx;
    this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_CONCAT;
  }
  bool Bypass_acl(const TShape& ishape,const TShape& oshape) {
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_||oshape[0]!=ishape[0]){
        bypass_acl=true;
    }
    return bypass_acl;
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
      if (Bypass_acl(ishape,oshape)){
          ConcatOp<xpu, DType>::Forward(ctx,in_data,req,out_data,aux_args);
          return;
      }
      SetupACLLayer(ctx,in_data,req,out_data,aux_args);
      mxnet::op::acl_run<DType>(ACLOp_Ptr(this),in_data,out_data,false);
  }
};  // class ACLConcatOp
}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_ACL_CONCAT_INL_H_
