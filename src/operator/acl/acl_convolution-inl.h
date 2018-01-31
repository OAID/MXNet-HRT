/*!
 * Copyright (c) 2016 by Contributors
 * \file acl_convolution-inl.h
 * \brief
 * \author Joey
*/
#ifndef MXNET_OPERATOR_ACL_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_ACL_CONVOLUTION_INL_H_

#if USE_ACL == 1
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../convolution-inl.h"
#include "acl_operator.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class ACLConvolutionOp : public ConvolutionOp<xpu, DType>,ACLOperator {
 private:
  ConvolutionParam param_;
  Context ctx_;

  void SetupACLLayer(const OpContext &ctx, const std::vector<TBlob> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &out_data,
                     const std::vector<TBlob> &aux_args){

      const TShape& ishape=in_data[conv::kData].shape_;
      const TShape& oshape=out_data[conv::kOut].shape_;
      arm_compute::TensorShape input_shape((unsigned int)ishape[3],(unsigned int)ishape[2], (unsigned int)ishape[1],(unsigned int)ishape[0]); //wxhxchxnum
      if (is_operator_init_done(input_shape)) return;
      set_operator_init_done();

      // Initialize ACL.
      this->force_bypass_acl_path_=false;

      TShape stride;
      TShape dilate;
      TShape pad;
      TShape kernel_shape;
      if (param_.kernel.ndim() == 1) {
        stride[1]=stride[0]=param_.stride[0];
        dilate[1]=dilate[0]=param_.dilate[0];
        pad[1]=pad[0]=param_.pad[0];
        kernel_shape[1]=kernel_shape[0]=param_.kernel[0];
      } else if (param_.kernel.ndim() == 2) {
        stride[1]=param_.stride[0];stride[0]=param_.stride[1];
        dilate[1]=param_.dilate[0];dilate[0]=param_.dilate[1];
        pad[1]=param_.pad[0];pad[0]=param_.pad[1];
        kernel_shape[1]=param_.kernel[0];kernel_shape[0]=param_.kernel[1];
      } else{
          stride[1]=stride[0]=0;
          dilate[1]=dilate[0]=0;
          pad[1]=pad[0]=0;
          kernel_shape[1]=kernel_shape[0]=0;
      }

      int stride_x =stride[1];
      int stride_y =stride[0];
      int pad_x=pad[1];
      int pad_y=pad[0];
      unsigned int channels = ishape[1];
      unsigned int num_output=oshape[1];
      unsigned int kernel_x=kernel_shape[1];
      unsigned int kernel_y=kernel_shape[0];
      arm_compute::PadStrideInfo conv_info(stride_x,stride_y,pad_x,pad_y);
      arm_compute::TensorShape weights_shape(kernel_x,kernel_y,channels/param_.num_group, num_output);
      arm_compute::TensorShape biases_shape (num_output);
      arm_compute::TensorShape output_shape((unsigned int)oshape[3],(unsigned int)oshape[2], (unsigned int)oshape[1],(unsigned int)oshape[0]);//wxhxchxnum
      group()=param_.num_group;

      //[kernel_x, kernel_y, IFM, OFM]
      new_tensor(weights(),weights_shape,GetDataPtr<DType>(ACLOp_Ptr(this),in_data,conv::kWeight));
      //[OFM]
      if (!param_.no_bias) {
          new_tensor(biases(),biases_shape,GetDataPtr<DType>(ACLOp_Ptr(this),in_data,conv::kBias));
      }

      //[width, height, IFM]
      new_tensor(input(),input_shape,InputdataPtr<DType>(ACLOp_Ptr(this),in_data,conv::kData));
      //[width, height, OFM]
      new_tensor(output(),output_shape,OutputdataPtr<DType>(ACLOp_Ptr(this),out_data,conv::kOut));
      acl_configure(conv,this,conv_info);
  }
  bool Bypass_acl() {
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_|| param_.num_group >=5){//for performance, more groups impact GPU performance
        bypass_acl=true;
    }
    if (param_.kernel.ndim()>2 || param_.stride.ndim() == 0 || param_.pad.ndim() ==0 || param_.dilate.ndim() == 0) {
        bypass_acl=true;
    }
    /* check dilation */
    int dilated=0;

    for(unsigned int i=0;i<param_.dilate.ndim();i++)
    {
        if(param_.dilate[i]!=1) 
           dilated=1;
    }

    if(dilated) {
        bypass_acl=true;
    }
    return bypass_acl;
  }
  void check_direct_conv(){
      bool use_direct_conv=false;
      const char* pDirectConv;
      pDirectConv = getenv ("DIRECTCONV");
      if (pDirectConv){
        unsigned int bdirectconv;
        sscanf(pDirectConv,"%i", &bdirectconv);
        if(bdirectconv != use_direct_conv){
            use_direct_conv = bdirectconv;
            printf("DIRECTCONV<%s>\n", pDirectConv);
            printf("DIRECTCONV: %x\n", use_direct_conv);
        }
      }
      int pad_data[2],kernel[2];
      if (param_.kernel.ndim() == 1) {
        pad_data[1]=pad_data[0]=param_.pad[0];
        kernel[1]=kernel[0]=param_.kernel[0];
      } else if (param_.kernel.ndim() == 2) {
        pad_data[1]=param_.pad[0];pad_data[0]=param_.pad[1];
        kernel[1]=param_.kernel[0];kernel[0]=param_.kernel[1];
      } else{
          pad_data[0]=0;pad_data[1]=0;
          kernel[0]=0;kernel[1]=0;
      }
      if (use_direct_conv && ( (kernel[0]==1 && kernel[1]==1 &&pad_data[0]==0 && pad_data[1]==0) || (kernel[0]==3 && kernel[1]==3 && pad_data[0]<=1 && pad_data[1] <=1 ) )) {
          setConvMethod(); //NEDirectConvolutionLayer only for 1x1 and 3x3
      }

  }

 public:
  explicit ACLConvolutionOp(Context & ctx,ConvolutionParam p)
      : ConvolutionOp<xpu, DType>(p) , ACLOperator(ctx.arm_gpu_mode()){
    this->param_ = p;
    this->ctx_ = ctx;
    this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_CONV;
    check_direct_conv();
  }

 public:
  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
#ifdef USE_PROFILING
    logtime_util log_time(ACL_CONV_INFO);
#endif //USE_PROFILING
      if (Bypass_acl()){
         ConvolutionOp<xpu, DType>::Forward(ctx,in_data,req,out_data,aux_args);
         return;
      }
      DType * input_data =in_data[conv::kData].dptr<DType>();
      DType * output_data =out_data[conv::kOut].dptr<DType>();
      SetupACLLayer(ctx,in_data,req,out_data,aux_args);
      mxnet::op::acl_run<DType>(ACLOp_Ptr(this),in_data,out_data);
  }
};  // class ACLConvolutionOp
}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_ACL_CONVOLUTION_INL_H_
