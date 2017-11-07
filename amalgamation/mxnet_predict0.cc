// mxnet.cc

#define MSHADOW_FORCE_STREAM

#ifndef MSHADOW_USE_CBLAS
#if (__MIN__ == 1)
#define MSHADOW_USE_CBLAS   0
#else
#define MSHADOW_USE_CBLAS   1
#endif
#endif

#define MSHADOW_USE_CUDA    0
#define MSHADOW_USE_MKL     0
#define MSHADOW_RABIT_PS    0
#define MSHADOW_DIST_PS     0

#if defined(__ANDROID__) || defined(__MXNET_JS__)
#define MSHADOW_USE_SSE         0
#endif

#define MXNET_USE_OPENCV    0
#define MXNET_PREDICT_ONLY  1
#define DISABLE_OPENMP 1
#define DMLC_LOG_STACK_TRACE 0

#include "nnvm/src/c_api/c_api_error.cc"
#include "nnvm/src/c_api/c_api_symbolic.cc"
#include "nnvm/src/core/graph.cc"
#include "nnvm/src/core/node.cc"
#include "nnvm/src/core/op.cc"
#include "nnvm/src/core/pass.cc"
#include "nnvm/src/core/symbolic.cc"
#include "nnvm/src/pass/infer_shape_type.cc"
#include "nnvm/src/pass/plan_memory.cc"
#include "nnvm/src/pass/saveload_json.cc"
#include "nnvm/src/pass/gradient.cc"

#include "src/ndarray/autograd.cc"
#include "src/ndarray/ndarray_function.cc"
#include "src/ndarray/ndarray.cc"

#include "src/engine/engine.cc"
#include "src/engine/naive_engine.cc"
#include "src/engine/profiler.cc"

#include "src/executor/graph_executor.cc"
#include "src/executor/attach_op_execs_pass.cc"
#include "src/executor/attach_op_resource_pass.cc"
#include "src/executor/inplace_addto_detect_pass.cc"

#include "src/io/io.cc"

#include "src/nnvm/legacy_json_util.cc"
#include "src/nnvm/legacy_op_util.cc"

#include "src/kvstore/kvstore.cc"

#include "src/operator/operator.cc"
#include "src/operator/operator_util.cc"
#include "src/operator/batch_norm.cc"
#include "src/operator/concat.cc"
#include "src/operator/deconvolution.cc"
#include "src/operator/dropout.cc"
#include "src/operator/leaky_relu.cc"
#include "src/operator/softmax_activation.cc"
#include "src/operator/tensor/elemwise_binary_broadcast_op_basic.cc"
#include "src/operator/tensor/elemwise_binary_op_basic.cc"
#include "src/operator/tensor/elemwise_binary_scalar_op_basic.cc"
#include "src/operator/tensor/elemwise_unary_op.cc"
#include "src/operator/tensor/elemwise_sum.cc"
#include "src/operator/tensor/matrix_op.cc"
#include "src/operator/tensor/sample_op.cc"
#include "src/operator/tensor/init_op.cc"
#include "src/operator/custom/custom.cc"
#include "src/operator/lrn.cc"
#include "src/operator/activation.cc"
#include "src/operator/convolution.cc"
#include "src/operator/fully_connected.cc"
#include "src/operator/pooling.cc"
#include "src/operator/softmax_output.cc"
#include "src/operator/acl/acl_layer.cc"

#include "src/storage/storage.cc"

#include "src/resource.cc"
#include "src/initialize.cc"

#include "src/c_api/c_predict_api.cc"
#include "src/c_api/c_api_symbolic.cc"
#include "src/c_api/c_api_ndarray.cc"
#include "src/c_api/c_api_error.cc"
#include "src/c_api/c_api_executor.cc"
#include "src/c_api/c_api.cc"
