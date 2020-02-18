// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_integer.h"
#include "matmul_integer.cuh"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    int8_t,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger<int8_t, int8_t>);

template <>
Status MatMulInteger<int8_t, int8_t>::ComputeInternal(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(1);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // validate zero points
  const int8_t* a_offset = nullptr;
  const int8_t* b_offset = nullptr;
  if (has_a_zero_point_) {
    auto a_zero_point = ctx->Input<Tensor>(2);
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point),
                "MatmulInteger : input1 zero point must be a scalar or 1D tensor of size 1");
    a_offset = a_zero_point->template Data<int8_t>();
  }
  if (has_b_zero_point_) {
    auto b_zero_point = ctx->Input<Tensor>(3);
    ORT_ENFORCE(IsScalarOr1ElementVector(b_zero_point),
                "MatmulInteger : input2 zero point must be a scalar or 1D tensor of size 1");
    b_offset = b_zero_point->template Data<int8_t>();
  }

  // intialize output c[i,j] to
  // k*a_offset*b_offset -
  // b_offset * (a[i,0] + a[i,1] ...+a[i,k]) -
  // a_offset * (b[0,j] + b[1,j] ... + b[k,j])
  IAllocatorUniquePtr<int32_t> a_row_buf = GetScratchBuffer<int32_t>(helper.OutputShape().Size() / helper.N());
  IAllocatorUniquePtr<int32_t> b_col_buf = GetScratchBuffer<int32_t>(helper.OutputShape().Size() / helper.M());
  ReduceSumOnLastAxis(a->template Data<int8_t>(), a_row_buf.get(), b_offset, helper);
  ReduceSumOnSecondToLastAxis(b->template Data<int8_t>(), b_col_buf.get(), a_offset, helper);
  InitializeOutput(a_row_buf.get(),
                   b_col_buf.get(),
                   Y->template MutableData<int32_t>(),
                   a_offset,
                   b_offset,
                   helper);

  int alpha = 1;
  int beta = 1;
  if (helper.OutputOffsets().size() == 1) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmEx(
        Base::CublasHandle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(helper.N()),
        static_cast<int>(helper.M()),
        static_cast<int>(helper.K()),
        &alpha,
        b->template Data<int8_t>(),
        CUDA_R_8I,
        static_cast<int>(helper.N()),
        a->template Data<int8_t>(),
        CUDA_R_8I,
        static_cast<int>(helper.K()),
        &beta,
        Y->template MutableData<int32_t>(),
        CUDA_R_32I,
        static_cast<int>(helper.N()),
        CUDA_R_32I,
        CUBLAS_GEMM_DFALT));
    return Status::OK();
  }

  CudaAsyncBuffer<const int8_t*> left_arrays(this, helper.LeftOffsets().size());
  CudaAsyncBuffer<const int8_t*> right_arrays(this, helper.RightOffsets().size());
  CudaAsyncBuffer<float*> output_arrays(this, helper.OutputOffsets().size());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const int8_t*>(a->template Data<int8_t>()), helper.LeftOffsets(), left_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const int8_t*>(b->template Data<int8_t>()), helper.RightOffsets(), right_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<float*>(Y->template MutableData<int8_t>()), helper.OutputOffsets(), output_arrays.CpuSpan());
  ORT_RETURN_IF_ERROR(left_arrays.CopyToGpu());
  ORT_RETURN_IF_ERROR(right_arrays.CopyToGpu());
  ORT_RETURN_IF_ERROR(output_arrays.CopyToGpu());

  for (int batch = 0; batch < helper.OutputOffsets().size(); batch++) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmEx(
        Base::CublasHandle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(helper.N()),
        static_cast<int>(helper.M()),
        static_cast<int>(helper.K()),
        &alpha,
        right_arrays.GpuPtr()[batch],
        CUDA_R_8I,
        static_cast<int>(helper.N()),
        left_arrays.GpuPtr()[batch],
        CUDA_R_8I,
        static_cast<int>(helper.K()),
        &beta,
        output_arrays.GpuPtr()[batch],
        CUDA_R_32I,
        static_cast<int>(helper.N()),
        CUDA_R_32I,
        CUBLAS_GEMM_DFALT));
  }

  //float one = 1.f;
  //float zero = 0.f;
  // note that onnxruntime OrtValue is row major, while cublas is column major,
  // so swap left/right operands
  //CUBLAS_RETURN_IF_ERROR(cublasGemmBatchedEx(
  //    Base::CublasHandle(),
  //    CUBLAS_OP_N,
  //    CUBLAS_OP_N,
  //    static_cast<int>(helper.N()),
  //    static_cast<int>(helper.M()),
  //    static_cast<int>(helper.K()),
  //    &one,
  //    reinterpret_cast<const void* const*>(right_arrays.GpuPtr()),
  //    CUDA_R_8I,
  //    static_cast<int>(helper.N()),
  //    reinterpret_cast<const void* const*>(left_arrays.GpuPtr()),
  //    CUDA_R_8I,
  //    static_cast<int>(helper.K()),
  //    &zero,
  //    reinterpret_cast<void**>(output_arrays.GpuPtr()),
  //    CUDA_R_32F,
  //    static_cast<int>(helper.N()),
  //    static_cast<int>(helper.OutputOffsets().size()),
  //    CUDA_R_32F,
  //    CUBLAS_GEMM_DFALT));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
