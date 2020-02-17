//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// Make generic operators for floating point types
/* This file contains:
   Generalized library calls
   kernels to be called for not supported data type
*/
// NV_TODO: optimize speed -- pass things needed in, optimize kernel speed, add half2
// NV_TODO: investigate cub support for half

#pragma once

#include "core/providers/hip/hip_common.h"

// Generalize library calls to be use in template functions

// gemm
inline hipblasStatus_t hipblasGemmHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) {
  return hipblasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline hipblasStatus_t hipblasGemmHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) {
  return hipblasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
// inline hipblasStatus_t hipblasGemmHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const half* alpha, const half* A, int lda, const half* B, int ldb, const half* beta, half* C, int ldc) {
//   // Disable below to make sure merged result is on par with before-merge.
//   // This does true FP16 computation which is slow for non-Volta GPUs
//   //if (onnxruntime::hip::DeviceProp().GetDeviceProps().major >= 7) {
//   //   onnxruntime::hip::CublasMathModeSetter math_mode_setter( handle, CUBLAS_TENSOR_OP_MATH );
//   //  return hipblasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
//   //}
//   // This does pseudo FP16 computation (input/output in fp16, computation in fp32)
//   float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
//   float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
//   hipblasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
//   return hipblasGemmEx(handle, transa, transb, m, n, k, &h_a, A, HIPBLAS_R_16F, lda, B, HIPBLAS_R_16F, ldb, &h_b, C, HIPBLAS_R_16F, ldc, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT);
// }

// batched gemm
inline hipblasStatus_t hipblasGemmBatchedHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const float* alpha, const float* Aarray[], int lda, const float* Barray[], int ldb, const float* beta, float* Carray[], int ldc, int batchCount) {
  return hipblasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}
inline hipblasStatus_t hipblasGemmBatchedHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const double* alpha, const double* Aarray[], int lda, const double* Barray[], int ldb, const double* beta, double* Carray[], int ldc, int batchCount) {
  return hipblasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}
// inline hipblasStatus_t hipblasGemmBatchedHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const half* alpha, const half* Aarray[], int lda, const half* Barray[], int ldb, const half* beta, half* Carray[], int ldc, int batchCount) {
//   hipblasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
//   return hipblasHgemmBatched(handle, transa, transb, m, n, k, alpha, (const __half**)Aarray, lda, (const __half**)Barray, ldb, beta, (__half**)Carray, ldc, batchCount);
// }

// strided batched gemm
inline hipblasStatus_t hipblasGemmStridedBatchedHelper(hipblasHandle_t handle,
                                                     hipblasOperation_t transa,
                                                     hipblasOperation_t transb,
                                                     int m, int n, int k,
                                                     const float* alpha,
                                                     const float* A, int lda,
                                                     long long int strideA,
                                                     const float* B, int ldb,
                                                     long long int strideB,
                                                     const float* beta,
                                                     float* C, int ldc,
                                                     long long int strideC,
                                                     int batchCount) {
  return hipblasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

inline hipblasStatus_t hipblasGemmStridedBatchedHelper(hipblasHandle_t handle,
                                                     hipblasOperation_t transa,
                                                     hipblasOperation_t transb,
                                                     int m, int n, int k,
                                                     const double* alpha,
                                                     const double* A, int lda,
                                                     long long int strideA,
                                                     const double* B, int ldb,
                                                     long long int strideB,
                                                     const double* beta,
                                                     double* C, int ldc,
                                                     long long int strideC,
                                                     int batchCount){
  return hipblasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

// inline hipblasStatus_t hipblasGemmStridedBatchedHelper(hipblasHandle_t handle,
//                                                      hipblasOperation_t transa,
//                                                      hipblasOperation_t transb,
//                                                      int m, int n, int k,
//                                                      const __half* alpha,
//                                                      const __half* A, int lda,
//                                                      long long int strideA,
//                                                      const __half* B, int ldb,
//                                                      long long int strideB,
//                                                      const __half* beta,
//                                                      __half* C, int ldc,
//                                                      long long int strideC,
//                                                      int batchCount) {
//   hipblasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
//   return hipblasHgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
// }

// // axpy
// inline hipblasStatus_t hipblasAxpyHelper(hipblasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy) {
//   return hipblasSaxpy(handle, n, alpha, x, incx, y, incy);
// }
// inline hipblasStatus_t hipblasAxpyHelper(hipblasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy) {
//   return hipblasDaxpy(handle, n, alpha, x, incx, y, incy);
// }
// inline hipblasStatus_t hipblasAxpyHelper(hipblasHandle_t handle, int n, const half* alpha, const half* x, int incx, half* y, int incy) {
//   float tmp_alpha = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
//   return hipblasAxpyEx(handle, n, (void*)&tmp_alpha, HIPBLAS_R_32F, (void*)x, HIPBLAS_R_16F, incx, (void*)y, HIPBLAS_R_16F, incy, HIPBLAS_R_32F);
// }

// transpose using geam
inline hipblasStatus_t hipblasTransposeHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc) {
  return hipblasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline hipblasStatus_t hipblasTransposeHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc) {
  return hipblasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
//hipblasStatus_t hipblasTransposeHelper(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int m, int n, const half*, const half* A, int, const half*, const half*, int, half* C, int);

// // asum
// inline hipblasStatus_t hipblasAsumHelper(hipblasHandle_t handle, int n, const float* x, int incx, float* result) {
//   return hipblasSasum(handle, n, x, incx, result);
// }
// inline hipblasStatus_t hipblasAsumHelper(hipblasHandle_t handle, int n, const double* x, int incx, double* result) {
//   return hipblasDasum(handle, n, x, incx, result);
// }
// inline hipblasStatus_t hipblasAsumHelper(hipblasHandle_t, int n, const half* x, int incx, half* result) {
//   // pass in cudnn handle/descriptor to remove overhead?
//   hipdnnHandle_t cudnnHandle;
//   hipdnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
//   hipdnnReduceTensorDescriptor_t reduceTensorDesc;

//   hipdnnCreate(&cudnnHandle);
//   hipdnnCreateTensorDescriptor(&srcTensorDesc);
//   hipdnnCreateTensorDescriptor(&dstTensorDesc);
//   hipdnnCreateReduceTensorDescriptor(&reduceTensorDesc);

//   hipdnnSetTensor4dDescriptorEx(srcTensorDesc, HIPDNN_DATA_HALF, 1, 1, 1, n, 1, 1, 1, incx);
//   hipdnnSetTensor4dDescriptorEx(dstTensorDesc, HIPDNN_DATA_HALF, 1, 1, 1, 1, 1, 1, 1, 1);
//   hipdnnSetReduceTensorDescriptor(reduceTensorDesc,
//                                  HIPDNN_REDUCE_TENSOR_NORM1,
//                                  HIPDNN_DATA_FLOAT,
//                                  HIPDNN_NOT_PROPAGATE_NAN,
//                                  HIPDNN_REDUCE_TENSOR_NO_INDICES,
//                                  HIPDNN_32BIT_INDICES);

//   void* workspace = NULL;
//   size_t workspaceSizeInBytes = 0;
//   hipdnnGetReductionWorkspaceSize(cudnnHandle, reduceTensorDesc, srcTensorDesc, dstTensorDesc, &workspaceSizeInBytes);
//   if (workspaceSizeInBytes > 0) hipMalloc(&workspace, workspaceSizeInBytes);

//   float alpha = 1.0f;
//   float beta = 0.0f;

//   void* d_res;
//   hipMalloc(&d_res, sizeof(half));

//   hipdnnReduceTensor(cudnnHandle,
//                     reduceTensorDesc,
//                     NULL,
//                     0,
//                     workspace,
//                     workspaceSizeInBytes,
//                     &alpha,
//                     srcTensorDesc,
//                     (void*)x,
//                     &beta,
//                     dstTensorDesc,
//                     d_res);

//   hipMemcpy((void*)result, d_res, sizeof(half), hipMemcpyDeviceToHost);

//   hipdnnDestroyReduceTensorDescriptor(reduceTensorDesc);
//   hipdnnDestroyTensorDescriptor(srcTensorDesc);
//   hipdnnDestroyTensorDescriptor(dstTensorDesc);
//   hipdnnDestroy(cudnnHandle);
//   hipFree(d_res);
//   hipFree(workspace);

//   return (hipblasStatus_t)0;
// }

// // amax
// inline hipblasStatus_t hipblasAmaxHelper(hipblasHandle_t handle, int n, const float* x, int incx, int* result) {
//   return hipblasIsamax(handle, n, x, incx, result);
// }
// inline hipblasStatus_t hipblasAmaxHelper(hipblasHandle_t handle, int n, const double* x, int incx, int* result) {
//   return hipblasIdamax(handle, n, x, incx, result);
// }
// inline hipblasStatus_t hipblasAmaxHelper(hipblasHandle_t, int n, const half* x, int incx, int* result) {
//   unsigned int h_result_uint = 0;
//   // pass in cudnn handle/descriptor to remove overhead?
//   hipdnnHandle_t cudnnHandle;
//   hipdnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
//   hipdnnReduceTensorDescriptor_t reduceTensorDesc;

//   hipdnnCreate(&cudnnHandle);
//   hipdnnCreateTensorDescriptor(&srcTensorDesc);
//   hipdnnCreateTensorDescriptor(&dstTensorDesc);
//   hipdnnCreateReduceTensorDescriptor(&reduceTensorDesc);

//   hipdnnSetTensor4dDescriptorEx(srcTensorDesc, HIPDNN_DATA_HALF, 1, 1, 1, n, 1, 1, 1, incx);
//   hipdnnSetTensor4dDescriptorEx(dstTensorDesc, HIPDNN_DATA_HALF, 1, 1, 1, 1, 1, 1, 1, 1);
//   hipdnnSetReduceTensorDescriptor(reduceTensorDesc,
//                                  HIPDNN_REDUCE_TENSOR_AMAX,
//                                  HIPDNN_DATA_FLOAT,
//                                  HIPDNN_NOT_PROPAGATE_NAN,
//                                  HIPDNN_REDUCE_TENSOR_FLATTENED_INDICES,
//                                  HIPDNN_32BIT_INDICES);

//   void* workspace = NULL;
//   size_t workspaceSizeInBytes = 0;
//   hipdnnGetReductionWorkspaceSize(cudnnHandle, reduceTensorDesc, srcTensorDesc, dstTensorDesc, &workspaceSizeInBytes);
//   if (workspaceSizeInBytes > 0) hipMalloc(&workspace, workspaceSizeInBytes);

//   float alpha = 1.0f;
//   float beta = 0.0f;
//   void* d_max;
//   hipMalloc(&d_max, sizeof(half));
//   void* d_result_uint;
//   hipMalloc(&d_result_uint, sizeof(unsigned int));

//   hipdnnReduceTensor(cudnnHandle,
//                     reduceTensorDesc,
//                     d_result_uint,
//                     sizeof(unsigned int),
//                     workspace,
//                     workspaceSizeInBytes,
//                     &alpha,
//                     srcTensorDesc,
//                     (void*)x,
//                     &beta,
//                     dstTensorDesc,
//                     d_max);

//   hipMemcpy(&h_result_uint, d_result_uint, sizeof(unsigned int), hipMemcpyDeviceToHost);

//   hipdnnDestroyReduceTensorDescriptor(reduceTensorDesc);
//   hipdnnDestroyTensorDescriptor(srcTensorDesc);
//   hipdnnDestroyTensorDescriptor(dstTensorDesc);
//   hipdnnDestroy(cudnnHandle);
//   hipFree(workspace);
//   hipFree(d_max);
//   hipFree(d_result_uint);

//   *result = (int)h_result_uint;
//   return (hipblasStatus_t)0;
// }

// scal
inline hipblasStatus_t hipblasScalHelper(hipblasHandle_t handle, int n, const float* alpha, float* x, int incx) {
  return hipblasSscal(handle, n, alpha, x, incx);
}
inline hipblasStatus_t hipblasScalHelper(hipblasHandle_t handle, int n, const double* alpha, double* x, int incx) {
  return hipblasDscal(handle, n, alpha, x, incx);
}
// inline hipblasStatus_t hipblasScalHelper(hipblasHandle_t handle, int n, const half* alpha, half* x, int incx) {
//   float tmp_alpha = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
//   return hipblasScalEx(handle, n, (void*)&tmp_alpha, HIPBLAS_R_32F, (void*)x, HIPBLAS_R_16F, incx, HIPBLAS_R_32F);
// }
inline hipblasStatus_t hipblasScalHelper(hipblasHandle_t, int, const char*, char*, int) {
  ORT_NOT_IMPLEMENTED("Unsupported template argument(char) in hipblas_scal");
}
inline hipblasStatus_t hipblasScalHelper(hipblasHandle_t, int, const short*, short*, int) {
  ORT_NOT_IMPLEMENTED("Unsupported template argument(short) in hipblas_scal");
}

// dot
inline hipblasStatus_t hipblasDotHelper(hipblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result) {
  return hipblasSdot(handle, n, x, incx, y, incy, result);
}
inline hipblasStatus_t hipblasDotHelper(hipblasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result) {
  return hipblasDdot(handle, n, x, incx, y, incy, result);
}
// inline hipblasStatus_t hipblasDotHelper(hipblasHandle_t handle, int n, const half* x, int incx, const half* y, int incy, half* result) {
//   return hipblasDotEx(handle, n, (void*)x, HIPBLAS_R_16F, incx, (void*)y, HIPBLAS_R_16F, incy, (void*)result, HIPBLAS_R_16F, HIPBLAS_R_32F);
// }

// copy
inline hipblasStatus_t hipblasCopyHelper(hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy) {
  return hipblasScopy(handle, n, x, incx, y, incy);
}
inline hipblasStatus_t hipblasCopyHelper(hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy) {
  return hipblasDcopy(handle, n, x, incx, y, incy);
}
//hipblasStatus_t hipblasCopyHelper(hipblasHandle_t handle, int n, const half* x, int incx, half* y, int incy);