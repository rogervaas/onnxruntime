// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_integer.cuh"

#include <cub/cub.cuh>
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <int TPB>
__global__ void ReduceSumOnLastAxisOneMatrix(const int8_t* a, int32_t* a_row_sum, const int8_t* offset, int32_t K) {
  int32_t thread_data = 0;
  const int8_t* a_row = a + blockIdx.x * K;
  for (int i = threadIdx.x; i < K; i += TPB) {
    thread_data += *(a_row + i);
  }

  using BlockReduce = cub::BlockReduce<int32_t, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int32_t sum = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) {
    a_row_sum[blockIdx.x] = (*offset) * sum;
  }
}

Status ReduceSumOnLastAxis(const int8_t* a, int32_t* row_sum, const int8_t* offset, const MatMulComputeHelper& helper) {
  for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
    ReduceSumOnLastAxisOneMatrix<static_cast<int>(GridDim::maxThreadsPerBlock)>
        <<<helper.M(), GridDim::maxThreadsPerBlock, 0>>>(a + helper.LeftOffsets()[batch],
                                                         row_sum + batch * helper.M(),
                                                         offset,
                                                         helper.K());
  }

  return Status::OK();
}

template <int TPB>
__global__ void ReduceSumOnSecondToLastAxisOneMatrix(const int8_t* b, int32_t* col_sum, const int8_t* offset, int32_t K, int32_t N) {
  int32_t thread_data = 0;
  const int8_t* col = b + blockIdx.x;
  for (int i = threadIdx.x; i < K; i += TPB) {
    thread_data += *(col + i * N);
  }

  using BlockReduce = cub::BlockReduce<int32_t, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int32_t sum = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) {
    col_sum[blockIdx.x] = (*offset) * sum;
  }
}

Status ReduceSumOnSecondToLastAxis(const int8_t* b, int32_t* col_sum, const int8_t* offset, const MatMulComputeHelper& helper) {
  for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
    ReduceSumOnSecondToLastAxisOneMatrix<static_cast<int>(GridDim::maxThreadsPerBlock)>
        <<<helper.N(), GridDim::maxThreadsPerBlock, 0>>>(b + helper.RightOffsets()[batch],
                                                         col_sum + batch * helper.N(),
                                                         offset,
                                                         helper.K(),
                                                         helper.N());
  }

  return Status::OK();
}

template <int TPB>
__global__ void InitializeMatrix(const int32_t* row_sum, const int32_t* col_sum, int32_t* output, const int8_t* a_offset, const int8_t* b_offset, int32_t K, int32_t N) {
  for (int32_t i = threadIdx.x; i < N; i += TPB) {
    *(output + blockIdx.x * N + i) = K * (*a_offset) * (*b_offset) + row_sum[blockIdx.x] + col_sum[i];
  }
}

Status InitializeOutput(const int32_t* row_sum,
                        const int32_t* col_sum,
                        int32_t* output,
                        const int8_t* a_offset,
                        const int8_t* b_offset,
                        const MatMulComputeHelper& helper) {
  int blocksPerGrid = static_cast<int>(CeilDiv(helper.M() * helper.N(), GridDim::maxThreadsPerBlock));
  for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
    InitializeMatrix<static_cast<int>(GridDim::maxThreadsPerBlock)>
        <<<helper.M(), GridDim::maxThreadsPerBlock, 0>>>(row_sum + batch * helper.M(),
                                                         col_sum + batch * helper.N(),
                                                         output + helper.OutputOffsets()[batch],
                                                         a_offset,
                                                         b_offset,
                                                         helper.K(),
                                                         helper.N());
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
