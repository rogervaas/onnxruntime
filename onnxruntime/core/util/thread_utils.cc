#include "thread_utils.h"
#include <algorithm>

#include <core/common/make_unique.h>

namespace onnxruntime {
namespace concurrency {

std::unique_ptr<ThreadPool> CreateThreadPool(int thread_pool_size, bool allow_spinning,
                                             ThreadPool::ThreadEnvironment& env, EigenAllocator* allocator) {
  if (thread_pool_size <= 0) {  // default
    thread_pool_size = std::max<int>(1, std::thread::hardware_concurrency() / 2);
  }

  // since we use the main thread for execution we don't have to create any threads on the thread pool when
  // the requested size is 1. For other cases, we will have thread_pool_size + 1 threads for execution  
  return thread_pool_size == 1
             ? nullptr
                               : onnxruntime::make_unique<ThreadPool>(thread_pool_size, allow_spinning, env, allocator);
}
}  // namespace concurrency
}  // namespace onnxruntime