// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/exceptions.h"

// Setup the infrastructure to throw ORT exceptions so they're caught by existing handlers.

template <typename E>
class SafeIntExceptionHandler;

template <>
class SafeIntExceptionHandler<onnxruntime::OnnxRuntimeException> {
 public:
  static __declspec(noreturn) void __stdcall SafeIntOnOverflow() {
    ORT_THROW("Integer overflow");
  }

  static __declspec(noreturn) void __stdcall SafeIntOnDivZero() {
    ORT_THROW("Divide by zero");
  }
};

// Add two #defines so that failure throws, and it throws OnnxRuntimeException so it's caught by existing handlers
#define SAFEINT_EXCEPTION_HANDLER_CPP 1
#define SafeIntDefaultExceptionHandler SafeIntExceptionHandler<onnxruntime::OnnxRuntimeException>
#include "safeint/SafeInt.hpp"
