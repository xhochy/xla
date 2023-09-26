/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/mock_nccl_sleep_kernel.h"

namespace xla::gpu {
namespace {

__global__ void mock_nccl_call(unsigned sleep_ns) {
#if __CUDA_ARCH__ >= 700  // __nanosleep requires compute capability 7.0
  // Passing too high a number to __nanosleep makes it sleep for much less time
  // than the passed-in number. So only pass 1,000 and keep calling
  // __nanosleep in a loop.
  for (int64_t i = 0; i + 1000 < sleep_ns; i += 1000) __nanosleep(1000);
  return;
#endif
}

}  // namespace

void* GetSleepKernel() { return reinterpret_cast<void*>(&mock_nccl_call); }

}  // namespace xla::gpu
