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

#ifndef XLA_SERVICE_GPU_MOCK_NCCL_UTILS_H_
#define XLA_SERVICE_GPU_MOCK_NCCL_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "xla/service/gpu/nccl_p2p_thunk_common.h"
#include "xla/service/gpu/nccl_utils.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

constexpr int kNcclNumFunctions = 5;
constexpr int kNcclNumProtocols = 3;
constexpr int kNcclNumAlgorithms = 6;  // Tree/Ring/CollNet*

struct MockNcclComm {
  int rank;       // my rank in the communicator
  int num_ranks;  // number of GPUs in communicator
  int num_nodes;
  bool collnet_support;
  int num_channels;
  int min_comp_cap;
  int max_comp_cap;
  // Algorithm/Protocols thresholds
  size_t thread_thresholds[kNcclNumAlgorithms][kNcclNumProtocols];
  float latencies[kNcclNumFunctions][kNcclNumAlgorithms][kNcclNumProtocols];
  float bandwidths[kNcclNumFunctions][kNcclNumAlgorithms][kNcclNumProtocols];
  int max_threads[kNcclNumAlgorithms][kNcclNumProtocols];
};

using MockNcclComm_t = MockNcclComm*;

StatusOr<std::unique_ptr<MockNcclComm>> InitializeMockNcclComm(
    const NcclExecuteParams& params,
    const std::vector<ReplicaGroup>& replica_groups,
    CollectiveOpGroupMode group_mode, int64_t op_id, int64_t stream_id,
    bool enable_clique_optimization);

Status RunMockNcclCollectives(std::vector<DeviceBufferPair>& buffers,
                              se::Stream& stream, MockNcclComm_t comm,
                              Thunk::Kind reduce_op);

Status RunMockNcclAllToAll(bool has_split_dimension,
                           std::vector<DeviceBufferPair>& buffers,
                           se::Stream& stream, MockNcclComm_t comm);

Status RunMockCollectivePermute(
    NcclP2PConfig::SourceTargetMapEntry source_target, DeviceBufferPair& buffer,
    se::Stream& stream, MockNcclComm_t comm, absl::string_view device_string,
    int64_t current_id);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MOCK_NCCL_UTILS_H_
