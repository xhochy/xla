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

#include "xla/service/gpu/mock_nccl_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "third_party/gpus/cuda/include/vector_types.h"
#include "third_party/nccl/nccl.h"
#include "xla/executable_run_options.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/mock_nccl_config.h"
#include "xla/service/gpu/mock_nccl_config.pb.h"
#include "xla/service/gpu/mock_nccl_sleep_kernel.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/service/gpu/nccl_p2p_thunk_common.h"
#include "xla/service/gpu/nccl_utils.h"
#include "xla/service/gpu/thunk.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/gpu/gpu_activation.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

constexpr int kNcclMaxNthreads = 640;
constexpr int kNcclSimpleMaxNthreads = 512;
constexpr int kNcclLLMaxNthreads = 512;
constexpr int kNcclLL128MaxNthreads = 640;
constexpr int kPciBw = 12.0;  // PCI Gen3 x16
constexpr int kNcclLLThreadThreshold = 8;
constexpr int kNcclLL128ThreadThreshold = 8;
constexpr int kNcclSimpleThreadThreshold = 64;
constexpr int kNcclMaxWorkElements = 9;

enum {
  kNcclAlgoTree,
  kNcclAlgoRing,
  kNcclAlgoCollNetDirect,
  kNcclAlgoCollNetChain,
  kNcclAlgoNvls,
  kNcclAlgoNvlsTree,
};

enum {
  kNcclProtoLL,
  kNcclProtoLL128,
  kNcclProtoSimple,
};

// NVLink, PCI, Network
enum { kNcclHwNvlink, kNcclHwPci, kNcclHwNet };

/* Array indexes used below */
enum { kVoltaCompcapIdx, kAmpereCompcapIdx, kHopperCompcapIdx };

enum NcclFuncType {
  ncclFuncBroadcast,
  ncclFuncReduce,
  ncclFuncAllGather,
  ncclFuncReduceScatter,
  ncclFuncAllReduce,
  ncclFuncSendRecv,
  ncclFuncSend,
  ncclFuncRecv,
  ncclNumFuncs
};

struct MockNcclInfo {
  NcclFuncType coll;
  // NCCL Coll Args
  size_t count;
  ncclDataType_t datatype;
  MockNcclComm_t comm;
  cudaStream_t stream;

  // Computed later
  int algorithm;
  int protocol;
  uint32_t num_channels;
  uint32_t num_threads;
  size_t num_bytes;
  int channel_id;
  uint32_t sleep_duration;
};

// Latencies in us, Bandwidths in GB/s
// Tree { LL, LL128, Simple } , Ring { LL, LL128, Simple }
static constexpr float baseLat[kNcclNumAlgorithms][kNcclNumProtocols] = {
    {6.8, 14.0, 0}, {6.6, 14.0, 8.4},  // Tree, Ring
    {6.8, 14.0, 0}, {6.8, 14.0, 0},    // CollNet Direct, Chain
    {0, 0, 23.0},   {0, 0, 23.0}       // NVLS, NVLS Tree
};

static constexpr float hwLat[3][kNcclNumAlgorithms][kNcclNumProtocols] =
    {/* NVLINK */
     {/* Tree (LL/LL128/Simple)*/ {.6, 1.25, 4},
      /* Ring (LL/LL128/Simple)*/ {.6, 1.9, 3.4},
      /* CollNetDirect (Simple)*/ {0, 0, 8.0},
      /* CollNetChain (Simple)*/ {0, 0, 4.75},
      /* NVLS */ {0, 0, 0}, /* NVLSTree */ {0, 0, 0}},
     /* PCI */
     {/* Tree (LL/LL128/Simple)*/ {1.0, 1.9, 6},
      /* Ring (LL/LL128/Simple)*/ {1.0, 2.5, 5.7},
      /* CollNetDirect (Simple)*/ {0, 0, 8.0},
      /* CollNetChain (Simple)*/ {0, 0, 8.0},
      /* NVLS */ {0, 0, 0}, /* NVLSTree */ {0, 0, 0}},
     /* NET */
     {/* Tree (LL/LL128/Simple)*/ {5.0, 8.5, 14},
      /* Ring (LL/LL128/Simple)*/ {2.7, 4.0, 14.0},
      /* CollNetDirect (Simple)*/ {0, 0, 10.7},
      /* CollNetChain (Simple)*/ {0, 0, 14},
      /* NVLS */ {0, 0, 18}, /* NVLSTree */ {0, 0, 19}}};

// LL128 max BW per channel
static constexpr double llMaxBws[3][3] = {
    /* Volta-N1/Intel-N2/Intel-N4 */ {39.0, 39.0, 20.4},
    /* Ampere-N1/AMD-N2/AMD-N4 */ {87.7, 22.5 /*avg of ring & tree*/, 19.0},
    /* Hopper-N1/AMD-N2/AMD-N4 */ {87.7, 22.5 /*avg of ring & tree*/, 19.0}};

static constexpr double perChMaxRingLL128Bws[3][3] = {
    /* Volta (N1/N2/N4) */ {20.0, 20.0, 20.0},
    /* Ampere (N1/N2/N4) */ {20.0, 20.0, 20.0},
    /* Hopper (N1/N2/N4) */ {36.7, 36.7, 36.7}};
static constexpr double perChMaxTreeLL128Bws[3][3] = {
    /* Volta (N1/N2/N4) */ {20.0, 20.0, 20.0},
    /* Ampere (N1/N2/N4) */ {20.0, 20.0, 20.0},
    /* Hopper (N1/N2/N4) */ {36.7, 36.7, 29.0}};
static constexpr double perChMaxTreeBws[3][3] = {
    /* Volta (N1/N2/N4) */ {26.5, 18.5, 10.0},
    /* Ampere (N1/N2/N4) */ {24.0, 23.6, 17.8},
    /* Hopper (N1/N2/N4) */ {38.7, 41.4, 36.0}};

// Trees are not perfectly sticking to the model for medium sizes. Applying a
// static correction factor is not ideal but works quite well. Powers of two, 64
// B to 256MB.
static constexpr float treeCorrectionFactor[kNcclNumProtocols][23] = {
    {1.0, 1.0, 1.0, 1.0, .9, .8, .7, .7,  .7,  .7,  .6, .5,
     .4,  .4,  .5,  .6,  .7, .8, .9, 1.0, 1.0, 1.0, 1.0},
    {1.0, 1.0, 1.0, 1.0, 1.0, .9, .8, .8, .8, .7,  .6, .6,
     .6,  .6,  .6,  .6,  .8,  .9, .9, .9, .9, 1.0, 1.0},
    {.9, .9, .9, .9, .9, .9, .9, .8, .7, .6, .6, .5,
     .5, .5, .5, .6, .7, .8, .7, .7, .8, .9, .9}};

static int64_t log2i(int64_t n) {
  int64_t l = 0;
  while (n >>= 1) l++;
  return l;
}

StatusOr<int> GetNcclDataTypeSize(ncclDataType_t dtype) {
  switch (dtype) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclInt32:
    case ncclUint32:
      return 4;
    case ncclInt64:
    case ncclUint64:
      return 8;
    case ncclFloat16:
      return 2;
    case ncclFloat32:
      return 4;
    case ncclFloat64:
      return 8;
#if defined(__CUDA_BF16_TYPES_EXIST__) || TENSORFLOW_USE_ROCM
    case ncclBfloat16:
      return 2;
#endif
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported nccl data type: %d", dtype));
  }
}

StatusOr<NcclFuncType> ToNcclFunctionType(Thunk::Kind reduce_op) {
  switch (reduce_op) {
    case Thunk::kNcclAllReduce:
      return ncclFuncAllReduce;
    case Thunk::kNcclAllGather:
      return ncclFuncAllGather;
    case Thunk::kNcclReduceScatter:
      return ncclFuncReduceScatter;
    case Thunk::kNcclSend:
      return ncclFuncSend;
    case Thunk::kNcclRecv:
      return ncclFuncRecv;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported nccl function type: %d", reduce_op));
  }
}

Status LaunchSleepKernel(se::gpu::GpuStreamHandle gpu_stream,
                         MockNcclInfo* info) {
  void* kernel = GetSleepKernel();
  void* kernel_args[] = {&info->sleep_duration};
  dim3 gridDim = {1, 1, 1};
  dim3 blockDim = {512, 1, 1};
  cudaError_t launch_status =
      cudaLaunchKernel(kernel, gridDim, blockDim, kernel_args, 0, gpu_stream);
  if (launch_status != cudaSuccess) {
    return absl::InternalError(absl::StrCat("Failed to launch kernel: ",
                                            cudaGetErrorString(launch_status)));
  }
  return absl::OkStatus();
}

Status MockNcclTopoTuneModel(MockNcclComm* comm, int minCompCap, int maxCompCap,
                             absl::Span<MockNcclTopoGraphConfig> graphs) {
  int simpleDefaultThreads = (graphs[kNcclAlgoRing].bw_intra() *
                                  graphs[kNcclAlgoRing].num_channels() <=
                              kPciBw)
                                 ? 256
                                 : kNcclSimpleMaxNthreads;
  // Initialize max_threads with default value
  comm->max_threads[kNcclAlgoRing][kNcclProtoSimple] = simpleDefaultThreads;
  comm->max_threads[kNcclAlgoTree][kNcclProtoSimple] = kNcclSimpleMaxNthreads;
  comm->max_threads[kNcclAlgoCollNetDirect][kNcclProtoSimple] =
      comm->max_threads[kNcclAlgoCollNetChain][kNcclProtoSimple] =
          comm->max_threads[kNcclAlgoNvls][kNcclProtoSimple] =
              comm->max_threads[kNcclAlgoNvlsTree][kNcclProtoSimple] =
                  kNcclMaxNthreads;
  comm->max_threads[kNcclAlgoRing][kNcclProtoLL] =
      comm->max_threads[kNcclAlgoTree][kNcclProtoLL] = kNcclLLMaxNthreads;
  comm->max_threads[kNcclAlgoRing][kNcclProtoLL128] =
      comm->max_threads[kNcclAlgoTree][kNcclProtoLL128] = kNcclLL128MaxNthreads;

  int nNodes = comm->num_nodes;
  int nRanks = comm->num_ranks;
  if (nRanks <= 1) return absl::OkStatus();

  int compCapIndex = minCompCap >= 90   ? kHopperCompcapIdx
                     : minCompCap >= 80 ? kAmpereCompcapIdx
                                        : kVoltaCompcapIdx;

  int index2 = nNodes <= 2 ? nNodes - 1 : 2;
  // LL: for single node, we look at GPU type; for multi-node, we look at CPU
  // type. We assume the CPU type is AMD
  int index1 = nNodes == 1 ? compCapIndex : 1;

  double llMaxBw = llMaxBws[index1][index2];
  double perChMaxTreeBw = perChMaxTreeBws[compCapIndex][index2];
  double perChMaxRingLL128Bw = perChMaxRingLL128Bws[compCapIndex][index2];
  double perChMaxTreeLL128Bw = perChMaxTreeLL128Bws[compCapIndex][index2];
  // De-penalize Tree/Simple latency on Power systems to favor Tree than Ring
  float ppn = (float)nRanks /
              nNodes;  // if ppn < 2, then we are sending/receiving at the same
                       // GPU through the NIC, apply some bw discount

  int intraHw[kNcclNumAlgorithms], hw[kNcclNumAlgorithms];
  for (int a = 0; a < kNcclNumAlgorithms; a++)
    intraHw[a] =
        graphs[a].type_intra() == PATH_NVL ? kNcclHwNvlink : kNcclHwPci;
  for (int a = 0; a < kNcclNumAlgorithms; a++)
    hw[a] = nNodes == 1 ? intraHw[a] : kNcclHwNet;

  for (int coll = 0; coll < kNcclNumFunctions; coll++) {
    int nsteps = coll == ncclFuncAllReduce ? 2 * (nRanks - 1)
                 : coll == ncclFuncReduceScatter || coll == ncclFuncAllGather
                     ? nRanks - 1
                     : nRanks;
    int nInterSteps =
        coll == ncclFuncAllReduce ? (nNodes > 1 ? 2 * nNodes : 0)
        : coll == ncclFuncReduceScatter || coll == ncclFuncAllGather
            ? nNodes - 1
            : nNodes;

    for (int a = 0; a < kNcclNumAlgorithms; a++) {
      if (coll == ncclFuncBroadcast && a != kNcclAlgoRing) continue;
      if (coll == ncclFuncReduce && a != kNcclAlgoRing) continue;
      if (coll == ncclFuncReduceScatter && a != kNcclAlgoRing) continue;
      if (coll == ncclFuncAllGather && a != kNcclAlgoRing) continue;

      for (int p = 0; p < kNcclNumProtocols; p++) {
        if ((a == kNcclAlgoNvls || a == kNcclAlgoNvlsTree) &&
            p != kNcclProtoSimple)
          continue;
        int collnet =
            (a == kNcclAlgoCollNetDirect || a == kNcclAlgoCollNetChain) ? 1 : 0;
        float bw = nNodes <= 2 || collnet ? graphs[a].bw_intra()
                                          : graphs[a].bw_inter();
        float busBw = graphs[a].num_channels() * bw;

        // Various model refinements
        if (a == kNcclAlgoRing && p == kNcclProtoLL) {
          busBw = std::min(llMaxBw,
                           busBw * ((nNodes > 1 || coll == ncclFuncAllReduce ||
                                     coll == ncclFuncReduce)
                                        ? 1.0 / 4.0
                                        : 1.0 / 3.0));
        }
        if (a == kNcclAlgoRing && p == kNcclProtoLL128)
          busBw = std::min(busBw * (ppn < 2 ? 0.7 : 0.92 /*120.0/128.0*/),
                           graphs[a].num_channels() * perChMaxRingLL128Bw);
        if (a == kNcclAlgoTree)
          busBw =
              std::min(busBw * .92, graphs[a].num_channels() * perChMaxTreeBw);
        if (a == kNcclAlgoTree && p == kNcclProtoLL)
          busBw = std::min(busBw * 1.0 / 3.8, llMaxBw);
        if (a == kNcclAlgoTree && p == kNcclProtoLL128)
          busBw = std::min(busBw * (nNodes == 1 ? 7.0 / 9.0 : 120.0 / 128.0),
                           graphs[a].num_channels() * perChMaxTreeLL128Bw);
        if (a == kNcclAlgoTree && graphs[a].pattern() == NCCL_TOPO_PATTERN_TREE)
          busBw *= .85;
        if (a == kNcclAlgoCollNetDirect && p != kNcclProtoSimple)
          busBw = 0;  // Not used
        if (a == kNcclAlgoCollNetChain && p != kNcclProtoSimple)
          busBw = 0;  // Not used
        if (a == kNcclAlgoCollNetDirect && p == kNcclProtoSimple) {
          // Collnet+Direct requires all GPUs to have a local NIC to work at
          // full speed
          float factor =
              ppn / (1.0 * graphs[a].num_channels());  // GPU/NIC ratio
          factor -= (factor - 1) / 2;
          busBw /= factor;
        }
        if (a == kNcclAlgoCollNetDirect && p == kNcclProtoSimple &&
            minCompCap >= 90)
          busBw *= .85;

        // Convert bus BW to algorithm BW
        float ratio;
        if (a == kNcclAlgoRing)
          ratio = (1.0 * nRanks) / nsteps;
        else if (a == kNcclAlgoNvls)
          ratio = 5.0 / 6.0;
        else if (a == kNcclAlgoNvlsTree)
          ratio = .70 * nNodes / (2 * (nNodes - 1));
        else
          ratio = .5;
        comm->bandwidths[coll][a][p] = busBw * ratio;

        comm->latencies[coll][a][p] = baseLat[a][p];
        float intraLat = hwLat[intraHw[a]][a][p];
        float interLat = hwLat[kNcclHwNet][a][p] + graphs[a].latency_inter();
        // Also add the flush extra latency
        if (p == kNcclProtoSimple) interLat += graphs[a].latency_inter();

        if (a == kNcclAlgoRing) {
          float lat = hwLat[hw[a]][a][p];
          if ((coll == ncclFuncReduce || coll == ncclFuncBroadcast)) {
            if (graphs[a].same_channels()) {
              comm->latencies[coll][a][p] += lat;
            } else {
              if (p == kNcclProtoSimple)
                lat = hwLat[hw[a]][kNcclAlgoTree]
                           [p];  // Add some chunk latency, waiting for proper
                                 // chunk modeling
              comm->latencies[coll][a][p] += nsteps * lat;
            }
          } else {
            // Inter-node rings still have to launch nsteps * net overhead.
            float netOverhead = 0.0;
            if (nNodes > 1) {
              // assum cpu type is amd
              netOverhead = 2.0;
              if (p == kNcclProtoSimple) netOverhead *= 3;
            }
            intraLat = std::max(intraLat, netOverhead);
            comm->latencies[coll][a][p] +=
                (nsteps - nInterSteps) * intraLat + nInterSteps * interLat;
          }
        } else if (a == kNcclAlgoTree) {
          comm->latencies[coll][a][p] +=
              2 * ((nRanks / nNodes - 1) * intraLat + log2i(nNodes) * interLat);
        } else if (a == kNcclAlgoCollNetDirect) {
          comm->latencies[coll][a][p] +=
              2 * (std::min(1, (nRanks / nNodes - 1)) * intraLat +
                   (nRanks / nNodes - 1) * 0.5) +
              interLat;  // Add 0.5 arity serialization latency
        } else if (a == kNcclAlgoCollNetChain) {
          comm->latencies[coll][a][p] +=
              2 * (nRanks / nNodes - 1) * intraLat + interLat;
        } else if (a == kNcclAlgoNvls) {
          if (nNodes > 1)
            comm->latencies[coll][a][p] += hwLat[kNcclHwNet][a][p];
        } else if (a == kNcclAlgoNvlsTree) {
          comm->latencies[coll][a][p] +=
              2 * (nNodes - 1) * hwLat[kNcclHwNet][a][p];
        }
      }
    }
  }

  // Protocols/Algorithms enable/disable, and user overrides.
  // All are enabled except ll128 which is enabled by default only in certain
  // cases.
  int protoEnable[kNcclNumProtocols] = {1, 2, 1};
  int algoEnable[kNcclNumAlgorithms] = {1, 1, 1, 1, 1, 1};

  if (comm->num_nodes == 1) algoEnable[kNcclAlgoNvlsTree] = 0;

  // Disable CollNet if it is not supported
  if (comm->collnet_support == 0) {
    algoEnable[kNcclAlgoCollNetDirect] = 0;
    algoEnable[kNcclAlgoCollNetChain] = 0;
    if (comm->num_nodes > 1) algoEnable[kNcclAlgoNvls] = 0;
    // If user has hard set NCCL_ALGO=COLLNET, ignore it
    if (algoEnable[kNcclAlgoRing] == 0 && algoEnable[kNcclAlgoTree] == 0 &&
        algoEnable[kNcclAlgoNvls] == 0 && algoEnable[kNcclAlgoNvlsTree] == 0) {
      algoEnable[kNcclAlgoRing] = algoEnable[kNcclAlgoTree] = 1;
      if (comm->rank == 0)
        VLOG(1) << "CollNet is not supported or fails to initialize, ignoring "
                   "NCCL_ALGO=COLLNET";
    }
  } else {
    // Assume not an NVSwitch system, disable CollNet+Direct
    algoEnable[kNcclAlgoCollNetDirect] = 0;
  }

  for (int c = 0; c < kNcclNumFunctions; c++)
    for (int a = 0; a < kNcclNumAlgorithms; a++)
      for (int p = 0; p < kNcclNumProtocols; p++) {
        int pEnable = protoEnable[p];
        if (pEnable == 2 && p == kNcclProtoLL128) {
          // Enable LL128 by default only on Volta/Ampere/Hopper+NVLink. Other
          // cases are not tested and may cause silent data corruption.
          pEnable = 1;
          pEnable &= (graphs[a].type_inter() <= PATH_PXB ||
                      (minCompCap >= 90 && graphs[a].type_inter() <= PATH_PXN));
          pEnable &= (graphs[a].type_intra() <= PATH_NVB);
          pEnable &= (minCompCap == maxCompCap);
          switch (minCompCap) {
            case 70:
              pEnable &= 1;
              break;
            case 80:
              pEnable &= 1;
              break;
            case 90:
              pEnable &= !(CUDART_VERSION == 11080 && c == ncclFuncAllReduce &&
                           a == kNcclAlgoRing && comm->num_ranks == 2);
              break;
            default:
              pEnable &= 0;
              break;
          }
        }
        if (pEnable == 0) comm->bandwidths[c][a][p] = 0;
        // Never disable ring for non-allreduce operations. That allows to run
        // real apps with NCCL_ALGO=TREE.
        if (a == kNcclAlgoRing && c != ncclFuncAllReduce) continue;
        if (algoEnable[a] == 0) comm->bandwidths[c][a][p] = 0;
      }

  // Set per-thread amount of work before we increase nThreads and nChannels
  for (int a = 0; a < kNcclNumAlgorithms; a++) {
    comm->thread_thresholds[a][kNcclProtoLL] = kNcclLLThreadThreshold;
    comm->thread_thresholds[a][kNcclProtoLL128] = kNcclLL128ThreadThreshold;
    comm->thread_thresholds[a][kNcclProtoSimple] = kNcclSimpleThreadThreshold;
  }
  comm->thread_thresholds[kNcclAlgoRing][kNcclProtoLL] *= nRanks;
  comm->thread_thresholds[kNcclAlgoCollNetDirect][kNcclProtoSimple] = 512;
  comm->thread_thresholds[kNcclAlgoCollNetChain][kNcclProtoSimple] = 512;

  return absl::OkStatus();
}

absl::StatusOr<float> ncclTopoGetAlgoTime(MockNcclInfo* info, int algorithm,
                                          int protocol, int numPipeOps) {
  float bw = info->comm->bandwidths[info->coll][algorithm][protocol];
  float lat = info->comm->latencies[info->coll][algorithm][protocol];
  if (bw == 0) {
    return -1.0f;
  }
  int logSize = log2i(info->num_bytes >> 6);
  if (algorithm == kNcclAlgoTree && logSize < 23)
    bw *= treeCorrectionFactor[protocol][logSize];
  if (info->num_channels != 0)
    bw = bw / info->comm->num_channels * info->num_channels;
  if (algorithm == kNcclAlgoRing && protocol == kNcclProtoSimple &&
      info->comm->num_nodes > 1 && info->coll == ncclFuncAllReduce &&
      info->num_bytes / (info->comm->num_channels * info->comm->num_ranks) >=
          64) {
    lat *= info->comm->min_comp_cap < 80 ? 1.9 : 1.4;  // Plateau effect of ring
  }
  // Tree pipelining saves latency in aggregation cases
  int latCount =
      algorithm == kNcclAlgoRing
          ? numPipeOps
          : (numPipeOps + kNcclMaxWorkElements - 1) / kNcclMaxWorkElements;
  float time = lat * latCount + (info->num_bytes) / (1000 * bw);
  return time;
}

inline absl::Status MockNcclInfoSetDerived(MockNcclInfo* info, int nRanks) {
  TF_ASSIGN_OR_RETURN(int dtype_size, GetNcclDataTypeSize(info->datatype));
  info->num_bytes = info->count * dtype_size;
  if (info->coll == ncclFuncAllGather || info->coll == ncclFuncBroadcast) {
    info->count = info->num_bytes;
    info->datatype = ncclInt8;
  }
  if (info->coll == ncclFuncAllGather || info->coll == ncclFuncReduceScatter)
    info->num_bytes *= nRanks;  // count is per rank
  return absl::OkStatus();
}

// numPipeOps: number of pipelined ops. Can be greater than 1 in aggregation
// mode. Used to adjust latency.
Status getAlgoInfo(MockNcclInfo* info, MockNcclComm_t comm, int numPipeOps) {
  info->algorithm = -1;
  info->protocol = -1;
  float minTime = 360000000;
  if (info->coll == ncclFuncAllReduce) {
    info->algorithm = kNcclAlgoRing;
    info->protocol = kNcclProtoSimple;
    TF_ASSIGN_OR_RETURN(
        float time, ncclTopoGetAlgoTime(info, kNcclAlgoRing, kNcclProtoSimple,
                                        /*numPipeOps=*/1));
    minTime = time;
  } else {
    for (int p = 0; p < 3; p++) {
      TF_ASSIGN_OR_RETURN(float time,
                          ncclTopoGetAlgoTime(info, kNcclAlgoRing, p, 1));
      if (time > 0 && time < minTime) {
        info->algorithm = kNcclAlgoRing;
        info->protocol = p;
        minTime = time;
      }
    }
  }
  info->sleep_duration += ceil(minTime * 1000);
  return absl::OkStatus();
}

Status MockNcclAccumulateSleepTime(size_t count, ncclDataType_t datatype,
                                   MockNcclComm_t comm, cudaStream_t stream,
                                   MockNcclInfo* info) {
  info->count = count;
  info->datatype = datatype;
  info->num_channels = 1;

  TF_RETURN_IF_ERROR(MockNcclInfoSetDerived(info, comm->num_ranks));

  return getAlgoInfo(info, comm, /*numPipeOps=*/0);
}

Status MockNcclCommInitRank(MockNcclComm_t comm, int nranks, int nnodes,
                            int rank) {
  absl::InlinedVector<MockNcclTopoGraphConfig, 6> graphs =
      GetNcclTopoGraphConfig();

  comm->rank = rank;
  comm->num_nodes = nnodes;
  comm->num_ranks = nranks;
  comm->num_channels = 1;
  comm->collnet_support = false;
  comm->min_comp_cap = 9.0;
  comm->max_comp_cap = 9.0;

  return MockNcclTopoTuneModel(comm, comm->min_comp_cap, comm->max_comp_cap,
                               absl::MakeSpan(graphs));
}

StatusOr<std::unique_ptr<MockNcclComm>> InitializeMockNcclComm(
    const NcclExecuteParams& params,
    const std::vector<ReplicaGroup>& replica_groups,
    CollectiveOpGroupMode group_mode, int64_t op_id, int64_t stream_id,
    bool enable_clique_optimization) {
  TF_ASSIGN_OR_RETURN(GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());

  TF_ASSIGN_OR_RETURN(
      std::vector<GlobalDeviceId> participants,
      GetParticipatingDevices(global_device_id, *params.device_assn,
                              replica_groups, group_mode));

  std::vector<GlobalDeviceId> local_devices;
  if (params.gpu_global_device_ids) {
    local_devices.reserve(params.gpu_global_device_ids->size());
    for (const auto& entry : *params.gpu_global_device_ids) {
      local_devices.push_back(entry.second);
    }
  }
  size_t num_local_participants = GetNumLocalParticipants(
      participants, params.gpu_global_device_ids ? &local_devices : nullptr);

  se::gpu::ScopedActivateExecutorContext scoped_context(params.stream_executor);

  int nranks = participants.size();
  int nnodes = nranks / num_local_participants;

  auto comm = std::make_unique<MockNcclComm>();
  TF_RETURN_IF_ERROR(
      MockNcclCommInitRank(comm.get(), nranks, nnodes, /*rank=*/0));
  return comm;
}

Status RunMockNcclCollectives(std::vector<DeviceBufferPair>& buffers,
                              se::Stream& stream, MockNcclComm_t comm,
                              Thunk::Kind reduce_op) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing the mock nccl collective call from device ordinal: "
          << device_ordinal;

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);
  MockNcclInfo info;

  TF_ASSIGN_OR_RETURN(info.coll, ToNcclFunctionType(reduce_op));
  info.comm = comm;
  info.stream = gpu_stream;
  info.sleep_duration = 0;
  int64_t total_element_count = 0;
  ncclDataType_t previous_dtype = ncclNumTypes;
  for (size_t i = 0; i < buffers.size(); ++i) {
    DeviceBufferPair& buffer = buffers[i];
    PrimitiveType element_type = buffer.element_type;
    TF_ASSIGN_OR_RETURN(
        auto dtype_and_multiplier,
        ToNcclDataTypeAndCountMultiplier(element_type, reduce_op));
    ncclDataType_t dtype = dtype_and_multiplier.first;
    int64_t element_count = buffer.element_count * dtype_and_multiplier.second;
    if (reduce_op == Thunk::kNcclReduceScatter)
      element_count = element_count / comm->num_ranks;
    if (i == 0 || dtype == previous_dtype) {
      previous_dtype = dtype;
      total_element_count += element_count;
      continue;
    }

    TF_RETURN_IF_ERROR(MockNcclAccumulateSleepTime(
        total_element_count, previous_dtype, comm, gpu_stream, &info));
    TF_RETURN_IF_ERROR(LaunchSleepKernel(gpu_stream, &info));
    info.sleep_duration = 0;
    total_element_count = element_count;
    previous_dtype = dtype;
  }

  TF_RETURN_IF_ERROR(MockNcclAccumulateSleepTime(
      total_element_count, previous_dtype, comm, gpu_stream, &info));

  TF_RETURN_IF_ERROR(LaunchSleepKernel(gpu_stream, &info));
  VLOG(3) << "Done performing the mock nccl collective call for ordinal: "
          << device_ordinal;
  return absl::OkStatus();
}

Status RunMockNcclAllToAll(bool has_split_dimension,
                           std::vector<DeviceBufferPair>& buffers,
                           se::Stream& stream, MockNcclComm_t comm) {
  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);

  int num_participants = comm->num_ranks;

  MockNcclInfo info;
  info.comm = comm;
  info.stream = gpu_stream;

  // AllToAll can operate in two modes. Either it specifies a split dimension,
  // in which case inputs are split and outputs concatenated in that dimension
  // (here, we only support dimension 0), or it takes a list of inputs
  // and produces a tuple of outputs.
  if (has_split_dimension) {
    for (size_t i = 0; i < buffers.size(); ++i) {
      DeviceBufferPair& buffer = buffers[i];
      const uint8_t* send_buffer =
          static_cast<uint8_t*>(buffer.source_buffer.opaque());
      uint8_t* recv_buffer =
          static_cast<uint8_t*>(buffer.destination_buffer.opaque());

      TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                          ToNcclDataTypeAndCountMultiplier(
                              buffer.element_type, Thunk::kNcclAllToAll));
      ncclDataType_t dtype = dtype_and_multiplier.first;
      int64_t element_count =
          buffer.element_count * dtype_and_multiplier.second;

      TF_RET_CHECK(element_count % num_participants == 0)
          << "Buffer was not an exact multiple of the number of participants.";
      size_t chunk_elements = element_count / num_participants;
      size_t chunk_bytes = chunk_elements * ShapeUtil::ByteSizeOfPrimitiveType(
                                                buffer.element_type);

      for (int rank = 0; rank < num_participants; ++rank) {
        VLOG(3) << absl::StreamFormat(
            "Calling mock ncclSend(sendbuff=%p, count=%d, peer=%d "
            "comm=%p, stream=%p)",
            send_buffer + rank * chunk_bytes, chunk_elements, rank,
            static_cast<const void*>(comm), gpu_stream);
        info.coll = ncclFuncSend;
        TF_RETURN_IF_ERROR(MockNcclAccumulateSleepTime(
            chunk_elements, dtype, comm, gpu_stream, &info));

        VLOG(3) << absl::StreamFormat(
            "Calling mock ncclRecv(recvbuff=%p, count=%d, peer=%d "
            "comm=%p, stream=%p)",
            recv_buffer + rank * chunk_bytes, chunk_elements, rank,
            static_cast<const void*>(comm), gpu_stream);

        info.coll = ncclFuncRecv;
        TF_RETURN_IF_ERROR(MockNcclAccumulateSleepTime(
            chunk_elements, dtype, comm, gpu_stream, &info));
      }
    }
  } else {
    TF_RET_CHECK(buffers.size() == num_participants)
        << "Number of inputs didn't match the number of participants.";

    for (size_t i = 0; i < buffers.size(); ++i) {
      DeviceBufferPair& buffer = buffers[i];
      const uint8_t* send_buffer =
          static_cast<uint8_t*>(buffer.source_buffer.opaque());
      uint8_t* recv_buffer =
          static_cast<uint8_t*>(buffer.destination_buffer.opaque());

      TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                          ToNcclDataTypeAndCountMultiplier(
                              buffer.element_type, Thunk::kNcclAllToAll));
      ncclDataType_t dtype = dtype_and_multiplier.first;
      int64_t element_count =
          buffer.element_count * dtype_and_multiplier.second;

      VLOG(3) << absl::StreamFormat(
          "Calling mock ncclSend(sendbuff=%p, count=%d, peer=%d "
          "comm=%p, stream=%p)",
          send_buffer, element_count, i, static_cast<const void*>(comm),
          gpu_stream);

      info.coll = ncclFuncSend;
      TF_RETURN_IF_ERROR(MockNcclAccumulateSleepTime(element_count, dtype, comm,
                                                     gpu_stream, &info));

      VLOG(3) << absl::StreamFormat(
          "Calling mock ncclRecv(recvbuff=%p, count=%d, peer=%d "
          "comm=%p, stream=%p)",
          recv_buffer, element_count, i, static_cast<const void*>(comm),
          gpu_stream);

      info.coll = ncclFuncRecv;
      TF_RETURN_IF_ERROR(MockNcclAccumulateSleepTime(element_count, dtype, comm,
                                                     gpu_stream, &info));
    }
  }

  VLOG(3) << "Done performing mock all-to-all ";
  return OkStatus();
}

Status RunMockCollectivePermute(
    NcclP2PConfig::SourceTargetMapEntry source_target, DeviceBufferPair& buffer,
    se::Stream& stream, MockNcclComm_t comm, absl::string_view device_string,
    int64_t current_id) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing collective permute from device ordinal: "
          << device_ordinal << "current_id " << current_id;

  const std::optional<int64_t> source_id = source_target.source;
  const std::optional<int64_t> target_id = source_target.target;

  se::DeviceMemoryBase src_addr = buffer.source_buffer;
  se::DeviceMemoryBase dest_addr = buffer.destination_buffer;

  VLOG(3) << absl::StreamFormat("%s : id = %d, source_id = %d, target_id = %d",
                                device_string, current_id,
                                source_id.value_or(-1), target_id.value_or(-1));

  TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                      ToNcclDataTypeAndCountMultiplier(
                          buffer.element_type, Thunk::kNcclCollectivePermute));
  ncclDataType_t dtype = dtype_and_multiplier.first;
  int64_t element_count = buffer.element_count * dtype_and_multiplier.second;

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);
  MockNcclInfo info;
  info.comm = comm;
  info.stream = gpu_stream;

  // Send source buffer to target peer if needed.
  if (target_id) {
    info.coll = ncclFuncSend;
    VLOG(3) << absl::StreamFormat(
        "%s : Calling mock ncclSend(sendbuff=%p, count=%d, peer=%d "
        "comm=%p, stream=%p)",
        device_string, src_addr.opaque(), element_count, *target_id,
        static_cast<const void*>(comm), gpu_stream);
    TF_RETURN_IF_ERROR(MockNcclAccumulateSleepTime(element_count, dtype, comm,
                                                   gpu_stream, &info));
  }

  // Receive data from the source peer to the destination buffer.
  if (source_id) {
    info.coll = ncclFuncRecv;
    VLOG(3) << absl::StreamFormat(
        "%s : Calling mock ncclRecv(recvbuff=%p, count=%d, peer=%d comm=%p, "
        "stream=%p)",
        device_string, dest_addr.opaque(), element_count, *source_id,
        static_cast<const void*>(comm), gpu_stream);
    TF_RETURN_IF_ERROR(MockNcclAccumulateSleepTime(element_count, dtype, comm,
                                                   gpu_stream, &info));
  }

  TF_RETURN_IF_ERROR(LaunchSleepKernel(gpu_stream, &info));
  VLOG(3) << "Done performing the mock nccl collective call for ordinal: "
          << device_ordinal;

  if (!source_id) {
    // If there is no source peer, i.e. no one send us any data, zero out dest
    // buffer.
    VLOG(3) << absl::StreamFormat(
        "%s : mock collective-Permute: Issuing MemZero", device_string);
    stream.ThenMemZero(&dest_addr, dest_addr.size());
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
