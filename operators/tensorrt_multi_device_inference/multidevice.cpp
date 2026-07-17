/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "multidevice.hpp"

#ifdef HOLOINFER_ENABLE_TRT_MULTI_DEVICE

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include <nccl.h>

namespace holoscan {
namespace inference {

namespace {
inline ncclComm_t* as_comms(std::vector<void*>& v) {
  static_assert(sizeof(void*) == sizeof(ncclComm_t),
                "ncclComm_t must be pointer-sized to alias as void*");
  return reinterpret_cast<ncclComm_t*>(v.data());
}

size_t dtype_bytes(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kINT64:
      return 8;
    case nvinfer1::DataType::kBOOL:
      return 1;
    case nvinfer1::DataType::kUINT8:
      return 1;
    case nvinfer1::DataType::kBF16:
      return 2;
    default:
      return 4;
  }
}

int64_t volume(const nvinfer1::Dims& d) {
  int64_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) v *= (d.d[i] < 0 ? 1 : d.d[i]);
  return v;
}
}  // namespace

void MultiDeviceTrt::compute_io_bytes(nvinfer1::ICudaEngine& eng, nvinfer1::IExecutionContext* ctx,
                                      std::map<std::string, size_t>& io_bytes,
                                      size_t& largest) const {
  largest = 0;
  const bool has_profile = eng.getNbOptimizationProfiles() > 0;
  // For dynamic inputs, pin the max profile shape so output extents resolve; for
  // static engines the engine shapes are already concrete.
  for (int i = 0; i < eng.getNbIOTensors(); ++i) {
    const char* name = eng.getIOTensorName(i);
    if (eng.getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
      continue;
    nvinfer1::Dims s = eng.getTensorShape(name);
    bool dynamic = false;
    for (int k = 0; k < s.nbDims; ++k) {
      if (s.d[k] < 0)
        dynamic = true;
    }
    if (dynamic && has_profile) {
      ctx->setInputShape(name, eng.getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX));
    }
  }
  for (int i = 0; i < eng.getNbIOTensors(); ++i) {
    const char* name = eng.getIOTensorName(i);
    nvinfer1::Dims d = ctx->getTensorShape(name);
    size_t bytes = static_cast<size_t>(volume(d)) * dtype_bytes(eng.getTensorDataType(name));
    io_bytes[name] = bytes;
    largest = std::max(largest, bytes);
  }
}

bool MultiDeviceTrt::initialize(const std::vector<char>& engine_bytes,
                                const std::vector<int>& device_ids,
                                nvinfer1::IExecutionContext* rank0_context,
                                nvinfer1::ILogger& logger) {
  // Single sharded plan deserialized on every rank (CP / one-shot TP): rank is
  // distinguished at runtime by the communicator.
  std::vector<std::vector<char>> per_rank(device_ids.size(), engine_bytes);
  return initialize_per_rank(per_rank, device_ids, rank0_context, logger);
}

bool MultiDeviceTrt::initialize_per_rank(const std::vector<std::vector<char>>& per_rank_bytes,
                                         const std::vector<int>& device_ids,
                                         nvinfer1::IExecutionContext* rank0_context,
                                         nvinfer1::ILogger& logger) {
  device_ids_ = device_ids;
  world_size_ = static_cast<int>(device_ids.size());
  if (world_size_ < 2 || static_cast<int>(per_rank_bytes.size()) != world_size_)
    return false;
  const int rank0_dev = device_ids_[0];

  comms_.assign(world_size_, nullptr);
  ncclResult_t nr = ncclCommInitAll(as_comms(comms_), world_size_, device_ids_.data());
  if (nr != ncclSuccess) {
    fprintf(stderr, "[holoinfer-md] ncclCommInitAll: %s\n", ncclGetErrorString(nr));
    return false;
  }

  runtimes_.resize(world_size_ - 1);
  engines_.resize(world_size_ - 1);
  contexts_.resize(world_size_ - 1);
  streams_.assign(world_size_ - 1, nullptr);
  io_buffers_.resize(world_size_ - 1);
  stage_.assign(world_size_ - 1, nullptr);

  const bool want_peer = (std::getenv("HOLOINFER_MD_USE_PEER") != nullptr);

  for (int r = 1; r < world_size_; ++r) {
    const int dev = device_ids_[r];
    const int idx = r - 1;
    if (cudaSetDevice(dev) != cudaSuccess) {
      fprintf(stderr, "[holoinfer-md] cudaSetDevice(%d) failed\n", dev);
      return false;
    }
    runtimes_[idx].reset(nvinfer1::createInferRuntime(logger));
    engines_[idx].reset(
        runtimes_[idx]->deserializeCudaEngine(per_rank_bytes[r].data(), per_rank_bytes[r].size()));
    if (!engines_[idx]) {
      fprintf(stderr, "[holoinfer-md] rank %d deserialize failed\n", r);
      return false;
    }
    contexts_[idx].reset(engines_[idx]->createExecutionContext());
    if (!contexts_[idx])
      return false;
    if (cudaStreamCreate(&streams_[idx]) != cudaSuccess)
      return false;

    std::map<std::string, size_t> io_bytes;
    size_t largest = 0;
    compute_io_bytes(*engines_[idx], contexts_[idx].get(), io_bytes, largest);
    for (auto& kv : io_bytes) {
      void* buf = nullptr;
      if (kv.second && cudaMalloc(&buf, kv.second) != cudaSuccess) {
        fprintf(stderr, "[holoinfer-md] rank %d cudaMalloc(%zu) failed\n", r, kv.second);
        return false;
      }
      io_buffers_[idx][kv.first] = {buf, kv.second};
      contexts_[idx]->setTensorAddress(kv.first.c_str(), buf);
    }
    if (largest && cudaMallocHost(&stage_[idx], largest) != cudaSuccess)
      return false;

    // Peer-access detection (actual use opt-in via HOLOINFER_MD_USE_PEER).
    int a = 0, b = 0;
    cudaDeviceCanAccessPeer(&a, rank0_dev, dev);
    cudaDeviceCanAccessPeer(&b, dev, rank0_dev);
    if (want_peer && a && b) {
      cudaSetDevice(rank0_dev);
      cudaDeviceEnablePeerAccess(dev, 0);
      cudaSetDevice(dev);
      cudaDeviceEnablePeerAccess(rank0_dev, 0);
    }
  }

  // Attach the communicator to every rank CONCURRENTLY (cross-rank handshake;
  // a sequential loop deadlocks). Rank 0 reuses rank0_context.
  std::vector<std::thread> th;
  std::vector<char> rc(world_size_, 1);
  for (int r = 0; r < world_size_; ++r) {
    th.emplace_back([&, r]() {
      cudaSetDevice(device_ids_[r]);
      nvinfer1::IExecutionContext* ctx = (r == 0) ? rank0_context : contexts_[r - 1].get();
      if (!ctx->setCommunicator(comms_[r]))
        rc[r] = 0;
    });
  }
  for (auto& t : th) t.join();
  for (int r = 0; r < world_size_; ++r) {
    if (!rc[r]) {
      fprintf(stderr, "[holoinfer-md] setCommunicator failed on rank %d\n", r);
      return false;
    }
  }
  cudaSetDevice(rank0_dev);
  return true;
}

void MultiDeviceTrt::enqueue_others(
    nvinfer1::IExecutionContext* rank0_context,
    const std::map<std::string, std::pair<const void*, size_t>>& inputs) {
  const int rank0_dev = device_ids_[0];
  ok_.assign(world_size_, 1);
  threads_.clear();

  for (int r = 1; r < world_size_; ++r) {
    const int idx = r - 1;
    const int dev = device_ids_[r];
    // Replicate rank-0 inputs to this rank (pinned host bounce) and match shapes.
    for (const auto& in : inputs) {
      const std::string& name = in.first;
      const void* src = in.second.first;
      size_t bytes = in.second.second;
      auto it = io_buffers_[idx].find(name);
      if (it == io_buffers_[idx].end())
        continue;
      void* dst = it->second.first;
      contexts_[idx]->setInputShape(name.c_str(), rank0_context->getTensorShape(name.c_str()));
      cudaSetDevice(rank0_dev);
      cudaMemcpy(stage_[idx], src, bytes, cudaMemcpyDeviceToHost);
      cudaSetDevice(dev);
      cudaMemcpy(dst, stage_[idx], bytes, cudaMemcpyHostToDevice);
    }
  }
  cudaSetDevice(rank0_dev);

  for (int r = 1; r < world_size_; ++r) {
    threads_.emplace_back([this, r]() {
      const int idx = r - 1;
      cudaSetDevice(device_ids_[r]);
      if (!contexts_[idx]->enqueueV3(streams_[idx]))
        ok_[r] = 0;
    });
  }
}

bool MultiDeviceTrt::wait() {
  for (auto& t : threads_) t.join();
  threads_.clear();
  for (int r = 1; r < world_size_; ++r) {
    cudaSetDevice(device_ids_[r]);
    cudaStreamSynchronize(streams_[r - 1]);
  }
  cudaSetDevice(device_ids_[0]);
  bool ok = true;
  for (int r = 0; r < world_size_; ++r)
    if (!ok_[r])
      ok = false;
  return ok;
}

MultiDeviceTrt::~MultiDeviceTrt() {
  for (int r = 1; r < world_size_; ++r) {
    const int idx = r - 1;
    if (idx >= static_cast<int>(contexts_.size()))
      break;
    cudaSetDevice(device_ids_[r]);
    if (streams_[idx])
      cudaStreamSynchronize(streams_[idx]);
    contexts_[idx].reset();
    engines_[idx].reset();
    runtimes_[idx].reset();
    for (auto& kv : io_buffers_[idx])
      if (kv.second.first)
        cudaFree(kv.second.first);
    if (stage_[idx])
      cudaFreeHost(stage_[idx]);
    if (streams_[idx])
      cudaStreamDestroy(streams_[idx]);
  }
  for (void* c : comms_)
    if (c)
      ncclCommDestroy(reinterpret_cast<ncclComm_t>(c));
  if (!device_ids_.empty())
    cudaSetDevice(device_ids_[0]);
}

}  // namespace inference
}  // namespace holoscan

#endif  // HOLOINFER_ENABLE_TRT_MULTI_DEVICE
