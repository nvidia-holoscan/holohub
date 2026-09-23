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
#ifndef HOLOSCAN_INFERENCE_TRT_MULTIDEVICE_HPP
#define HOLOSCAN_INFERENCE_TRT_MULTIDEVICE_HPP

/* TensorRT Multi-Device (multi-GPU) support for the HoloInfer TRT backend.
 *
 * Runs one TensorRT engine sharded across N GPUs using TensorRT's Multi-Device
 * feature (NCCL DistCollective ops), so a single InferenceOp drives N GPUs.
 * Mirrors the validated DeepStream nvdsinfer / Triton tensorrt_backend MD work
 * (TRT-28040). Gated behind HOLOINFER_ENABLE_TRT_MULTI_DEVICE so a build without
 * NCCL / TensorRT>=11 is byte-for-byte identical on the single-GPU path.
 *
 * Requires TensorRT >= 11.0 (Multi-Device GA), NCCL, and >= 2 GPUs (SM80+ for
 * DistCollective). The engine is sharded offline (e.g. polygraphy multi-device
 * shard); the same plan is deserialized on every rank and the rank is
 * distinguished at runtime by the communicator.
 */

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

namespace holoscan {
namespace inference {

#ifdef HOLOINFER_ENABLE_TRT_MULTI_DEVICE

/**
 * Owns the extra GPUs (ranks 1..N-1) for a Multi-Device TrtInfer. Rank 0 reuses
 * the TrtInfer's existing engine/context/stream (running on device_ids[0]);
 * ranks 1..N-1 each deserialize the same plan on their GPU and get their own
 * engine/context/stream/IO buffers held here.
 */
class MultiDeviceTrt {
 public:
  ~MultiDeviceTrt();

  /**
   * @brief Create the in-process NCCL communicators and per-rank contexts and
   * attach the communicator to every rank (including rank 0) concurrently.
   * @param engine_bytes serialized (sharded) plan, deserialized on every rank
   * @param device_ids   physical GPU id per rank; device_ids[0] == rank 0's GPU
   * @param rank0_context the TrtInfer's existing context (rank 0)
   * @param logger        TRT logger for the per-rank runtimes
   * @return true on success
   */
  bool initialize(const std::vector<char>& engine_bytes, const std::vector<int>& device_ids,
                  nvinfer1::IExecutionContext* rank0_context, nvinfer1::ILogger& logger);

  /**
   * @brief Per-rank-engine variant: rank r deserializes per_rank_bytes[r] on its
   * GPU (for weight-tensor-parallel models where each rank holds a distinct
   * weight shard). per_rank_bytes[0] is rank 0 (already running in rank0_context).
   */
  bool initialize_per_rank(const std::vector<std::vector<char>>& per_rank_bytes,
                           const std::vector<int>& device_ids,
                           nvinfer1::IExecutionContext* rank0_context, nvinfer1::ILogger& logger);

  /**
   * @brief Replicate rank-0 inputs to every other rank and launch enqueueV3 on
   * those ranks concurrently (one thread per rank) so the in-engine collectives
   * can rendezvous with rank 0. Non-blocking: pair with wait().
   * @param rank0_context rank-0 context (read current input shapes from it)
   * @param inputs        input tensor name -> (rank-0 device ptr, byte size)
   */
  void enqueue_others(nvinfer1::IExecutionContext* rank0_context,
                      const std::map<std::string, std::pair<const void*, size_t>>& inputs);

  /** @brief Join the per-rank enqueue threads and sync their streams. */
  bool wait();

  int world_size() const { return world_size_; }

 private:
  void compute_io_bytes(nvinfer1::ICudaEngine& eng, nvinfer1::IExecutionContext* ctx,
                        std::map<std::string, size_t>& io_bytes, size_t& largest) const;

  std::vector<int> device_ids_;
  int world_size_ = 0;
  // NCCL communicators (void* keeps nccl.h out of this header), index == rank.
  std::vector<void*> comms_;
  // Per-rank resources for ranks 1..N-1 (index r-1).
  std::vector<std::unique_ptr<nvinfer1::IRuntime>> runtimes_;
  std::vector<std::unique_ptr<nvinfer1::ICudaEngine>> engines_;
  std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> contexts_;
  std::vector<cudaStream_t> streams_;
  std::vector<std::map<std::string, std::pair<void*, size_t>>> io_buffers_;  // name->(buf,bytes)
  std::vector<void*> stage_;          // pinned host bounce per non-zero rank
  std::vector<char> ok_;              // per-rank enqueue status
  std::vector<std::thread> threads_;  // per-rank enqueue threads (ranks 1..N-1)
};

#endif  // HOLOINFER_ENABLE_TRT_MULTI_DEVICE

}  // namespace inference
}  // namespace holoscan

#endif  // HOLOSCAN_INFERENCE_TRT_MULTIDEVICE_HPP
