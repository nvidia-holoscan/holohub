/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <holoscan/holoscan.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

#include "streaming_server_resource.hpp"
#include <holoscan/core/domain/tensor.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <unordered_set>

namespace holoscan::ops {

/**
 * @brief Operator that handles upstream (receiving) video streaming from clients
 *
 * This operator receives frames from streaming clients and emits holoscan::Tensor
 * objects to the Holoscan pipeline. It uses StreamingServerResource to manage
 * the server connection and frame reception.
 */
class StreamingServerUpstreamOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(StreamingServerUpstreamOp)

  StreamingServerUpstreamOp() = default;
  ~StreamingServerUpstreamOp();

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void stop() override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override;

 private:
  // Configuration parameters (can override StreamingServerResource defaults)
  Parameter<uint32_t> width_;           ///< Frame width
  Parameter<uint32_t> height_;          ///< Frame height
  Parameter<uint32_t> fps_;             ///< Frames per second
  Parameter<std::shared_ptr<Allocator>> allocator_; ///< Memory allocator

  // StreamingServerResource reference
  Parameter<std::shared_ptr<StreamingServerResource>> streaming_server_resource_;

  // Internal state
  std::atomic<bool> is_shutting_down_{false};
  std::atomic<bool> upstream_connected_{false};

  // Performance tracking
  std::atomic<uint64_t> frames_received_{0};
  std::atomic<std::chrono::steady_clock::time_point::rep> start_time_ticks_{0};

  // Duplicate detection and frame tracking
  std::atomic<uint64_t> last_processed_timestamp_{0};
  std::atomic<uint64_t> duplicate_frames_detected_{0};
  std::atomic<uint64_t> unique_frames_processed_{0};
  std::unordered_set<uint64_t> processed_frame_timestamps_;
  std::mutex frame_tracking_mutex_;

  // Internal event and frame handlers
  void on_streaming_server_event(const StreamingServerResource::Event& event);

  // Helper to convert Frame to holoscan::Tensor
  holoscan::Tensor convert_frame_to_tensor(const Frame& frame);

  // Duplicate detection helper
  bool is_duplicate_frame(const Frame& frame);
};

} // namespace holoscan::ops
