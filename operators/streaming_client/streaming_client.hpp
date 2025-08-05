/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <memory>
#include <mutex>
#include <string>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/io_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include <holoscan/core/execution_context.hpp>

#include "StreamingClient.h"  


namespace holoscan::ops {

/**
 * @brief Operator that wraps the StreamingClient for video streaming in Holoscan
 * 
 * This operator provides integration with the StreamingClient library,
 * allowing Holoscan applications to send and receive video streams.
 */
class StreamingClientOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(StreamingClientOp)

  StreamingClientOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& op_input,
              holoscan::OutputContext& op_output,
              holoscan::ExecutionContext& context) override;

 private:
  // Configuration parameters
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<uint32_t> fps_;
  Parameter<std::string> server_ip_;
  Parameter<uint16_t> signaling_port_;
  Parameter<bool> receive_frames_;
  Parameter<bool> send_frames_;

  // Streaming client
  std::unique_ptr<StreamingClient> client_;
  
  // For handling received frames
  std::mutex frame_mutex_;
  VideoFrame current_frame_;
  bool has_new_frame_ = false;
  
  // Timing control for frame rate management
  std::chrono::steady_clock::time_point last_frame_time_;
  std::chrono::microseconds frame_interval_;
  
  // Frame callback handler
  void onFrameReceived(const VideoFrame& frame);
  
  // Frame generation method
  VideoFrame generateFrame();
};

}  // namespace holoscan::ops
