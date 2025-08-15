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


#include <memory>
#include <mutex>
#include <string>

#include <holoscan/holoscan.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/io_spec.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

#include "StreamingServer.h"

namespace holoscan::ops {

/**
 * @brief Operator that wraps the StreamingServer for video streaming in Holoscan
 * 
 * This operator provides integration with the StreamingServer library,
 * allowing Holoscan applications to receive and send video streams.
 */
class StreamingServerOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(StreamingServerOp)

  StreamingServerOp() = default;
  ~StreamingServerOp();

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void stop() override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override;

 private:
  // Configuration parameters
  Parameter<uint32_t> width_;           // Frame width
  Parameter<uint32_t> height_;          // Frame height
  Parameter<uint32_t> fps_;             // Frames per second
  Parameter<uint16_t> port_;            // Server port
  Parameter<bool> multi_instance_;       // Allow multiple server instances
  Parameter<std::string> server_name_;   // Server name identifier
  Parameter<bool> receive_frames_;       // Whether to receive frames
  Parameter<bool> send_frames_;          // Whether to send frames
  Parameter<std::shared_ptr<Allocator>> allocator_; // Memory allocator

  // Streaming server instance
  std::unique_ptr<StreamingServer> server_;
  
  // For handling received frames
  std::mutex frame_mutex_;
  Frame current_frame_;
  std::atomic<bool> has_new_frame_{false};
  
  // Shutdown flag for safer cleanup
  std::atomic<bool> is_shutting_down_{false};
  
  // Event callback handler
  void onEvent(const StreamingServer::Event& event);
};

}  // namespace holoscan::ops 