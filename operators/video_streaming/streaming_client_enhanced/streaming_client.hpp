/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file streaming_client.hpp
 * @brief StreamingClientOp for Holoscan SDK integration with Holoscan Streaming Stack
 *
 * This operator provides streaming client functionality for the Holoscan SDK, allowing
 * real-time video streaming to remote servers using the Holoscan Streaming Stack library.
 *
 * Key Features:
 * - Receives video frame tensors from Holoscan pipeline
 * - Converts BGR format to BGRA format for Holoscan Streaming Stack compatibility
 * - Sends frames to streaming server via WebRTC
 * - Frame validation and error handling
 * - Configurable streaming parameters
 *
 * The operator expects 3D tensors in [height, width, channels] format with BGR data
 * and automatically converts them to BGRA format before transmission.
 *
 * @author NVIDIA Corporation
 * @date 2024
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
#include "VideoFrame.h"


namespace holoscan::ops {

/**
 * @brief Operator that wraps the StreamingClient for video streaming in Holoscan
 *
 * This operator provides integration with the StreamingClient library,
 * allowing Holoscan applications to send and receive video streams.
 *
 * Features include:
 * - Frame validation to prevent sending empty frames during startup
 * - Configurable minimum non-zero byte threshold for frame validation
 * - This helps prevent server disconnections due to invalid frame data
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
  /**
   * @brief Parameter for minimum non-zero bytes required in frame data
   *
   * This parameter sets the minimum number of non-zero bytes that must be present
   * in the first 1000 bytes of frame data to consider the frame valid for transmission.
   * This helps filter out empty or nearly empty frames during application startup
   * or video source initialization.
   *
   * Default: 100 bytes
   * Range: 0-1000 bytes
   *
   * Lower values may allow transmission of frames with minimal data,
   * while higher values ensure more substantial frame content before transmission.
   */
  Parameter<uint32_t> min_non_zero_bytes_;
  Parameter<std::shared_ptr<Allocator>> allocator_;  ///< Memory allocator for output buffers

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

  // Tensor validation helper
  bool validateTensorData(const std::shared_ptr<holoscan::Tensor>& tensor);

  // Connection retry state
  int retry_count_ = 0;
  std::chrono::steady_clock::time_point last_retry_time_ = std::chrono::steady_clock::now();
  static constexpr int max_retries_ = 3;

  // Helper methods to reduce compute() function size
  bool handleConnectionRetry(int compute_call_count);
  bool validateAndPrepareTensor(const std::shared_ptr<holoscan::Tensor>& tensor,
                                 int& expected_width,
                                 int& expected_height);
  std::shared_ptr<std::vector<uint8_t>> convertBGRtoBGRA(
      const std::shared_ptr<holoscan::Tensor>& tensor,
      int expected_width,
      int expected_height);
  VideoFrame createVideoFrame(const std::shared_ptr<std::vector<uint8_t>>& bgra_buffer,
                                int expected_width,
                                int expected_height);
  void emitBGRATensorForVisualization(holoscan::OutputContext& op_output,
                                      holoscan::ExecutionContext& context,
                                      const std::shared_ptr<std::vector<uint8_t>>& bgra_buffer,
                                      int expected_width,
                                      int expected_height);
  bool sendFrameWithRetry(const VideoFrame& frame);
};

}  // namespace holoscan::ops
