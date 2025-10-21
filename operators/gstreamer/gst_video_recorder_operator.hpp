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

#ifndef GST_VIDEO_RECORDER_OPERATOR_HPP
#define GST_VIDEO_RECORDER_OPERATOR_HPP

#include <memory>
#include <string>
#include <future>
#include <holoscan/holoscan.hpp>
#include <gst/gst.h>
#include "gst_src_bridge.hpp"

namespace holoscan {

/**
 * @brief Operator for recording video streams to file using GStreamer
 *
 * This operator receives Holoscan entities containing video tensor data,
 * encodes them using GStreamer, and writes them to a file. It supports
 * various video formats and codecs through GStreamer's encoding pipeline.
 * 
 * The operator uses GstSrcBridge directly to bridge Holoscan data into a GStreamer
 * pipeline that handles encoding and muxing to produce output files.
 */
class GstVideoRecorderOperator : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GstVideoRecorderOperator)

  /**
   * @brief Setup the operator specification
   * 
   * Defines the operator's inputs and parameters:
   * - input: GXF Entity containing video frame tensor(s)
   * - encoder: Encoder base name (e.g., "nvh264", "nvh265", "x264", "x265") - default: "nvh264"
   *            Note: "enc" suffix is automatically appended to form the element name
   * - framerate: Video framerate as fraction or decimal - default: "30/1"
   *              Formats: "30/1", "30000/1001", "29.97", "60"
   *              Special: "0/1" for live mode (no framerate control, process frames as fast as they come)
   *              Note: In live mode, timestamps reflect actual frame arrival times (real-time)
   * - queue_limit: Maximum number of buffers to queue (0 = unlimited) - default: 10
   * - timeout_ms: Timeout in milliseconds for buffer push - default: 1000ms
   * - filename: Output video filename - default: "output.mp4"
   *              Note: If no extension is provided, ".mp4" is automatically appended
   * 
   * Note: Width, height, format, and storage type are automatically detected from the first frame
   * Note: Parser element is automatically determined from the encoder name
   * Note: Muxer element is automatically determined from the file extension:
   *       .mp4 -> mp4mux, .mkv -> matroskamux
   *       Unsupported extensions default to mp4mux
   */
  void setup(OperatorSpec& spec) override;

  /**
   * @brief Start function called after initialization but before first compute
   * 
   * Creates and starts the GStreamer pipeline, and begins bus monitoring.
   */
  void start() override;

  /**
   * @brief Compute function that processes video frames
   * 
   * This function:
   * 1. Receives a video frame entity from the input port
   * 2. Converts the entity to a GStreamer buffer
   * 3. Pushes the buffer into the encoding pipeline
   * 
   * @param input Input context for receiving video frame entities
   * @param output Output context (unused - this is a sink operator)
   * @param context Execution context for GXF operations
   */
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

  /**
   * @brief Stop function called when recording ends
   * 
   * Sends EOS (End-Of-Stream) to the GStreamer pipeline to signal completion,
   * waits for pipeline to finish processing, and cleans up resources.
   * This ensures proper finalization of the output file, including writing
   * headers/trailers for container formats.
   */
  void stop() override;

 private:

  // Parameters
  Parameter<std::string> encoder_name_;
  Parameter<std::string> framerate_;
  Parameter<size_t> queue_limit_;
  Parameter<uint64_t> timeout_ms_;
  Parameter<std::string> filename_;
  
  // Bridge and pipeline management
  std::shared_ptr<holoscan::gst::GstSrcBridge> bridge_;
  holoscan::gst::GstElementGuard pipeline_;
  holoscan::gst::GstElementGuard encoder_;  // Keep reference to link dynamically created converter to it
  bool bridge_initialized_{false};
  
  // Bus monitoring
  std::future<void> bus_monitor_future_;
};

}  // namespace holoscan

#endif /* GST_VIDEO_RECORDER_OPERATOR_HPP */

