/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef GST_VIDEO_RECORDER_OP_HPP
#define GST_VIDEO_RECORDER_OP_HPP

#include <atomic>
#include <future>
#include <map>
#include <memory>
#include <string>

#include <gst/gst.h>

#include <holoscan/holoscan.hpp>

#include "gst/element.hpp"
#include "gst/object.hpp"
#include "gst/pipeline.hpp"
#include "gst_src_bridge.hpp"
#include "gst_pipeline_bus_monitor.hpp"

namespace holoscan {

/**
 * @brief Operator for recording video streams to file using GStreamer
 *
 * This operator receives TensorMap containing video tensor data,
 * encodes them using GStreamer, and writes them to a file. It supports
 * various video formats and codecs through GStreamer's encoding pipeline.
 *
 * The operator uses GstSrcBridge directly to bridge Holoscan data into a GStreamer
 * pipeline that handles encoding and muxing to produce output files.
 */
class GstVideoRecorderOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GstVideoRecorderOp)

  /**
   * @brief Setup the operator specification
   *
   * Defines the operator's inputs and parameters:
   * - input: TensorMap containing video frame tensor(s)
   * - encoder: Encoder base name (e.g., "nvh264", "nvh265", "x264", "x265") - default: "nvh264"
   *            Note: "enc" suffix is automatically appended to form the element name
   * - format: Pixel format for video data (e.g., "RGBA", "RGB", "BGRA", "BGR", "GRAY8") - default:
   * "RGBA" Note: This format is used for GStreamer caps generation and tensor interpretation
   * - framerate: Video framerate as fraction or decimal - default: "30/1"
   *              Formats: "30/1", "30000/1001", "29.97", "60"
   *              Special: "0/1" for live mode (no framerate control, process frames as fast as they
   * come) Note: In live mode, timestamps reflect actual frame arrival times (real-time)
   * - max_buffers: Maximum number of buffers to queue (0 = unlimited) - default: 10
   * - block: Whether push_buffer() should block when the queue is full (true = block,
   *          false = non-blocking, may drop/timeout) - default: true
   * - filename: Output video filename - default: "output.mp4"
   *              Note: If no extension is provided, ".mp4" is automatically appended
   * - properties: Map of encoder-specific properties - default: empty map
   *              Examples: {"bitrate": "8000", "preset": "1", "gop-size": "30"}
   *              Property types are automatically detected and converted (int, uint, bool, float,
   * etc.)
   *
   * Note: Width, height, and storage type are automatically detected from the first frame
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
   * @throws std::runtime_error if pipeline or element creation/initialization fails
   */
  void start() override;

  /**
   * @brief Compute function that processes video frames
   *
   * This function:
   * 1. Receives a video frame TensorMap from the input port
   * 2. Converts the TensorMap to a GStreamer buffer
   * 3. Pushes the buffer into the encoding pipeline
   *
   * @param input Input context for receiving video frame TensorMap
   * @param output Output context (unused - this is a sink operator)
   * @param context Execution context for GXF operations
   * @throws std::runtime_error if bridge creation fails or tensor format is invalid
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
  Parameter<std::string> format_;
  Parameter<std::string> framerate_;  // Fraction string: "30/1", "30000/1001", "29.97",
                                      // or "0/1" for live mode (real-time timestamps)
  Parameter<size_t> max_buffers_;
  Parameter<bool> block_;
  Parameter<std::string> filename_;
  Parameter<std::map<std::string, std::string>> properties_;  // Encoder-specific properties
                                                              // (e.g., bitrate, preset)

  // Bridge and pipeline management
  std::shared_ptr<GstSrcBridge> src_bridge_;
  gst::Pipeline pipeline_;
  gst::Element encoder_;  // Keep reference to link dynamically created converter to it

  // Bus monitoring
  std::unique_ptr<gst::PipelineBusMonitor> bus_monitor_;

  // Frame tracking
  size_t frame_count_ = 0;
};

}  // namespace holoscan

#endif /* GST_VIDEO_RECORDER_OP_HPP */
