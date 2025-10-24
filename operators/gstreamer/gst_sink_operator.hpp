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

#ifndef GST_SINK_OPERATOR_HPP
#define GST_SINK_OPERATOR_HPP

#include <holoscan/holoscan.hpp>
#include "gst_sink_resource.hpp"

namespace holoscan {

/**
 * @brief Operator for bridging GStreamer data into Holoscan
 *
 * This operator receives buffers from a GStreamer pipeline via GstSinkResource
 * and outputs them as Holoscan entities containing tensor data. It supports
 * both packed formats (RGBA, RGB) and planar formats (I420, NV12).
 */
class GstSinkOperator : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GstSinkOperator)

  /**
   * @brief Setup the operator specification
   * 
   * Defines the operator's outputs and parameters:
   * - output: GXF Entity containing video frame tensor(s)
   * - gst_sink_resource: GstSinkResource for pipeline communication
   * - timeout_ms: Timeout in milliseconds for buffer wait (default: 1000ms)
   */
  void setup(OperatorSpec& spec) override;

  /**
   * @brief Compute function that processes GStreamer buffers
   * 
   * This function:
   * 1. Pops a buffer from the GStreamer pipeline (with timeout)
   * 2. Creates a GXF Entity with tensor(s) from the buffer
   * 3. Emits the entity to downstream operators
   * 
   * @param input Input context (unused - this is a source operator)
   * @param output Output context for emitting entities
   * @param context Execution context for GXF operations
   */
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
  Parameter<std::shared_ptr<GstSinkResource>> gst_sink_resource_;
  Parameter<uint64_t> timeout_ms_;
};

}  // namespace holoscan

#endif /* GST_SINK_OPERATOR_HPP */

