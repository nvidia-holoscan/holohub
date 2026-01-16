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

#ifndef GST_SRC_OP_HPP
#define GST_SRC_OP_HPP

#include <holoscan/holoscan.hpp>
#include "gst_src_resource.hpp"

namespace holoscan {

/**
 * @brief Operator for bridging Holoscan data into GStreamer
 *
 * This operator receives Holoscan entities containing tensor data and pushes
 * them into a GStreamer pipeline via GstSrcResource. It converts the tensor
 * data into GStreamer buffers that can be consumed by GStreamer elements.
 */
class GstSrcOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GstSrcOp)

  GstSrcOp() = default;

  /**
   * @brief Setup the operator specification
   * 
   * Defines the operator's inputs and parameters:
   * - input: GXF Entity containing video frame tensor(s)
   * - gst_src_resource: GstSrcResource for pipeline communication
   */
  void setup(OperatorSpec& spec) override;

  /**
   * @brief Compute function that processes Holoscan entities
   * 
   * This function:
   * 1. Receives an entity from the input port
   * 2. Converts the entity to a GStreamer buffer
   * 3. Pushes the buffer into the GStreamer pipeline
   * 
   * @param input Input context for receiving entities
   * @param output Output context (unused - this is a sink operator)
   * @param context Execution context for GXF operations
   */
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

  /**
   * @brief Stop function called when operator execution ends
   * 
   * Sends EOS (End-Of-Stream) to the GStreamer pipeline to signal completion
   * and allow proper finalization of output files.
   */
  void stop() override;

 private:
  Parameter<std::shared_ptr<GstSrcResource>> gst_src_resource_;
};

}  // namespace holoscan

#endif /* GST_SRC_OP_HPP */

