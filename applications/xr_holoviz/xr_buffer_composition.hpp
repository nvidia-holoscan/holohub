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
#include "holoscan/holoscan.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"
#include "xr_begin_frame_op.hpp"
#include "xr_composition_layers.hpp"
#include "xr_end_frame_op.hpp"
#include "xr_session.hpp"
#include "xr_swapchain_cuda.hpp"
#include "holoviz/holoviz.hpp"
#include "openxr/openxr.hpp"

#include "holoscan/operators/holoviz/holoviz.hpp"
#include "xr_manager.hpp"
namespace holoscan::ops {

/*
    This operator is used to compose the video buffer into the XR composition layer.
*/
class XrBufferCompositionOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(XrBufferCompositionOp)

  XrBufferCompositionOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<nvidia::gxf::VideoBuffer>("color_render_buffer_output");
    spec.output<std::shared_ptr<xr::CompositionLayerBaseHeader>>("xr_composition_layer");
    spec.param(xr_manager_, "xr_manager");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // Get the render buffer output from HolovizOp, but not used because it is already written to
    // the swapchain
    auto color_render_buffer = op_input.receive<gxf::Entity>("color_render_buffer_output").value();

    // Create an XR composition layer for the frame
    auto composition_layer = xr_manager_->create_composition_layer();

    // Release the swapchains
    cudaStream_t cuda_stream = cudaStreamDefault;
    xr_manager_->release_swapchains(cuda_stream);

    op_output.emit(std::static_pointer_cast<xr::CompositionLayerBaseHeader>(composition_layer),
                   "xr_composition_layer");
  }

 private:
  Parameter<std::shared_ptr<holoscan::XrManager>> xr_manager_;
};
}  // namespace holoscan::ops
