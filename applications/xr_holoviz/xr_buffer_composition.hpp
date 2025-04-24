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
    Caveat: HolovizOp doesn't support read depth buffer now.
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
    // Get the render buffer output from HolovizOp
    auto color_render_buffer = op_input.receive<gxf::Entity>("color_render_buffer_output").value();
    auto xr_manager = xr_manager_.get();

    // Get the VideoBuffer from the entity
    auto video_buffer = holoscan::gxf::get_videobuffer(color_render_buffer);
    auto frame = video_buffer.get();
    // Access the video frame info
    const auto& buffer_info = frame->video_frame_info();
    int height = buffer_info.height;
    int width = buffer_info.width;
    int channels = 0;

    // Determine channels based on color format
    switch (buffer_info.color_format) {
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA:
        channels = 4;  // RGBA
        break;
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB:
        channels = 3;  // RGB
        break;
    }

    // Get the pointer to the tensor data
    void* data = frame->pointer();

    // Create an XR composition layer for the frame
    auto composition_layer = xr_manager->create_composition_layer();

    // Acquire swapchains
    holoscan::Tensor color_tensor = xr_manager->acquire_color_swapchain();
    holoscan::Tensor depth_tensor = xr_manager->acquire_depth_swapchain();

    // Copy the frame buffer to the swapchain
    cudaError_t err =
        cudaMemcpy(color_tensor.data(), data, color_tensor.nbytes(), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      std::cerr << "Render buffer to swapchain memcpy failed: " << cudaGetErrorString(err)
                << std::endl;
    }

    cudaStream_t cuda_stream = cudaStreamDefault;
    xr_manager->release_swapchains(cuda_stream);

    op_output.emit(std::static_pointer_cast<xr::CompositionLayerBaseHeader>(composition_layer),
                   "xr_composition_layer");
  }

 private:
  Parameter<std::shared_ptr<holoscan::XrManager>> xr_manager_;
};
}  // namespace holoscan::ops
