/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include "xr_render_rgbd_op.hpp"

#include <cuda_runtime.h>

#include <memory>

#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "xr_composition_layers.hpp"
#include "openxr/openxr.hpp"

namespace holoscan::ops {

void XrRenderRgbdOp::setup(OperatorSpec& spec) {
  spec.input<xr::FrameState>("xr_frame_state");

  // Run this operator regardless of whether a new camera image is available,
  // and always use the latest camera image.
  spec.input<holoscan::TensorMap>("camera_frame")
      .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer,
                 Arg("capacity", static_cast<uint64_t>(1)),
                 Arg("policy", static_cast<uint64_t>(0)))
      .condition(ConditionType::kNone);

  spec.output<std::shared_ptr<xr::CompositionLayerBaseHeader>>("xr_composition_layer");

  spec.param(xr_session_, "xr_session", "OpenXR Session", "OpenXR Session");

  cuda_stream_handler_.define_params(spec);
}

void XrRenderRgbdOp::start() {
  std::shared_ptr<holoscan::XrSession> xr_session = xr_session_.get();

  uint32_t width = xr_session->view_configurations()[0].recommendedImageRectWidth *
                   xr_session->view_configurations().size();
  uint32_t height = xr_session->view_configurations()[0].recommendedImageRectHeight;

  holoviz_instance_ = viz::Create();
  viz::SetCurrent(holoviz_instance_);
  viz::Init(width, height, "Holoviz", viz::InitFlags::HEADLESS);

  color_swapchain_ = std::make_unique<XrSwapchainCuda>(
      *xr_session, XrSwapchainCuda::Format::R8G8B8A8_UNORM, width, height);
  depth_swapchain_ = std::make_unique<XrSwapchainCuda>(
      *xr_session, XrSwapchainCuda::Format::D32_SFLOAT, width, height);
}

void XrRenderRgbdOp::stop() {
  viz::Shutdown(holoviz_instance_);
  holoviz_instance_ = nullptr;
}

void XrRenderRgbdOp::compute(InputContext& input, OutputContext& output,
                             ExecutionContext& context) {
  std::shared_ptr<holoscan::XrSession> xr_session = xr_session_.get();

  auto frame_state = input.receive<xr::FrameState>("xr_frame_state");

  // Update the camera image if a new image is available.
  if (!input.empty("camera_frame")) {
    camera_frame_ = input.receive<holoscan::TensorMap>("camera_frame").value();
  }

  // Return early if no camera image has been received yet.
  if (!camera_frame_.has_value()) {
    output.emit(std::shared_ptr<xr::CompositionLayerBaseHeader>(nullptr), "xr_composition_layer");
    return;
  }

  std::shared_ptr<holoscan::Tensor> rgba_tensor = camera_frame_.value()["rgba"];

  auto composition_layer = XrCompositionLayerProjectionStorage::create_for_frame(
      *frame_state, *xr_session, *color_swapchain_, *depth_swapchain_);

  holoscan::Tensor color_swapchain_tensor = color_swapchain_->acquire();
  holoscan::Tensor depth_swapchain_tensor = depth_swapchain_->acquire();

  cudaStream_t cuda_stream = cuda_stream_handler_.get_cuda_stream(context.context());

  viz::SetCurrent(holoviz_instance_);
  viz::SetCudaStream(cuda_stream);
  viz::Begin();
  viz::BeginImageLayer();

  // Render the image.
  // TODO: Visualize as a depth map instead of a 2D image.
  viz::ImageCudaDevice(rgba_tensor->shape()[1],
                       rgba_tensor->shape()[0],
                       viz::ImageFormat::R8G8B8A8_UNORM,
                       reinterpret_cast<CUdeviceptr>(rgba_tensor->data()));

  // Add the views from the composition layer.
  for (int i = 0; i < composition_layer->viewCount; i++) {
    xr::CompositionLayerProjectionView& view = composition_layer->views[i];

    // Calculate the model-view-projection matrix for each view.

    // The model matrix transforms the normalized camera image coordinates [0,1]
    // into world space. It is y-flipped and y-scaled by the inverse camera aspect
    // ratio in order to preserve aspect in the world.
    const glm::mat4 model_matrix = [rgba_tensor](const auto& pose_tensor) {
      glm::mat4 ret_val{1};
      constexpr float flip_y = -1.f;
      const float inverse_camera_aspect_ratio =
          static_cast<float>(rgba_tensor->shape()[0]) / static_cast<float>(rgba_tensor->shape()[1]);

      constexpr auto check_pose_shape = [](const auto& shape) {
        if (shape.size() != 2 || shape[0] != 4 || shape[1] != 4) {
          HOLOSCAN_LOG_WARN("Ignoring camera pose (unexpected shape)");
          return false;
        } else {
          return true;
        }
      };

      if (pose_tensor && check_pose_shape(pose_tensor->shape())) {
        const float* data = static_cast<const float*>(pose_tensor->data());
        ret_val = glm::make_mat4(data);
        ret_val = glm::transpose(ret_val);
      } else {
        constexpr float distance_to_image_plane = 3.f;
        ret_val = glm::translate(glm::mat4{1}, glm::vec3(0.f, 0.f, -distance_to_image_plane));
      }

      ret_val = glm::scale(ret_val, glm::vec3(1.f, flip_y * inverse_camera_aspect_ratio, 1.f));

      return ret_val;
    }(camera_frame_.value()["camera_transform"]);

    // The view matrix transforms points from world space to view space.
    glm::mat4 view_orientation = glm::mat4_cast(glm::make_quat(&view.pose.orientation.x));
    glm::mat4 view_translation =
        glm::translate(glm::mat4{1}, glm::make_vec3(&view.pose.position.x));
    glm::mat4 view_matrix = glm::inverse(view_translation * view_orientation);

    // The projection matrices transform points from view space into clip space.
    // Holoviz uses zero-to-one projection matrix. OpenXR uses right-handed
    // coordinate system.
    glm::mat4 projection_matrix =
        glm::frustumRH_ZO(composition_layer->depth_info[i].nearZ * glm::tan(view.fov.angleLeft),
                          composition_layer->depth_info[i].nearZ * glm::tan(view.fov.angleRight),
                          composition_layer->depth_info[i].nearZ * glm::tan(view.fov.angleUp),
                          composition_layer->depth_info[i].nearZ * glm::tan(view.fov.angleDown),
                          composition_layer->depth_info[i].nearZ,
                          composition_layer->depth_info[i].farZ);

    glm::mat4 view_projection_matrix_row_major =
        glm::transpose(projection_matrix * view_matrix * model_matrix);

    viz::LayerAddView(
        static_cast<float>(view.subImage.imageRect.offset.x) / color_swapchain_->width(),
        static_cast<float>(view.subImage.imageRect.offset.y) / color_swapchain_->height(),
        static_cast<float>(view.subImage.imageRect.extent.width) / color_swapchain_->width(),
        static_cast<float>(view.subImage.imageRect.extent.height) / color_swapchain_->height(),
        glm::value_ptr(view_projection_matrix_row_major));
  }

  viz::EndLayer();
  viz::End();

  // Read the Holoviz framebuffer into the color and depth swapchain images.
  viz::ReadFramebuffer(viz::ImageFormat::R8G8B8A8_UNORM,
                       color_swapchain_->width(),
                       color_swapchain_->height(),
                       color_swapchain_tensor.nbytes(),
                       reinterpret_cast<CUdeviceptr>(color_swapchain_tensor.data()));
  viz::ReadFramebuffer(viz::ImageFormat::D32_SFLOAT,
                       depth_swapchain_->width(),
                       depth_swapchain_->height(),
                       depth_swapchain_tensor.nbytes(),
                       reinterpret_cast<CUdeviceptr>(depth_swapchain_tensor.data()));

  // Release the OpenXR swapchain images.
  color_swapchain_->release(cuda_stream);
  depth_swapchain_->release(cuda_stream);

  // Emit the composition layer.
  output.emit(std::static_pointer_cast<xr::CompositionLayerBaseHeader>(composition_layer),
              "xr_composition_layer");
}

}  // namespace holoscan::ops
