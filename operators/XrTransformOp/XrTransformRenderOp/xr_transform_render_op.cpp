/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "xr_transform_render_op.hpp"

#include "holoviz/holoviz.hpp"

namespace holoscan::openxr {

void XrTransformRenderOp::setup(OperatorSpec& spec) {
  spec.input<nvidia::gxf::Pose3D>("left_camera_pose").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::Pose3D>("right_camera_pose").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::CameraModel>("left_camera_model").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::CameraModel>("right_camera_model").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::Vector2f>("depth_range");
  spec.input<holoscan::gxf::Entity>("color_buffer_in");
  spec.input<holoscan::gxf::Entity>("depth_buffer_in");
  spec.input<UxBoundingBox>("ux_box");
  spec.input<UxCursor>("ux_cursor");

  spec.output<holoscan::gxf::Entity>("color_buffer_out");
  spec.output<holoscan::gxf::Entity>("depth_buffer_out");

  spec.param(display_width_, "display_width", "Display Width", "Display width in pixels", {});
  spec.param(display_height_, "display_height", "Display Height", "Display height in pixels", {});
}

void XrTransformRenderOp::start() {
  // Create a cuda stream for synchronizing CUDA events with the vulkan device queue.
  if (cudaStreamCreate(&cuda_stream_) != cudaSuccess) {
    throw std::runtime_error("cudaStreamCreate failed");
  }

  viz::SetCudaStream(cuda_stream_);

  // assume left and right displays vertically stacked
  viz::Init(display_width_.get(), display_height_.get() * 2, "Holoviz", viz::InitFlags::HEADLESS);
}

void XrTransformRenderOp::stop() {
  cudaStreamDestroy(cuda_stream_);
}

void XrTransformRenderOp::compute(InputContext& input, OutputContext& output,
                                  ExecutionContext& context) {
  auto color_message = input.receive<holoscan::gxf::Entity>("color_buffer_in").value();
  auto depth_message = input.receive<holoscan::gxf::Entity>("depth_buffer_in").value();
  auto depth_range = input.receive<nvidia::gxf::Vector2f>("depth_range");

  auto left_pose = input.receive<nvidia::gxf::Pose3D>("left_camera_pose");
  auto left_model = input.receive<nvidia::gxf::CameraModel>("left_camera_model");
  auto right_pose = input.receive<nvidia::gxf::Pose3D>("right_camera_pose");
  auto right_model = input.receive<nvidia::gxf::CameraModel>("right_camera_model");

  auto ux_box = input.receive<UxBoundingBox>("ux_box");
  auto ux_cursor = input.receive<UxCursor>("ux_cursor");

  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> color_buffer =
      static_cast<nvidia::gxf::Entity&>(color_message).get<nvidia::gxf::VideoBuffer>().value();
  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> depth_buffer =
      static_cast<nvidia::gxf::Entity&>(depth_message).get<nvidia::gxf::VideoBuffer>().value();

  auto color_cuda_events = color_message.findAll<nvidia::gxf::CudaEvent>().value();
  for (auto cuda_event : color_cuda_events) {
    if (cuda_event.has_value() && cuda_event.value()->event().has_value()) {
      if (cudaStreamWaitEvent(cuda_stream_, cuda_event.value()->event().value()) != cudaSuccess) {
        throw std::runtime_error("cudaStreamWaitEvent failed");
      }
    }
  }

  auto depth_cuda_events = depth_message.findAll<nvidia::gxf::CudaEvent>().value();
  for (auto cuda_event : depth_cuda_events) {
    if (cuda_event.has_value() && cuda_event.value()->event().has_value()) {
      if (cudaStreamWaitEvent(cuda_stream_, cuda_event.value()->event().value()) != cudaSuccess) {
        throw std::runtime_error("cudaStreamWaitEvent failed");
      }
    }
  }

  viz::Begin();

  viz::BeginImageLayer();

  viz::ImageCudaDevice(color_buffer->video_frame_info().width,
                       color_buffer->video_frame_info().height,
                       viz::ImageFormat::R8G8B8A8_UNORM,
                       reinterpret_cast<CUdeviceptr>(color_buffer->pointer()));

  viz::ImageCudaDevice(depth_buffer->video_frame_info().width,
                       depth_buffer->video_frame_info().height,
                       viz::ImageFormat::D32_SFLOAT,
                       reinterpret_cast<CUdeviceptr>(depth_buffer->pointer()));

  viz::EndLayer();

  viz::BeginGeometryLayer();

  std::array<float, 16> matrix;
  matrix = toModelView(*left_pose, *left_model, depth_range->x, depth_range->y);
  viz::LayerAddView(0.0, 0.0, 1.0, 0.5, matrix.data());
  matrix = toModelView(*right_pose, *right_model, depth_range->x, depth_range->y);
  viz::LayerAddView(0.0, 0.5, 1.0, 0.5, matrix.data());

  if (ux_box) { ui_box_renderer_.render(ux_box.value()); }

  viz::EndLayer();

  viz::End();

  viz::ReadFramebuffer(viz::ImageFormat::R8G8B8A8_UNORM,
                       color_buffer->video_frame_info().width,
                       color_buffer->video_frame_info().height,
                       color_buffer->size(),
                       reinterpret_cast<CUdeviceptr>(color_buffer->pointer()));

  nvidia::gxf::Handle<nvidia::gxf::CudaEvent> color_cuda_event =
      static_cast<nvidia::gxf::Entity&>(color_message).add<nvidia::gxf::CudaEvent>().value();
  color_cuda_event->init();
  if (cudaEventRecord(color_cuda_event->event().value(), cuda_stream_) != cudaSuccess) {
    throw std::runtime_error("cudaEventRecord failed");
  }

  output.emit(color_message, "color_buffer_out");

  viz::ReadFramebuffer(viz::ImageFormat::D32_SFLOAT,
                       depth_buffer->video_frame_info().width,
                       depth_buffer->video_frame_info().height,
                       depth_buffer->size(),
                       reinterpret_cast<CUdeviceptr>(depth_buffer->pointer()));

  nvidia::gxf::Handle<nvidia::gxf::CudaEvent> depth_cuda_event =
      static_cast<nvidia::gxf::Entity&>(depth_message).add<nvidia::gxf::CudaEvent>().value();
  depth_cuda_event->init();
  if (cudaEventRecord(depth_cuda_event->event().value(), cuda_stream_) != cudaSuccess) {
    throw std::runtime_error("cudaEventRecord failed");
  }

  output.emit(depth_message, "depth_buffer_out");
}

std::array<float, 16> XrTransformRenderOp::toModelView(nvidia::gxf::Pose3D pose,
                                                       nvidia::gxf::CameraModel camera,
                                                       float near_z, float far_z) {
  Eigen::Affine3f world;
  world.setIdentity();
  world.translation() = Eigen::Map<Eigen::Vector3f>(pose.translation.data());
  world.matrix().block(0, 0, 3, 3) =
      Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(pose.rotation.data());

  Eigen::Matrix4f projection = Eigen::Matrix4f::Zero();
  float x_range = 0.5 * camera.dimensions.x / camera.focal_length.x;
  float y_range = 0.5 * camera.dimensions.y / camera.focal_length.y;
  projection(0, 0) = 1 / x_range;
  projection(0, 2) = 0;
  projection(1, 1) = 1 / y_range;
  projection(1, 2) = 0;
  projection(2, 2) = -(far_z) / (far_z - near_z);
  projection(2, 3) = -(far_z * near_z) / (far_z - near_z);
  projection(3, 2) = -1;

  std::array<float, 16> matrix;
  Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(matrix.data()) =
      projection * world.matrix().inverse();

  return matrix;
}

}  // namespace holoscan::openxr
