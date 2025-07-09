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

#include "realsense_camera.hpp"

#include "cuda_runtime.h"
#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
#include "librealsense2/rs.hpp"

namespace holoscan::ops {

void RealsenseCameraOp::setup(OperatorSpec& spec) {
  spec.output<holoscan::gxf::Entity>("color_buffer");
  spec.output<holoscan::gxf::Entity>("depth_buffer");

  spec.output<nvidia::gxf::CameraModel>("color_camera_model").condition(ConditionType::kNone);
  spec.output<nvidia::gxf::CameraModel>("depth_camera_model").condition(ConditionType::kNone);

  spec.param(allocator_, "allocator", "Allocator", "Allocator to allocate output tensor.");
}

void RealsenseCameraOp::start() {
  rs2::config config;
  config.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGBA8, 30);
  config.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
  profile_ = pipeline_.start(config);
}

void RealsenseCameraOp::stop() {
  pipeline_.stop();
}

void RealsenseCameraOp::compute(InputContext& op_input, OutputContext& op_output,
                                ExecutionContext& context) {
  auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      fragment()->executor().context(), allocator_->gxf_cid());

  // Wait for the next set of camera frames.
  rs2::frameset frameset =
      pipeline_.wait_for_frames().apply_filter(align_).apply_filter(units_transform_);
  rs2::video_frame color_frame = frameset.get_color_frame();
  rs2::video_frame depth_frame =
      frameset.first(RS2_STREAM_DEPTH, RS2_FORMAT_DISTANCE).as<rs2::video_frame>();

  // Emit the color buffer.
  auto color_buffer_message = gxf::Entity::New(&context);
  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> color_buffer =
      static_cast<nvidia::gxf::Entity>(color_buffer_message)
          .add<nvidia::gxf::VideoBuffer>()
          .value();
  color_buffer->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
      color_frame.get_width(),
      color_frame.get_height(),
      nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
      nvidia::gxf::MemoryStorageType::kDevice,
      allocator.value());
  nvidia::gxf::VideoBufferInfo color_buffer_info = color_buffer->video_frame_info();
  assert(color_buffer_info.color_planes[0].stride == color_frame.get_stride_in_bytes());
  assert(color_buffer_info.color_planes[0].bytes_per_pixel == color_frame.get_bytes_per_pixel());
  cudaError_t cuda_error = cudaMemcpy(color_buffer->pointer(),
                                      color_frame.get_data(),
                                      color_frame.get_data_size(),
                                      cudaMemcpyHostToDevice);
  if (cuda_error != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy() failed for color_frame");
  }
  op_output.emit(color_buffer_message, "color_buffer");

  // Emit the color camera model.
  // TODO: Add distortion models.
  rs2_intrinsics color_intrinsics =
      color_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
  const nvidia::gxf::CameraModel color_camera_model{
      .dimensions = {static_cast<uint32_t>(color_intrinsics.width),
                     static_cast<uint32_t>(color_intrinsics.height)},
      .focal_length = {color_intrinsics.fx, color_intrinsics.fy},
      .principal_point = {color_intrinsics.ppx, color_intrinsics.ppy},
      .distortion_type = nvidia::gxf::DistortionType::Perspective,
  };
  op_output.emit(color_camera_model, "color_camera_model");

  // Emit the depth buffer.
  auto depth_buffer_message = gxf::Entity::New(&context);
  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> depth_buffer =
      static_cast<nvidia::gxf::Entity>(depth_buffer_message)
          .add<nvidia::gxf::VideoBuffer>()
          .value();
  depth_buffer->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32F>(
      depth_frame.get_width(),
      depth_frame.get_height(),
      nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
      nvidia::gxf::MemoryStorageType::kDevice,
      allocator.value());
  nvidia::gxf::VideoBufferInfo depth_buffer_info = depth_buffer->video_frame_info();
  assert(depth_buffer_info.color_planes[0].stride == depth_frame.get_stride_in_bytes());
  assert(depth_buffer_info.color_planes[0].bytes_per_pixel == depth_frame.get_bytes_per_pixel());
  cuda_error = cudaMemcpy(depth_buffer->pointer(),
                          depth_frame.get_data(),
                          depth_frame.get_data_size(),
                          cudaMemcpyHostToDevice);
  if (cuda_error != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy() failed for depth_frame");
  }
  op_output.emit(depth_buffer_message, "depth_buffer");

  // Emit the depth camera model.
  // TODO: Add distortion models.
  rs2_intrinsics depth_intrinsics =
      depth_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
  const nvidia::gxf::CameraModel depth_camera_model{
      .dimensions = {static_cast<uint32_t>(depth_intrinsics.width),
                     static_cast<uint32_t>(depth_intrinsics.height)},
      .focal_length = {depth_intrinsics.fx, depth_intrinsics.fy},
      .principal_point = {depth_intrinsics.ppx, depth_intrinsics.ppy},
      .distortion_type = nvidia::gxf::DistortionType::Perspective,
  };
  op_output.emit(depth_camera_model, "depth_camera_model");
}

}  // namespace holoscan::ops
