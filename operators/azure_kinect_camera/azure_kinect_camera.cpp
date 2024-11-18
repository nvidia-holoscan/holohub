/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "azure_kinect_camera.hpp"

#include "cuda_runtime.h"
#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"


namespace holoscan::ops {

void AzureKinectCameraOp::setup(OperatorSpec& spec) {
  spec.output<holoscan::gxf::Entity>("color_buffer");
  spec.output<holoscan::gxf::Entity>("depth_buffer");
  //spec.output<holoscan::gxf::Entity>("infrared_buffer");

  spec.output<nvidia::gxf::CameraModel>("color_camera_model").condition(ConditionType::kNone);
  spec.output<nvidia::gxf::CameraModel>("depth_camera_model").condition(ConditionType::kNone);
  spec.output<nvidia::gxf::Pose3D>("depth_to_color_transform").condition(ConditionType::kNone);

  spec.param(allocator_, "allocator", "Allocator", "Allocator to allocate output tensor.");
  spec.param(device_serial_, "device_serial", "Device Serial", "Serial number of the device or ANY");
  spec.param(capture_timeout_ms_, "capture_timeout_ms", "Capture Timeout (ms)", "Timeout to wait for an image to be captured.");

}

gxf_result_t AzureKinectCameraOp::open_device() {

  auto device_count = k4a::device::get_installed_count();
  if (device_count == 0) {
    holoscan::log_error("No Azure Kinect devices detected!");
    return GXF_FAILURE;
  }
  holoscan::log_info("Found {0} connected devices", device_count);

  bool success{true};
  for (int deviceIndex = 0; deviceIndex < device_count; deviceIndex++) {
    try {
      m_handle = k4a::device::open(deviceIndex);
      auto serialnr = m_handle.get_serialnum();
      if (device_serial_.get() == "ANY" || device_serial_.get() == serialnr) {
        holoscan::log_info(
            "Found Kinect4Azure "
            "Camera with serial {0} under index {1}",
            serialnr, deviceIndex);
        break;
      }

    } catch (k4a::error& e) {
      holoscan::log_error(
          "Error opening "
          "device: {0} -> {1}",
          deviceIndex, e.what());
      return GXF_FAILURE;
    }
  }

  // some defaults for now
  m_camera_config.camera_fps = K4A_FRAMES_PER_SECOND_30;
  m_camera_config.depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED;
  m_camera_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
  m_camera_config.color_resolution = K4A_COLOR_RESOLUTION_1536P;

  m_camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
  m_camera_config.synchronized_images_only = true;

  m_device_calibration = m_handle.get_calibration(m_camera_config.depth_mode,m_camera_config.color_resolution);

  return GXF_SUCCESS;
}

void AzureKinectCameraOp::start() {
  if (open_device() != GXF_SUCCESS) {
    return;
  }
  // set camera parameters here ..

  try {
    m_handle.start_cameras(&m_camera_config);
  } catch(const k4a::error& e) {
    holoscan::log_error("Error starting camera: {0}", e.what());
  }
}

void AzureKinectCameraOp::stop() {
  m_handle.stop_cameras();
  m_handle.close();
}

void AzureKinectCameraOp::compute(InputContext& op_input, OutputContext& op_output,
                                ExecutionContext& context) {
  auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      fragment()->executor().context(), allocator_->gxf_cid());

  // Wait for the next set of camera frames.
  if (m_handle.get_capture(&m_capture, std::chrono::milliseconds(capture_timeout_ms_.get()))) {
    if (!m_capture.is_valid()) {
      holoscan::log_warn("CaptureCamera received invalid capture");
      return;
    }

  } else {
    holoscan::log_warn("CaptureCamera timed out while grabbing frame");
    return;
  }

  auto color_frame = m_capture.get_color_image();
  auto depth_frame = m_capture.get_depth_image();

  // Emit the color buffer.
  auto color_buffer_message = gxf::Entity::New(&context);
  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> color_buffer =
      static_cast<nvidia::gxf::Entity>(color_buffer_message)
          .add<nvidia::gxf::VideoBuffer>()
          .value();
  color_buffer->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRA>(
      color_frame.get_width_pixels(),
      color_frame.get_height_pixels(),
      nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
      nvidia::gxf::MemoryStorageType::kDevice,
      allocator.value());
  nvidia::gxf::VideoBufferInfo color_buffer_info = color_buffer->video_frame_info();
  assert(color_buffer_info.color_planes[0].stride == color_frame.get_stride_bytes());
  assert(color_buffer_info.color_planes[0].bytes_per_pixel == 4); // will change with config ..
  cudaError_t cuda_error = cudaMemcpy(color_buffer->pointer(),
                                      color_frame.get_buffer(),
                                      color_frame.get_size(),
                                      cudaMemcpyHostToDevice);
  if (cuda_error != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy() failed for color_frame");
  }
  op_output.emit(color_buffer_message, "color_buffer");

  // Emit the color camera model.
  auto& color_intrinsics = m_device_calibration.color_camera_calibration;

  const nvidia::gxf::CameraModel color_camera_model{
      .dimensions = {static_cast<uint32_t>(color_intrinsics.resolution_width),
                     static_cast<uint32_t>(color_intrinsics.resolution_height)},
      .focal_length = {color_intrinsics.intrinsics.parameters.param.fx,
                       color_intrinsics.intrinsics.parameters.param.fy},
      .principal_point = {color_intrinsics.intrinsics.parameters.param.cx,
                          color_intrinsics.intrinsics.parameters.param.cy},
      .distortion_type = nvidia::gxf::DistortionType::Brown,
      .distortion_coefficients = {
          color_intrinsics.intrinsics.parameters.param.k1,
          color_intrinsics.intrinsics.parameters.param.k2,
          color_intrinsics.intrinsics.parameters.param.p1,
          color_intrinsics.intrinsics.parameters.param.p2,
          color_intrinsics.intrinsics.parameters.param.k3,
          color_intrinsics.intrinsics.parameters.param.k4,
          color_intrinsics.intrinsics.parameters.param.k5,
          color_intrinsics.intrinsics.parameters.param.k6,
      },
  };
  op_output.emit(color_camera_model, "color_camera_model");

  // Emit the depth buffer.
  auto depth_buffer_message = gxf::Entity::New(&context);
  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> depth_buffer =
      static_cast<nvidia::gxf::Entity>(depth_buffer_message)
          .add<nvidia::gxf::VideoBuffer>()
          .value();
  depth_buffer->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY16>(
      depth_frame.get_width_pixels(),
      depth_frame.get_height_pixels(),
      nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
      nvidia::gxf::MemoryStorageType::kDevice,
      allocator.value());
  nvidia::gxf::VideoBufferInfo depth_buffer_info = depth_buffer->video_frame_info();
  assert(depth_buffer_info.color_planes[0].stride == depth_frame.get_stride_bytes());
  assert(depth_buffer_info.color_planes[0].bytes_per_pixel == 2);
  cuda_error = cudaMemcpy(depth_buffer->pointer(),
                          depth_frame.get_buffer(),
                          depth_frame.get_size(),
                          cudaMemcpyHostToDevice);
  if (cuda_error != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy() failed for depth_frame");
  }
  op_output.emit(depth_buffer_message, "depth_buffer");

  // Emit the depth camera model.
  auto& depth_intrinsics = m_device_calibration.depth_camera_calibration;

  const nvidia::gxf::CameraModel depth_camera_model{
      .dimensions = {static_cast<uint32_t>(depth_intrinsics.resolution_width),
                     static_cast<uint32_t>(depth_intrinsics.resolution_height)},
      .focal_length = {depth_intrinsics.intrinsics.parameters.param.fx,
                       depth_intrinsics.intrinsics.parameters.param.fy},
      .principal_point = {depth_intrinsics.intrinsics.parameters.param.cx,
                          depth_intrinsics.intrinsics.parameters.param.cy},
      .distortion_type = nvidia::gxf::DistortionType::Brown,
      .distortion_coefficients = {
          depth_intrinsics.intrinsics.parameters.param.k1,
          depth_intrinsics.intrinsics.parameters.param.k2,
          depth_intrinsics.intrinsics.parameters.param.p1,
          depth_intrinsics.intrinsics.parameters.param.p2,
          depth_intrinsics.intrinsics.parameters.param.k3,
          depth_intrinsics.intrinsics.parameters.param.k4,
          depth_intrinsics.intrinsics.parameters.param.k5,
          depth_intrinsics.intrinsics.parameters.param.k6,
      },
  };
  op_output.emit(depth_camera_model, "depth_camera_model");
}

}  // namespace holoscan::ops
