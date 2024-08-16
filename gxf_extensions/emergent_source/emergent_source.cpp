/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string.h>
#include <utility>

#include "gxf/core/handle.hpp"
#include "gxf/multimedia/video.hpp"
#include "emergent_source.hpp"


namespace nvidia {
namespace holoscan {

EmergentSource::EmergentSource() {}

gxf_result_t EmergentSource::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(
    signal_, "signal", "Output", "Output channel");
  result &= registrar->parameter(
    width_, "width", "Width", "Width of the stream", kDefaultWidth);
  result &= registrar->parameter(
    height_, "height", "Height", "Height of the stream", kDefaultHeight);
  result &= registrar->parameter(
    framerate_, "framerate", "Framerate",
    "Framerate of the stream.", kDefaultFramerate);
  result &= registrar->parameter(
    use_rdma_, "rdma", "RDMA", "Enable RDMA.", kDefaultRDMA);
  result &= registrar->parameter(
    exposure_, "exposure", "Exposure",
    "Exposure time", kDefaultExposure);
  result &= registrar->parameter(
    gain_, "gain", "gain",
    "Analog Gain", kDefaultGain);
  return gxf::ToResultCode(result);
}

EVT_ERROR EmergentSource::OpenEVTCamera() {
  struct GigEVisionDeviceInfo deviceInfo[kMaxCameras];
  unsigned int count, camera_index;
  EVT_ERROR err = EVT_SUCCESS;

  // Find all cameras in system.
  unsigned int listcam_buf_size = kMaxCameras;
  EVT_ListDevices(deviceInfo, &listcam_buf_size, &count);
  if (count == 0) {
    GXF_LOG_ERROR("No cameras found.\n");
    return EVT_ERROR_ENODEV;
  }

  // Find and use first EVT camera.
  for (camera_index = 0; camera_index < kMaxCameras; camera_index++) {
    const char* EVT_models[] = { "HS", "HT", "HR", "HB", "LR", "LB", "HZ" };
    int EVT_models_count = sizeof(EVT_models) / sizeof(EVT_models[0]);
    bool is_EVT_camera = false;
    for (int i = 0; i < EVT_models_count; i++) {
      if (strncmp(deviceInfo[camera_index].modelName, EVT_models[i], 2) == 0) {
        is_EVT_camera = true;
        break;  // it is an EVT camera
      }
    }
    if (is_EVT_camera) {
      break;  // Use the first EVT camera.
    }
    if (camera_index == (kMaxCameras - 1)) {
      GXF_LOG_ERROR("No EVT cameras found.\n");
      return EVT_ERROR_ENODEV;
    }
  }

  // Open the camera with GPU 0 if use_rdma_ == true.
  if (use_rdma_) {
    camera_.gpuDirectDeviceId = 0;
  }
  err = EVT_CameraOpen(&camera_, &deviceInfo[camera_index]);
  if (err != EVT_SUCCESS) {
    GXF_LOG_ERROR("Error while opening camera.\n");
    return err;
  }
  return err;
}

EVT_ERROR EmergentSource::CheckCameraCapabilities() {
  EVT_ERROR err = EVT_SUCCESS;
  unsigned int height_max, width_max;
  unsigned int frame_rate_max, frame_rate_min;

  // Check resolution
  EVT_CameraGetUInt32ParamMax(&camera_, "Height", &height_max);
  EVT_CameraGetUInt32ParamMax(&camera_, "Width", &width_max);
  if ((width_ == 0U) || (width_ > width_max) ||
    (height_ == 0U) || (height_ > height_max)) {
    GXF_LOG_ERROR("Given resolution is not supported. Supported max"
      " resolution is (%u, %u)\n", width_max, height_max);
    return EVT_ERROR_INVAL;
  }
  EVT_CameraSetUInt32Param(&camera_, "Width", width_);
  EVT_CameraSetUInt32Param(&camera_, "Height", height_);

  // Check Framerate
  EVT_CameraGetUInt32ParamMax(&camera_, "FrameRate", &frame_rate_max);
  EVT_CameraGetUInt32ParamMin(&camera_, "FrameRate", &frame_rate_min);
  if ((framerate_ > frame_rate_max) || (framerate_ < frame_rate_min)) {
    GXF_LOG_ERROR("Given framerate is not supported. Supported framrate"
      " range is [%u, %u]\n", frame_rate_min, frame_rate_max);
    return EVT_ERROR_INVAL;
  }
  EVT_CameraSetUInt32Param(&camera_, "FrameRate", framerate_);

  EVT_CameraSetUInt32Param(&camera_, "Exposure", exposure_);
  EVT_CameraSetUInt32Param(&camera_, "Gain", gain_);

  return err;
}

void EmergentSource::SetDefaultConfiguration() {
  unsigned int width_max, height_max, param_val_max;
  unsigned long enum_buffer_size_return = 0;
  const unsigned long enum_buffer_size = 1000;
  const size_t kEnum_buffer_size = enum_buffer_size;
  char enum_buffer[kEnum_buffer_size];
  char* next_token;

  // Order is important as param max/mins get updated.
  EVT_CameraGetEnumParamRange(&camera_, "PixelFormat", enum_buffer,
    enum_buffer_size, &enum_buffer_size_return);
  char* enum_member = strtok_s(enum_buffer, ",", &next_token);
  EVT_CameraSetEnumParam(&camera_, "PixelFormat", enum_member);

  EVT_CameraSetUInt32Param(&camera_, "FrameRate", 30);

  EVT_CameraSetUInt32Param(&camera_, "OffsetX", 0);
  EVT_CameraSetUInt32Param(&camera_, "OffsetY", 0);

  EVT_CameraGetUInt32ParamMax(&camera_, "Width", &width_max);
  EVT_CameraSetUInt32Param(&camera_, "Width", width_max);

  EVT_CameraGetUInt32ParamMax(&camera_, "Height", &height_max);
  EVT_CameraSetUInt32Param(&camera_, "Height", height_max);

  EVT_CameraSetEnumParam(&camera_, "AcquisitionMode", "Continuous");
  EVT_CameraSetUInt32Param(&camera_, "AcquisitionFrameCount", 1);
  EVT_CameraSetEnumParam(&camera_, "TriggerSelector", "AcquisitionStart");
  EVT_CameraSetEnumParam(&camera_, "TriggerMode", "Off");
  EVT_CameraSetEnumParam(&camera_, "TriggerSource", "Software");
  EVT_CameraSetEnumParam(&camera_, "BufferMode", "Off");
  EVT_CameraSetUInt32Param(&camera_, "BufferNum", 0);

  EVT_CameraGetUInt32ParamMax(&camera_, "GevSCPSPacketSize", &param_val_max);
  EVT_CameraSetUInt32Param(&camera_, "GevSCPSPacketSize", param_val_max);

  EVT_CameraSetUInt32Param(&camera_, "Exposure", 3072);

  EVT_CameraSetUInt32Param(&camera_, "Gain", 4095);

  // Optical black correction is set to 10
  EVT_CameraSetUInt32Param(&camera_, "Offset", 10);

  EVT_CameraSetBoolParam(&camera_, "LUTEnable", false);
  EVT_CameraSetBoolParam(&camera_, "AutoGain", false);

  EVT_CameraSetUInt32Param(&camera_, "WB_R_GAIN_Value", 256);
  EVT_CameraSetUInt32Param(&camera_, "WB_GR_GAIN_Value", 166);
  EVT_CameraSetUInt32Param(&camera_, "WB_GB_GAIN_Value", 162);
  EVT_CameraSetUInt32Param(&camera_, "WB_B_GAIN_Value", 272);
}

gxf_result_t EmergentSource::start() {
  GXF_LOG_INFO("Emergent Source: RDMA is %s", use_rdma_ ? "enabled" : "disabled");

  EVT_ERROR err = EVT_SUCCESS;

  // Open EVT camera in system.
  if (OpenEVTCamera() != EVT_SUCCESS) {
    GXF_LOG_ERROR("No EVT camera found.\n");
    return GXF_FAILURE;
  }

  SetDefaultConfiguration();

  // Check Camera Capabilities
  if (CheckCameraCapabilities() != EVT_SUCCESS) {
    GXF_LOG_ERROR("EVT Camera does not support requested format.\n");
    return GXF_FAILURE;
  }

  // Prepare for streaming.
  err = EVT_CameraOpenStream(&camera_);
  if (err != EVT_SUCCESS) {
    GXF_LOG_ERROR("EVT_CameraOpenStream failed. Error: %d\n", err);
    EVT_CameraClose(&camera_);
    GXF_LOG_ERROR("Camera Closed\n");
    return GXF_FAILURE;
  }

  // Allocate buffers
  for (unsigned int frame_count = 0U; frame_count < FRAMES_BUFFERS; frame_count++) {
    evt_frame_[frame_count].size_x = width_;
    evt_frame_[frame_count].size_y = height_;
    // TODO: Add the option for providing different pixel_type
    evt_frame_[frame_count].pixel_type = kDefaultPixelFormat;

    err = EVT_AllocateFrameBuffer(&camera_, &evt_frame_[frame_count], EVT_FRAME_BUFFER_ZERO_COPY);
    if (err != EVT_SUCCESS) {
      GXF_LOG_ERROR("EVT_AllocateFrameBuffer Error!\n");
      return GXF_FAILURE;
    }

    err = EVT_CameraQueueFrame(&camera_, &evt_frame_[frame_count]);
    if (err != EVT_SUCCESS) {
      GXF_LOG_ERROR("EVT_CameraQueueFrame Error!\n");
      return GXF_FAILURE;
    }
  }

  // Start streaming
  err = EVT_CameraExecuteCommand(&camera_, "AcquisitionStart");
  if (err != EVT_SUCCESS) {
    GXF_LOG_ERROR("Acquisition start failed. Error %d\n", err);
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t EmergentSource::tick() {
  EVT_ERROR err = EVT_SUCCESS;

  err = EVT_CameraGetFrame(&camera_, &evt_frame_recv_, EVT_INFINITE);
  if (err != EVT_SUCCESS) {
    GXF_LOG_ERROR("Failed to get frame. Error %d\n", err);
    return GXF_FAILURE;
  }

  auto message = gxf::Entity::New(context());
  if (!message) {
    GXF_LOG_ERROR("Failed to allocate message.\n");
    return GXF_FAILURE;
  }

  auto buffer = message.value().add<gxf::VideoBuffer>();
  if (!buffer) {
    GXF_LOG_ERROR("Failed to allocate video buffer.\nr");
    return GXF_FAILURE;
  }

  gxf::VideoTypeTraits<gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY> video_type;
  gxf::VideoFormatSize<gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY> color_format;
  auto color_planes = color_format.getDefaultColorPlanes(width_, height_);
  gxf::VideoBufferInfo info{width_, height_, video_type.value, color_planes,
                            gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
  auto storage_type = use_rdma_ ? gxf::MemoryStorageType::kDevice : gxf::MemoryStorageType::kHost;
  buffer.value()->wrapMemory(info, evt_frame_recv_.bufferSize, storage_type,
                                         evt_frame_recv_.imagePtr, nullptr);

  signal_->publish(std::move(message.value()));

  err = EVT_CameraQueueFrame(&camera_, &evt_frame_recv_);  // Re-queue.
  if (err != EVT_SUCCESS) {
    GXF_LOG_ERROR("Failed to queue the frame.\n");
    return GXF_FAILURE;
  }

  return gxf::ToResultCode(message);
}

gxf_result_t EmergentSource::stop() {
  EVT_ERROR err = EVT_SUCCESS;

  // Tell camera to stop streaming
  err = EVT_CameraExecuteCommand(&camera_, "AcquisitionStop");
  if (err != EVT_SUCCESS) {
    GXF_LOG_ERROR("EVT_CameraExecuteCommand failed. Error: %d\n", err);
    return GXF_FAILURE;
  }

  // Release frame buffers
  for (unsigned int frame_count = 0U; frame_count < kNumBuffers; frame_count++) {
    if (EVT_ReleaseFrameBuffer(&camera_, &evt_frame_[frame_count]) != EVT_SUCCESS) {
      GXF_LOG_ERROR("Failed to release buffers.\n");
      return GXF_FAILURE;
    }
  }

  // Host side tear down for stream.
  if (EVT_CameraCloseStream(&camera_) != EVT_SUCCESS) {
    GXF_LOG_ERROR("Failed to close camera successfully.\n");
    return GXF_FAILURE;
  }

  if (EVT_CameraClose(&camera_) != EVT_SUCCESS) {
    GXF_LOG_ERROR("Failed to close camera successfully.\n");
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

}  // namespace holoscan
}  // namespace nvidia
