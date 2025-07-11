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

#include <gxf/std/tensor.hpp>
#include <holoscan/holoscan.hpp>

#include "apriltag_detector.hpp"

namespace holoscan::ops {

/**
 * CUDA driver API error check helper
 */
#define CudaCheck(FUNC)                                                                     \
  {                                                                                         \
    const CUresult result = FUNC;                                                           \
    if (result != CUDA_SUCCESS) {                                                           \
      const char* error_name = "";                                                          \
      cuGetErrorName(result, &error_name);                                                  \
      const char* error_string = "";                                                        \
      cuGetErrorString(result, &error_string);                                              \
      std::stringstream buf;                                                                \
      buf << "[" << __FILE__ << ":" << __LINE__ << "] CUDA driver error " << result << " (" \
          << error_name << "): " << error_string;                                           \
      throw std::runtime_error(buf.str().c_str());                                          \
    }                                                                                       \
  }

void ApriltagDetectorOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("input");
  spec.output<std::vector<float2>>("output");

  spec.param(width_, "width", "Width", "Width of the input image");
  spec.param(height_, "height", "Height", "Height of the input image");
  spec.param(number_of_tags_, "number_of_tags", "Number of tags", "Number of tags to detect");

  cuda_stream_handler_.define_params(spec);
}

void ApriltagDetectorOp::start() {
  int status = -1;

  const float tag_dim = 0.0f;
  const uint32_t tile_size = 4;

  status = nvCreateAprilTagsDetector(
      &apriltag_handle_, width_, height_, tile_size, NVAT_TAG36H11, NULL, tag_dim);
  if (status != 0) {
    throw std::runtime_error("Failed to create the handle for AprilTag detector\n");
  }
  CudaCheck(cuInit(0));
  CUdevice cudaDevice;
  CudaCheck(cuDeviceGet(&cuda_device_, 0));
  CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
}

void ApriltagDetectorOp::stop() {
  int status = -1;

  status = cuAprilTagsDestroy(apriltag_handle_);
  if (status != 0) {
    throw std::runtime_error("Failed to destroy the handle for AprilTag detector\n");
  }

  CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
  cuda_context_ = nullptr;
}

void ApriltagDetectorOp::compute(holoscan::InputContext& input, holoscan::OutputContext& output,
                                 holoscan::ExecutionContext& context) {
  auto maybe_entity = input.receive<holoscan::gxf::Entity>("input");
  if (!maybe_entity) { throw std::runtime_error("Failed to receive input"); }

  auto& entity = maybe_entity.value();

  // get the CUDA stream from the input message
  gxf_result_t stream_handler_result = cuda_stream_handler_.from_message(context.context(), entity);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
  }

  const auto input_tensor = entity.get<holoscan::Tensor>();
  if (!input_tensor) { throw std::runtime_error("Tensor not found in message"); }

  DLDevice input_device = input_tensor->device();

  if (input_device.device_type != kDLCUDA) { throw std::runtime_error("The tensor is not device"); }

  if (input_tensor->ndim() != 3) { throw std::runtime_error("Tensor must be an RGB image"); }

  DLDataType dtype = input_tensor->dtype();
  if (dtype.code != kDLUInt || dtype.bits != 8 || dtype.lanes != 1) {
    throw std::runtime_error(
        fmt::format("Unexpected image data type '(code: {}, bits: {}, lanes: {})',"
                    "expected '(code: {}, bits: {}, lanes: {})'",
                    dtype.code,
                    dtype.bits,
                    dtype.lanes,
                    kDLUInt,
                    8,
                    1));
  }

  const auto input_shape = input_tensor->shape();
  const uint32_t height = input_shape[0];
  const uint32_t width = input_shape[1];
  const uint32_t components = input_shape[2];
  if (components != 3) {
    throw std::runtime_error(
        fmt::format("Unexpected component count {}, expected '3'", components));
  }
  std::vector<cuAprilTagsID_t> tags(number_of_tags_);
  uint32_t num_tags;
  cuAprilTagsImageInput_t input_image = {
      reinterpret_cast<uchar3*>(input_tensor->data()), width * sizeof(uchar3), width, height};

  cudaStream_t cuda_stream = cuda_stream_handler_.get_cuda_stream(context.context());
  cuAprilTagsDetect(
      apriltag_handle_, &input_image, tags.data(), &num_tags, number_of_tags_, cuda_stream);

  std::vector<output_corners> output_vec(number_of_tags_);
  if (num_tags == number_of_tags_) {
    for (uint8_t i = 0; i < number_of_tags_; i++) {
      output_vec[i].id = tags[i].id;
      output_vec[i].corners[0] = tags[i].corners[0];
      output_vec[i].corners[1] = tags[i].corners[1];
      output_vec[i].corners[2] = tags[i].corners[2];
      output_vec[i].corners[3] = tags[i].corners[3];
    }
  }
  output.emit(output_vec, "output");
}

}  // namespace holoscan::ops
