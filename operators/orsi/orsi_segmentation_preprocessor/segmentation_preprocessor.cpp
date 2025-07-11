/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "segmentation_preprocessor.hpp"

#include <limits>
#include <string>
#include <utility>

#include <npp.h>

#include "gxf/std/tensor.hpp"

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"


// TODO: consider to add this as utility macro in Holoscan SDK
#define CUDA_TRY(stmt)                                                                     \
  ({                                                                                       \
    cudaError_t _holoscan_cuda_err = stmt;                                                 \
    if (cudaSuccess != _holoscan_cuda_err) {                                               \
      HOLOSCAN_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", \
                    #stmt,                                                                 \
                    __LINE__,                                                              \
                    __FILE__,                                                              \
                    cudaGetErrorString(_holoscan_cuda_err),                                \
                    static_cast<int>(_holoscan_cuda_err));                                 \
    }                                                                                      \
    _holoscan_cuda_err;                                                                    \
  })

using holoscan::ops::orsi::segmentation_preprocessor::cuda_preprocess;
using holoscan::ops::orsi::segmentation_preprocessor::DataFormat;
using holoscan::ops::orsi::segmentation_preprocessor::output_type_t;
using holoscan::ops::orsi::segmentation_preprocessor::Shape;

namespace holoscan::ops::orsi {

// Utility sigmoid function for on host compute
double sigmoid(double a) {
      return 1.0 / (1.0 + exp(-a));
}

// Utility function to extract named tensor from message
// TODO: consider to add this as utility function in Holoscan SDK
std::shared_ptr<holoscan::Tensor> getTensorByName(holoscan::gxf::Entity in_message,
                                                  const std::string& in_tensor_name) {
  auto maybe_tensor = in_message.get<Tensor>(in_tensor_name.c_str());
  if (!maybe_tensor) {
    maybe_tensor = in_message.get<Tensor>();
    if (!maybe_tensor) {
      HOLOSCAN_LOG_ERROR("Tensor '{%s}' not found in message ", in_tensor_name.c_str());
      throw std::runtime_error(fmt::format("Tensor '{}' not found in message", in_tensor_name));
    }
  }
  return maybe_tensor;
}

void SegmentationPreprocessorOp::setup(OperatorSpec& spec) {
  auto& in_tensor = spec.input<gxf::Entity>("in_tensor");
  auto& out_tensor = spec.output<gxf::Entity>("out_tensor");

  spec.param(in_, "in", "Input", "Input channel.", &in_tensor);
  spec.param(out_, "out", "Output", "Output channel.", &out_tensor);

  spec.param(in_tensor_name_,
             "in_tensor_name",
             "InputTensorName",
             "Name of the input tensor.",
             std::string(""));
  spec.param(out_tensor_name_,
             "out_tensor_name",
             "OutputTensorName",
             "Name of the output tensor.",
             std::string(""));
  spec.param(data_format_,
             "data_format",
             "DataFormat",
             "Data format of network output.",
             std::string("hwc"));

  spec.param(normalize_means_,
             "normalize_means",
             "NormalizationMeans",
             "Means to use when normalizing input frame.");
  spec.param(normalize_stds_,
             "normalize_stds",
             "NormalizationSTDs",
             "Standard deviations to use when normalizing input frame.");

  spec.param(allocator_, "allocator", "Allocator", "Output Allocator");

  cuda_stream_handler_.define_params(spec);

  // TODO (gbae): spec object holds an information about errors
  // TODO (gbae): incorporate std::expected to not throw exceptions
}

void SegmentationPreprocessorOp::compute(InputContext& op_input, OutputContext& op_output,
                                          ExecutionContext& context) {
  constexpr size_t kMaxChannelCount = std::numeric_limits<output_type_t>::max();

  // Process input message
  // The type of `in_message` is 'holoscan::gxf::Entity'.
  auto in_message = op_input.receive<gxf::Entity>("in_tensor").value();

  // if (!in_message || in_message.value().is_null()) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }

  const std::string in_tensor_name = in_tensor_name_.get();
  // Get tensor attached to the message
  auto in_tensor = getTensorByName(in_message, in_tensor_name);
  // get the CUDA stream from the input message
  gxf_result_t stream_handler_result =
      cuda_stream_handler_.from_message(context.context(), in_message);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
  }

  segmentation_preprocessor::Shape shape = {};
  switch (data_format_value_) {
    case DataFormat::kHWC: {
      shape.height = in_tensor->shape()[0];
      shape.width = in_tensor->shape()[1];
      shape.channels = in_tensor->shape()[2];
    } break;
    case DataFormat::kNCHW: {
      shape.channels = in_tensor->shape()[1];
      shape.height = in_tensor->shape()[2];
      shape.width = in_tensor->shape()[3];
    } break;
    case DataFormat::kNHWC: {
      shape.height = in_tensor->shape()[1];
      shape.width = in_tensor->shape()[2];
      shape.channels = in_tensor->shape()[3];
    } break;
  }

  if (static_cast<size_t>(shape.channels) > kMaxChannelCount) {
    throw std::runtime_error(fmt::format(
        "Input channel count larger than allowed: {} > {}", shape.channels, kMaxChannelCount));
  }

  // Create a new message (nvidia::gxf::Entity)
  auto out_message = nvidia::gxf::Entity::New(context.context());

  const std::string out_tensor_name = out_tensor_name_.get();
  auto out_tensor = out_message.value().add<nvidia::gxf::Tensor>(out_tensor_name.c_str());
  if (!out_tensor) { throw std::runtime_error("Failed to allocator_ate output tensor"); }

  // Allocate and convert output buffer on the device.
  nvidia::gxf::Shape output_shape{shape.height, shape.width, shape.channels};

  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator_ator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                       allocator_.get()->gxf_cid());
  out_tensor.value()->reshape<float>(
      output_shape, nvidia::gxf::MemoryStorageType::kDevice, allocator_ator.value());
  if (!out_tensor.value()->pointer()) {
    throw std::runtime_error("Failed to allocator_ate output tensor buffer.");
  }

  const float* in_tensor_data = static_cast<float*>(in_tensor->data());

  nvidia::gxf::Expected<float*> out_tensor_data = out_tensor.value()->data<float>();
  if (!out_tensor_data) { throw std::runtime_error("Failed to get out tensor data!"); }

  cuda_preprocess(data_format_value_, shape, in_tensor_data, out_tensor_data.value(), means_cuda_,
                  stds_cuda_);

  // pass the CUDA stream to the output message
  stream_handler_result = cuda_stream_handler_.to_message(out_message);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to add the CUDA stream to the outgoing messages");
  }

  auto result = gxf::Entity(std::move(out_message.value()));
  op_output.emit(result);
}

void SegmentationPreprocessorOp::start() {
  const std::string data_format = data_format_.get();
  if (data_format == "nchw") {
    data_format_value_ = DataFormat::kNCHW;
  } else if (data_format == "hwc") {
    data_format_value_ = DataFormat::kHWC;
  } else if (data_format == "nhwc") {
    data_format_value_ = DataFormat::kNHWC;
  } else {
    throw std::runtime_error(fmt::format("Unsupported data format type {}", data_format));
  }

  means_host_ = normalize_means_;
  stds_host_ = normalize_stds_;

  if (means_host_.empty() || stds_host_.empty()) {
    throw std::runtime_error("Invalid per channel mean and std vectors");
  }

  const std::size_t sizeInBytesMeans = sizeof(float) * means_host_.size();
  const std::size_t sizeInBytesStds = sizeof(float) * stds_host_.size();

  // means vector
  cudaError_t cuda_error = CUDA_TRY(cudaMalloc((void **)&means_cuda_, sizeInBytesMeans));
  if (cudaSuccess != cuda_error) {
    throw std::runtime_error("Could not allocator_ate cuda memory for per channel mean vector");
  }
  cuda_error = CUDA_TRY(cudaMemcpy(means_cuda_, means_host_.data(), sizeInBytesMeans,
                                   cudaMemcpyHostToDevice));
  if (cudaSuccess != cuda_error) {
    throw std::runtime_error("Failed to copy per channel std vector from host to device");
  }

  // std vector
  cuda_error = CUDA_TRY(cudaMalloc((void **)&stds_cuda_, sizeInBytesStds));
  if (cudaSuccess != cuda_error) {
    throw std::runtime_error("Could not allocator_ate cuda memory for per channel std vector");
  }
  cuda_error = CUDA_TRY(cudaMemcpy(stds_cuda_, stds_host_.data(), sizeInBytesStds,
                                                                      cudaMemcpyHostToDevice));
  if (cudaSuccess != cuda_error) {
    throw std::runtime_error("Failed to copy per channel std vector from host to device");
  }
}

void SegmentationPreprocessorOp::stop() {
  CUDA_TRY(cudaFree(means_cuda_));
  CUDA_TRY(cudaFree(stds_cuda_));
}

}  // namespace holoscan::ops::orsi
