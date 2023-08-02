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

#include "segmentation_postprocessor.hpp"

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
                    _holoscan_cuda_err);                                                   \
    }                                                                                      \
    _holoscan_cuda_err;                                                                    \
  })

using holoscan::ops::orsi::segmentation_postprocessor::cuda_postprocess;
using holoscan::ops::orsi::segmentation_postprocessor::cuda_resize;
using holoscan::ops::orsi::segmentation_postprocessor::DataFormat;
using holoscan::ops::orsi::segmentation_postprocessor::NetworkOutputType;
using holoscan::ops::orsi::segmentation_postprocessor::output_type_t;
using holoscan::ops::orsi::segmentation_postprocessor::Shape;

namespace holoscan::ops::orsi {

// Utility sigmoid function for on host compute
double sigmoid(double a) {
      return 1.0 / (1.0 + exp(-a)); 
}

void SegmentationPostprocessorOp::setup(OperatorSpec& spec) {
  auto& in_tensor = spec.input<gxf::Entity>("in_tensor");
  auto& out_tensor = spec.output<gxf::Entity>("out_tensor");

  spec.param(in_, "in", "Input", "Input channel.", &in_tensor);
  spec.param(out_, "out", "Output", "Output channel.", &out_tensor);

  spec.param(in_tensor_name_,
             "in_tensor_name",
             "InputTensorName",
             "Name of the input tensor.");
  spec.param(network_output_type_,
             "network_output_type",
             "NetworkOutputType",
             "Network output type.",
             std::string("softmax"));
  spec.param(data_format_,
             "data_format",
             "DataFormat",
             "Data format of network output.",
             std::string("hwc"));
  spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
  // Modifications to Holoscan SDK 0.6 post processing operator
  spec.param(out_tensor_name_,
             "out_tensor_name",
             "OutputTensorName",
             "Name of the output tensor.");
  spec.param(cropped_width_, 
            "cropped_width", 
            "Cropped width",
            "Width for the reverse cropping. No actions if this value is zero.",
            0);
  spec.param(cropped_height_, 
            "cropped_height", 
            "Cropped height",
            "Height for the reverse cropping. No actions if this value is zero.",
            0);
  spec.param(offset_x_, 
            "offset_x", 
            "Offset x",
            "X coordinate of the top left corner from which the image resizing starts.",
            0);
  spec.param(offset_y_, 
            "offset_y", 
            "Offset y",
            "Y coordinate of the top left corner from which the image resizing starts.",
            0);
  spec.param(resolution_width_, 
             "resolution_width", 
             "Resolution width",
             "Width for the output resolution.", 
             1920);
  spec.param(resolution_height_, 
             "resolution_height", 
             "Resolution height",
             "Height for the output resolution.",
             1080);

  cuda_stream_handler_.defineParams(spec);

  // TODO (gbae): spec object holds an information about errors
  // TODO (gbae): incorporate std::expected to not throw exceptions
}

void SegmentationPostprocessorOp::compute(InputContext& op_input, OutputContext& op_output,
                                          ExecutionContext& context) {
  constexpr size_t kMaxChannelCount = std::numeric_limits<output_type_t>::max();

  // Process input message
  // The type of `in_message` is 'holoscan::gxf::Entity'.
  auto in_message = op_input.receive<gxf::Entity>("in_tensor").value();

  // if (!in_message || in_message.value().is_null()) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }

  const std::string in_tensor_name = in_tensor_name_.get();

  // Get tensor attached to the message
  // The type of `maybe_tensor` is 'std::shared_ptr<holoscan::Tensor>'.
  auto maybe_tensor = in_message.get<Tensor>(in_tensor_name.c_str());
  if (!maybe_tensor) {
    maybe_tensor = in_message.get<Tensor>();
    if (!maybe_tensor) {
      throw std::runtime_error(fmt::format("Tensor '{}' not found in message", in_tensor_name));
    }
  }
  auto in_tensor = maybe_tensor;

  // get the CUDA stream from the input message
  gxf_result_t stream_handler_result =
      cuda_stream_handler_.fromMessage(context.context(), in_message);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
  }

  segmentation_postprocessor::Shape shape = {};
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

  // TMP workaround for bug in Holoscan SDK's inference op
  int expected_shape[2] =  { 1, 1};
  if(in_tensor_name == "tool_seg_infer") {
    expected_shape[0] = 512;
    expected_shape[1] = 512;
  }

  if(   shape.width != expected_shape[0]
     ||shape.height != expected_shape[1]){

    HOLOSCAN_LOG_WARN(fmt::format(
        "Received shape [{}, {}] for tensor: {}. Expected [{}, {}]!",  shape.width,  shape.height, in_tensor_name,expected_shape[0], expected_shape[1]));
    shape.width = expected_shape[0];
    shape.height = expected_shape[1];
  }

  if (static_cast<size_t>(shape.channels) > kMaxChannelCount) {
    throw std::runtime_error(fmt::format(
        "Input channel count larger than allowed: {} > {}", shape.channels, kMaxChannelCount));
  }

  // Create a new message (nvidia::gxf::Entity)
  auto out_message = nvidia::gxf::Entity::New(context.context());

  const std::string out_tensor_name = out_tensor_name_.get();
  auto out_tensor = out_message.value().add<nvidia::gxf::Tensor>(out_tensor_name.c_str());
  if (!out_tensor) { throw std::runtime_error("Failed to allocate output tensor"); }

  // Allocate and convert output buffer on the device.
  nvidia::gxf::Shape output_shape{shape.height, shape.width, 1};

  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                       allocator_->gxf_cid());
  out_tensor.value()->reshape<uint8_t>(
      output_shape, nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
  if (!out_tensor.value()->pointer()) {
    throw std::runtime_error("Failed to allocate output tensor buffer.");
  }

  const float* in_tensor_data = static_cast<float*>(in_tensor->data());

  nvidia::gxf::Expected<uint8_t*> out_tensor_data = out_tensor.value()->data<uint8_t>();
  if (!out_tensor_data) { throw std::runtime_error("Failed to get out tensor data!"); }


  // process small tensor on host and not on GPU.
  if(network_output_type_value_ == NetworkOutputType::kRawValues && shape.height == 1 && shape.width == 1 && shape.channels == 1) {

    cudaError_t cuda_rv = cudaSuccess;

    float in_tensor_host = -1.0f;
    cuda_rv = CUDA_TRY(cudaMemcpy(&in_tensor_host, in_tensor_data, sizeof(float), cudaMemcpyDeviceToHost));
    const double sigmoid_value = sigmoid(in_tensor_host);
    const uint8_t sigmoid_result = sigmoid_value  > 0.5 ? 1 : 0;
    cuda_rv = CUDA_TRY(cudaMemcpy(out_tensor_data.value(), (void *) &sigmoid_result, sizeof(uint8_t), cudaMemcpyHostToDevice)); // works
  } else {
    cuda_postprocess(network_output_type_value_,
                    data_format_value_,
                    shape,
                    in_tensor_data,
                    out_tensor_data.value(),
                    cuda_stream_handler_.getCudaStream(context.context()));
  }


  if (cropped_width_ > 0 && cropped_height_ > 0) {
    
    nvidia::gxf::PrimitiveType out_primitive_type = out_tensor.value()->element_type();

    auto resize_result = resizeImage(out_tensor_data.value(), output_shape.dimension(0), output_shape.dimension(1), output_shape.dimension(2), out_primitive_type,
                                     	  cropped_width_, cropped_height_);

    if (!resize_result) {
      throw std::runtime_error("Failed to resize output image.");
    }

    const auto converted_tensor_ptr = resize_buffer_->pointer();

    output_shape = nvidia::gxf::Shape{resolution_height_, resolution_width_, 1};
    out_tensor.value()->reshape<uint8_t>(output_shape, nvidia::gxf::MemoryStorageType::kDevice, allocator.value());

    out_tensor_data = out_tensor.value()->data<uint8_t>();
    cuda_resize({cropped_height_, cropped_width_, 1},
                {output_shape.dimension(0), output_shape.dimension(1), output_shape.dimension(2)},
                converted_tensor_ptr,
                out_tensor_data.value(),
                offset_x_,
                offset_y_);
  }

  // pass the CUDA stream to the output message
  stream_handler_result = cuda_stream_handler_.toMessage(out_message);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to add the CUDA stream to the outgoing messages");
  }

  auto result = gxf::Entity(std::move(out_message.value()));
  op_output.emit(result);
}

void SegmentationPostprocessorOp::start() {

  resize_buffer_ = std::make_unique<nvidia::gxf::MemoryBuffer>();

  const std::string network_output_type = network_output_type_.get();
  if (network_output_type == "sigmoid") {
    network_output_type_value_ = NetworkOutputType::kSigmoid;
  } else if (network_output_type == "softmax") {
    network_output_type_value_ = NetworkOutputType::kSoftmax;
  } else if(network_output_type == "raw") {
    network_output_type_value_ = NetworkOutputType::kRawValues;
  }
  else {
    throw std::runtime_error(
        fmt::format("Unsupported network output type {}", network_output_type));
  }

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
}


void SegmentationPostprocessorOp::stop() {
  resize_buffer_.reset();
}


nvidia::gxf::Expected<void*> SegmentationPostprocessorOp::resizeImage(
                                 const void* in_tensor_data, const int32_t rows,
                                 const int32_t columns, const int16_t channels,
                                 const nvidia::gxf::PrimitiveType primitive_type,
                                 const int32_t resize_width,
                                 const int32_t resize_height) {
  if (resize_buffer_->size() == 0) {

    auto frag = fragment();
    // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto pool = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(frag->executor().context(),
                                                                    allocator_->gxf_cid());

    uint64_t buffer_size = resize_width * resize_height * channels;
    resize_buffer_->resize(pool.value(), buffer_size, nvidia::gxf::MemoryStorageType::kDevice);
  }

  const auto converted_tensor_ptr = resize_buffer_->pointer();
  if (converted_tensor_ptr == nullptr) {
    HOLOSCAN_LOG_ERROR("Failed to allocate memory for the resizing image");
    return nvidia::gxf::ExpectedOrCode(GXF_FAILURE, nullptr);
  }

  // Resize image
  const NppiSize src_size = {static_cast<int>(columns), static_cast<int>(rows)};
  const NppiRect src_roi = {0, 0, static_cast<int>(columns), static_cast<int>(rows)};
  const NppiSize dst_size = {static_cast<int>(resize_width), static_cast<int>(resize_height)};
  const NppiRect dst_roi = {0, 0, static_cast<int>(resize_width), static_cast<int>(resize_height)};

  NppStatus status = nppiResize_8u_C1R(static_cast<const Npp8u*>(in_tensor_data), columns * channels,
                                     src_size, src_roi, converted_tensor_ptr,
                                     resize_width * channels, dst_size, dst_roi, NPPI_INTER_CUBIC);

  if (status != NPP_SUCCESS) { return nvidia::gxf::ExpectedOrCode(GXF_FAILURE, nullptr); }

  return nvidia::gxf::ExpectedOrCode(GXF_SUCCESS, converted_tensor_ptr);
}

}  // namespace holoscan::ops
