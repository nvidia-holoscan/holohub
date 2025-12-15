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
#include "holoscan/utils/cuda_macros.hpp"

using holoscan::ops::segmentation_postprocessor::cuda_postprocess;
using holoscan::ops::segmentation_postprocessor::DataFormat;
using holoscan::ops::segmentation_postprocessor::NetworkOutputType;
using holoscan::ops::segmentation_postprocessor::output_type_t;
using holoscan::ops::segmentation_postprocessor::Shape;

using holoscan::ops::orsi::segmentation_postprocessor::cuda_resize;

namespace holoscan::ops::orsi {

void SegmentationPostprocessorOp::initialize() {
#if CUDART_VERSION >= 13000
  // Workaround pending proper NPP support to get stream context in CUDA 13.0+
  int device = 0;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGetDevice(&device), "Failed to get CUDA device");

  cudaDeviceProp prop{};
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGetDeviceProperties(&prop, device),
                                  "Failed to get CUDA device properties");

  npp_stream_ctx_.nCudaDeviceId = device;
  npp_stream_ctx_.nMultiProcessorCount = prop.multiProcessorCount;
  npp_stream_ctx_.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  npp_stream_ctx_.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  npp_stream_ctx_.nSharedMemPerBlock = prop.sharedMemPerBlock;
  npp_stream_ctx_.nCudaDevAttrComputeCapabilityMajor = prop.major;
  npp_stream_ctx_.nCudaDevAttrComputeCapabilityMinor = prop.minor;
#else
  auto nppStatus = nppGetStreamContext(&npp_stream_ctx_);
  if (NPP_SUCCESS != nppStatus) {
    throw std::runtime_error("Failed to get NPP CUDA stream context");
  }
#endif

  Operator::initialize();
}

void SegmentationPostprocessorOp::setup(OperatorSpec& spec) {
  auto& in_tensor = spec.input<gxf::Entity>("in_tensor");
  auto& out_tensor = spec.output<gxf::Entity>("out_tensor");

  spec.param(in_, "in", "Input", "Input channel.", &in_tensor);
  spec.param(out_, "out", "Output", "Output channel.", &out_tensor);

  spec.param(in_tensor_name_,
             "in_tensor_name",
             "InputTensorName",
             "Name of the input tensor.",
             std::string());
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
             "Name of the output tensor.",
             std::string());
  spec.param(output_roi_rect_,
             "output_roi_rect",
             "output image roi rect",
             "output image roi rect [x, y, width, height]");

  spec.param(output_img_size_,
             "output_img_size",
             "Output image size after resize",
             "Output image size [ width, height ] after resize");

  cuda_stream_handler_.define_params(spec);

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
      cuda_stream_handler_.from_message(context.context(), in_message);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
  }

  const auto& in_shape = in_tensor->shape();
  const auto in_rank = in_shape.size();

  if (in_rank != 4) {
    throw std::runtime_error(
        fmt::format("Unsupported input tensor rank {}. Supported rank: 4!", in_rank));
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

  if (static_cast<size_t>(shape.channels) > kMaxChannelCount) {
    throw std::runtime_error(fmt::format(
        "Input channel count larger than allowed: {} > {}", shape.channels, kMaxChannelCount));
  }

  nvidia::gxf::Shape scratch_buffer_process_size{shape.height, shape.width, 1};

  // Create a new message (nvidia::gxf::Entity)
  auto out_message = nvidia::gxf::Entity::New(context.context());

  const std::string out_tensor_name = out_tensor_name_.get();
  auto out_tensor = out_message.value().add<nvidia::gxf::Tensor>(out_tensor_name.c_str());
  if (!out_tensor) {
    throw std::runtime_error("Failed to allocate output tensor");
  }

  const bool roi_enabled = output_roi_.width > 0 && output_roi_.height > 0;
  // Allocate and convert output buffer on the device.
  nvidia::gxf::Shape output_shape{shape.height, shape.width, 1};
  nvidia::gxf::Shape scratch_buffer_process_shape{shape.height, shape.width, 1};
  nvidia::gxf::Shape scratch_buffer_resize_shape{output_roi_.width, output_roi_.height, 1};

  auto frag = fragment();
  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(frag->executor().context(),
                                                                       allocator_->gxf_cid());
  if (roi_enabled) {
    // set new output shape
    output_shape = nvidia::gxf::Shape{output_size_.height, output_size_.width, 1};
    // resize scratch buffer
    if (scratch_buffer_process_->size() == 0) {
      const uint64_t buffer_size = scratch_buffer_process_shape.dimension(0) *
                                   scratch_buffer_process_shape.dimension(1) *
                                   scratch_buffer_process_shape.dimension(2);
      scratch_buffer_process_->resize(
          allocator.value(), buffer_size, nvidia::gxf::MemoryStorageType::kDevice);
    }

    if (scratch_buffer_resize_->size() == 0) {
      const uint64_t buffer_size = scratch_buffer_resize_shape.dimension(0) *
                                   scratch_buffer_resize_shape.dimension(1) *
                                   scratch_buffer_resize_shape.dimension(2);
      scratch_buffer_resize_->resize(
          allocator.value(), buffer_size, nvidia::gxf::MemoryStorageType::kDevice);
    }
  }

  // reshape out tensor buffer
  out_tensor.value()->reshape<uint8_t>(
      output_shape, nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
  if (!out_tensor.value()->pointer()) {
    throw std::runtime_error("Failed to allocate output tensor buffer.");
  }
  nvidia::gxf::Expected<uint8_t*> out_tensor_data = out_tensor.value()->data<uint8_t>();
  if (!out_tensor_data) {
    throw std::runtime_error("Failed to get out tensor data!");
  }

  // choose output buffer for post processing. By default use out tensor buffer
  // When ROI enabled use scratch buffer
  uint8_t* post_process_output_buffer = out_tensor_data.value();
  if (roi_enabled) {
    post_process_output_buffer = scratch_buffer_process_->pointer();
  }

  const float* in_tensor_data = static_cast<float*>(in_tensor->data());

  cuda_postprocess(network_output_type_value_,
                   data_format_value_,
                   shape,
                   in_tensor_data,
                   post_process_output_buffer,
                   cuda_stream_handler_.get_cuda_stream(context.context()));

  if (roi_enabled) {
    // ------------------------------------------------------------------------
    //
    //  Step 1: resize to original ROI region dimensions
    //
    const NppiSize src_size = {static_cast<int>(scratch_buffer_process_shape.dimension(0)),
                               static_cast<int>(scratch_buffer_process_shape.dimension(1))};
    const NppiRect src_roi = {0,
                              0,
                              static_cast<int>(scratch_buffer_process_shape.dimension(0)),
                              static_cast<int>(scratch_buffer_process_shape.dimension(1))};
    const NppiSize dst_size = {static_cast<int>(output_roi_.width),
                               static_cast<int>(output_roi_.height)};
    const NppiRect dst_roi = {
        0, 0, static_cast<int>(output_roi_.width), static_cast<int>(output_roi_.height)};

    const uint8_t* src_buffer = scratch_buffer_process_->pointer();
    uint8_t* dst_buffer = scratch_buffer_resize_->pointer();

    NppStatus status = nppiResize_8u_C1R_Ctx(src_buffer,
                                             src_size.width,
                                             src_size,
                                             src_roi,
                                             dst_buffer,
                                             dst_size.width,
                                             dst_size,
                                             dst_roi,
                                             NPPI_INTER_CUBIC,
                                             npp_stream_ctx_);

    if (status != NPP_SUCCESS) {
      throw std::runtime_error("Failed to insert post processed buffer into output buffer");
    }

    // ------------------------------------------------------------------------
    //
    //  Step 2: Insert into output buffer
    //

    const auto converted_tensor_ptr = scratch_buffer_resize_->pointer();
    out_tensor_data = out_tensor.value()->data<uint8_t>();
    cuda_resize({output_roi_.height, output_roi_.width, 1},
                {output_shape.dimension(1), output_shape.dimension(1), output_shape.dimension(2)},
                converted_tensor_ptr,
                out_tensor_data.value(),
                output_roi_.x,
                output_roi_.y);
  }

  // pass the CUDA stream to the output message
  stream_handler_result = cuda_stream_handler_.to_message(out_message);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to add the CUDA stream to the outgoing messages");
  }

  auto result = gxf::Entity(std::move(out_message.value()));
  op_output.emit(result);
}

void SegmentationPostprocessorOp::start() {
  const std::string network_output_type = network_output_type_.get();
  if (network_output_type == "sigmoid") {
    network_output_type_value_ = NetworkOutputType::kSigmoid;
  } else if (network_output_type == "softmax") {
    network_output_type_value_ = NetworkOutputType::kSoftmax;
  } else {
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

  const std::vector<int32_t> roi = output_roi_rect_;
  if (roi.size() != 4) {
    throw std::runtime_error(
        fmt::format("Invalid number: {} of dimensions for output image region of interest. "
                    "Expected ROI in format [x, y, width, height]!",
                    roi.size()));
  }
  output_roi_.x = roi[0];
  output_roi_.y = roi[1];
  output_roi_.width = roi[2];
  output_roi_.height = roi[3];

  const std::vector<int32_t> out_img_size = output_img_size_;
  if (out_img_size.size() != 2) {
    throw std::runtime_error(
        fmt::format("Invalid number: {} of dimensions for output image size. Expected size in "
                    "format [width, height]!",
                    out_img_size.size()));
  }
  output_size_.width = out_img_size[0];
  output_size_.height = out_img_size[1];

  scratch_buffer_process_ = std::make_unique<nvidia::gxf::MemoryBuffer>();
  scratch_buffer_resize_ = std::make_unique<nvidia::gxf::MemoryBuffer>();
}

void SegmentationPostprocessorOp::stop() {
  scratch_buffer_process_.reset();
  scratch_buffer_resize_.reset();
}

}  // namespace holoscan::ops::orsi
