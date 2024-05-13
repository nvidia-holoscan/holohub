/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "npp_filter.hpp"

#include <npp.h>
#include <gxf/multimedia/video.hpp>

struct NppStreamContext_ : public NppStreamContext {};

namespace holoscan::ops {

/**
 * @brief Given a mask size adjust the ROI avoiding out of bounds memory accesses.
 *
 * @param mask_size mask size
 * @param size ROI size
 * @return adjusted ROI
 */
static NppiSize adjust_roi(uint32_t mask_size, const NppiSize& size) {
  return NppiSize{size.width - (int(mask_size) - 1), size.height - (int(mask_size) - 1)};
}

/**
 * @brief Given a mask size adjust the ROI avoiding out of bounds memory accesses.
 *
 * @param mask_size mask size
 * @param color_plane color plane information
 * @param address color plane information
 * @return adjusted address
 */
static void* adjust_roi_address(uint32_t mask_size, const nvidia::gxf::ColorPlane& color_plane,
                                void* address) {
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(address) +
                                 (mask_size / 2) * color_plane.stride +
                                 (mask_size / 2) * color_plane.bytes_per_pixel);
}

void NppFilterOp::initialize() {
  npp_stream_ctx_ = std::make_shared<NppStreamContext_>();
  auto nppStatus = nppGetStreamContext(npp_stream_ctx_.get());
  if (NPP_SUCCESS != nppStatus) {
    throw std::runtime_error("Failed to get NPP CUDA stream context");
  }
  Operator::initialize();
}

void NppFilterOp::setup(OperatorSpec& spec) {
  spec.param(filter_,
             "filter",
             "Filter",
             "Name of the filter to apply (supported Gauss, SobelHoriz, SobelVert).",
             std::string(""));
  spec.param(mask_size_,
             "mask_size",
             "MaskSize",
             "Filter mask size (supported values 3, 5, 7, 9, 11, 13).",
             3u);
  spec.param(allocator_, "allocator", "Allocator", "Allocator to allocate output tensor.");

  spec.input<holoscan::gxf::Entity>("input");
  spec.output<holoscan::gxf::Entity>("output");

  cuda_stream_handler_.defineParams(spec);
}

void NppFilterOp::compute(InputContext& op_input, OutputContext& op_output,
                          ExecutionContext& context) {
  auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
  if (!maybe_entity) { throw std::runtime_error("Failed to receive input"); }

  auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

  // get the CUDA stream from the input message
  gxf_result_t stream_handler_result = cuda_stream_handler_.fromMessage(context.context(), entity);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
  }

  // assign the CUDA stream to the NPP stream context
  npp_stream_ctx_->hStream = cuda_stream_handler_.getCudaStream(context.context());

  nvidia::gxf::VideoBufferInfo in_video_buffer_info{};
  void* in_pointer;

  const auto maybe_video_buffer = entity.get<nvidia::gxf::VideoBuffer>();
  if (maybe_video_buffer) {
    const auto video_buffer = maybe_video_buffer.value();

    if (video_buffer->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
      throw std::runtime_error("VideoBuffer must be in device memory");
    }

    in_video_buffer_info = video_buffer->video_frame_info();

    if (in_video_buffer_info.color_format != nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA) {
      throw std::runtime_error("Input VideoBuffer must be of format GXF_VIDEO_FORMAT_RGBA");
    }

    in_pointer = video_buffer->pointer();
  } else {
    const auto maybe_tensor = entity.get<nvidia::gxf::Tensor>();
    if (!maybe_tensor) {
      throw std::runtime_error("Neither VideoBuffer not Tensor found in message");
    }

    const auto tensor = maybe_tensor.value();

    if (tensor->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
      throw std::runtime_error("Tensor must be in device memory");
    }

    if ((tensor->rank() != 3) || (tensor->shape().dimension(2) != 4) ||
        (tensor->element_type() != nvidia::gxf::PrimitiveType::kUnsigned8)) {
      throw std::runtime_error("Tensor must be of rank 3 and have 4 uint8 components");
    }

    in_video_buffer_info.color_format = nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA;
    in_video_buffer_info.width = tensor->shape().dimension(1);
    in_video_buffer_info.height = tensor->shape().dimension(0);
    in_video_buffer_info.color_planes.resize(1);
    in_video_buffer_info.color_planes[0].color_space = "RGBA";
    in_video_buffer_info.color_planes[0].bytes_per_pixel =
        tensor->bytes_per_element() * tensor->shape().dimension(2);
    in_video_buffer_info.color_planes[0].width = in_video_buffer_info.width;
    in_video_buffer_info.color_planes[0].height = in_video_buffer_info.height;
    in_video_buffer_info.color_planes[0].stride = tensor->stride(0);
    in_video_buffer_info.color_planes[0].size =
        in_video_buffer_info.color_planes[0].height * in_video_buffer_info.color_planes[0].stride;
    in_pointer = tensor->pointer();
  }

  // get handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      fragment()->executor().context(), allocator_->gxf_cid());

  // allocate output message
  nvidia::gxf::Expected<nvidia::gxf::Entity> out_message =
      nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE};
  nvidia::gxf::VideoBufferInfo out_video_buffer_info{};
  void* out_pointer;
  if (maybe_video_buffer) {
    // if we received a video buffer allocate a video buffer as output as well
    out_message = nvidia::gxf::Entity::New(context.context());
    if (!out_message) { throw std::runtime_error("Failed to allocate message for the output."); }

    auto video_buffer = out_message.value().add<nvidia::gxf::VideoBuffer>("output");

    video_buffer.value()->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
        in_video_buffer_info.width,
        in_video_buffer_info.height,
        nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR,
        nvidia::gxf::MemoryStorageType::kDevice,
        allocator.value());
    if (!video_buffer.value()->pointer()) {
      throw std::runtime_error("Failed to allocate render output buffer.");
    }
    out_video_buffer_info = video_buffer.value()->video_frame_info();
    out_pointer = video_buffer.value()->pointer();
  } else {
    // if we received a video buffer allocate a video buffer as output as well
    nvidia::gxf::Shape shape{static_cast<int32_t>(in_video_buffer_info.height),
                             static_cast<int32_t>(in_video_buffer_info.width),
                             4};
    out_message = CreateTensorMap(
        context.context(),
        allocator.value(),
        {{"output",
          nvidia::gxf::MemoryStorageType::kDevice,
          shape,
          nvidia::gxf::PrimitiveType::kUnsigned8,
          0,
          nvidia::gxf::ComputeTrivialStrides(
              shape, nvidia::gxf::PrimitiveTypeSize(nvidia::gxf::PrimitiveType::kUnsigned8))}},
        false);

    if (!out_message) { std::runtime_error("failed to create out_message"); }
    const auto tensor = out_message.value().get<nvidia::gxf::Tensor>();
    if (!tensor) { std::runtime_error("failed to create out_tensor"); }

    void* out_tensor_data = tensor.value()->pointer();
    if (out_tensor_data == nullptr) {
      throw std::runtime_error("Failed to allocate memory for the output image");
    }

    out_video_buffer_info.color_format = nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA;
    out_video_buffer_info.width = tensor.value()->shape().dimension(1);
    out_video_buffer_info.height = tensor.value()->shape().dimension(0);
    out_video_buffer_info.color_planes.resize(1);
    out_video_buffer_info.color_planes[0].color_space = "RGBA";
    out_video_buffer_info.color_planes[0].bytes_per_pixel =
        tensor.value()->bytes_per_element() * tensor.value()->shape().dimension(2);
    out_video_buffer_info.color_planes[0].width = out_video_buffer_info.width;
    out_video_buffer_info.color_planes[0].height = out_video_buffer_info.height;
    out_video_buffer_info.color_planes[0].stride = tensor.value()->stride(0);
    out_video_buffer_info.color_planes[0].size =
        out_video_buffer_info.color_planes[0].height * out_video_buffer_info.color_planes[0].stride;
    out_pointer = tensor.value()->pointer();
  }

  const NppiSize src_size = {static_cast<int>(in_video_buffer_info.width),
                             static_cast<int>(in_video_buffer_info.height)};
  NppStatus status = NPP_ERROR;
  // Execute the filter.
  // Note: these filters execute neighborhood operations. Therefore the ROI (region of interest),
  // needs to be adjusted to avoid accessing pixels outside of the image. This will result in a
  // black border on the output image. See
  // https://docs.nvidia.com/cuda/archive/11.1.0/npp/nppi_conventions_lb.html#sampling_beyond_image_boundaries
  // for more information
  if (filter_.get() == "Gauss") {
    NppiMaskSize nppi_mask_size_enum;
    switch (mask_size_) {
      case 3:
        nppi_mask_size_enum = NPP_MASK_SIZE_3_X_3;
        break;
      case 5:
        nppi_mask_size_enum = NPP_MASK_SIZE_5_X_5;
        break;
      case 7:
        nppi_mask_size_enum = NPP_MASK_SIZE_7_X_7;
        break;
      case 9:
        nppi_mask_size_enum = NPP_MASK_SIZE_9_X_9;
        break;
      case 11:
        nppi_mask_size_enum = NPP_MASK_SIZE_11_X_11;
        break;
      case 13:
        nppi_mask_size_enum = NPP_MASK_SIZE_13_X_13;
        break;
      default:
        throw std::runtime_error("Unsupported mask size");
    }

    status = nppiFilterGauss_8u_C4R_Ctx(
        static_cast<const Npp8u*>(
            adjust_roi_address(mask_size_, in_video_buffer_info.color_planes[0], in_pointer)),
        in_video_buffer_info.color_planes[0].stride,
        static_cast<Npp8u*>(
            adjust_roi_address(mask_size_, out_video_buffer_info.color_planes[0], out_pointer)),
        out_video_buffer_info.color_planes[0].stride,
        adjust_roi(mask_size_, src_size),
        nppi_mask_size_enum,
        *npp_stream_ctx_.get());
  } else if (filter_.get() == "SobelHoriz") {
    const uint32_t mask_size = 3;
    status = nppiFilterSobelHoriz_8u_C4R_Ctx(
        static_cast<const Npp8u*>(
            adjust_roi_address(mask_size, in_video_buffer_info.color_planes[0], in_pointer)),
        in_video_buffer_info.color_planes[0].stride,
        static_cast<Npp8u*>(
            adjust_roi_address(mask_size, out_video_buffer_info.color_planes[0], out_pointer)),
        out_video_buffer_info.color_planes[0].stride,
        adjust_roi(mask_size, src_size),
        *npp_stream_ctx_.get());
  } else if (filter_.get() == "SobelVert") {
    const uint32_t mask_size = 3;
    status = nppiFilterSobelVert_8u_C4R_Ctx(
        static_cast<const Npp8u*>(
            adjust_roi_address(mask_size, in_video_buffer_info.color_planes[0], in_pointer)),
        in_video_buffer_info.color_planes[0].stride,
        static_cast<Npp8u*>(
            adjust_roi_address(mask_size, out_video_buffer_info.color_planes[0], out_pointer)),
        out_video_buffer_info.color_planes[0].stride,
        adjust_roi(mask_size, src_size),
        *npp_stream_ctx_.get());
  } else {
    throw std::runtime_error(fmt::format("Unknown filter {}.", filter_.get()));
  }
  if (status != NPP_SUCCESS) {
    throw std::runtime_error(fmt::format("Filter {} failed with error {}", filter_.get(), status));
  }

  // pass the CUDA stream to the output message
  stream_handler_result = cuda_stream_handler_.toMessage(out_message);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to add the CUDA stream to the outgoing messages");
  }

  // Emit the tensor
  auto result = gxf::Entity(std::move(out_message.value()));
  op_output.emit(result);
}

}  // namespace holoscan::ops
