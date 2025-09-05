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

#include "format_converter.hpp"

#include <string>
#include <utility>
#include <vector>

#include "gxf/std/tensor.hpp"
#if __has_include("gxf/std/dlpack_utils.hpp")
  #define GXF_HAS_DLPACK_SUPPORT 1
#else
  #define GXF_HAS_DLPACK_SUPPORT 0
  // Holoscan 1.0 used GXF without DLPack so gxf_tensor.hpp was needed to add it
  #include "holoscan/core/gxf/gxf_tensor.hpp"
#endif
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/utils/cuda_macros.hpp"

namespace holoscan::ops::orsi {

// Note that currently "uint8" for the `input_dtype` or `output_dtype` arguments to the operator
// is identical to using "rgb888" and the format "float32" implies a 32-bit floating point RGB
// image (3 channels).. With the exception of "yuv420" and "nv12", this operator currently operates
// only on three channel (RGB) or four channel (RGBA) inputs stored in a channel-packed format.
// In other words, for a given pixel, the RGB values are adjacent in memory rather than being
// stored as separate planes. There currently is no support for grayscale images.

static FormatDType toFormatDType(const std::string& str) {
  if (str == "rgb888") {
    return FormatDType::kRGB888;
  } else if (str == "uint8") {
    return FormatDType::kUnsigned8;
  } else if (str == "float32") {
    return FormatDType::kFloat32;
  } else if (str == "rgba8888") {
    return FormatDType::kRGBA8888;
  } else if (str == "yuv420") {
    return FormatDType::kYUV420;
  } else if (str == "nv12") {
    return FormatDType::kNV12;
  } else {
    return FormatDType::kUnknown;
  }
}

static constexpr FormatConversionType getFormatConversionType(FormatDType from, FormatDType to) {
  if (from != FormatDType::kUnknown && to != FormatDType::kUnknown && from == to) {
    return FormatConversionType::kNone;
  } else if (from == FormatDType::kUnsigned8 && to == FormatDType::kFloat32) {
    return FormatConversionType::kUnsigned8ToFloat32;
  } else if (from == FormatDType::kFloat32 && to == FormatDType::kUnsigned8) {
    return FormatConversionType::kFloat32ToUnsigned8;
  } else if (from == FormatDType::kUnsigned8 && to == FormatDType::kRGBA8888) {
    return FormatConversionType::kRGB888ToRGBA8888;
  } else if (from == FormatDType::kRGBA8888 && to == FormatDType::kUnsigned8) {
    return FormatConversionType::kRGBA8888ToRGB888;
  } else if (from == FormatDType::kRGBA8888 && to == FormatDType::kFloat32) {
    return FormatConversionType::kRGBA8888ToFloat32;
  } else if (from == FormatDType::kUnsigned8 && to == FormatDType::kYUV420) {
    return FormatConversionType::kRGB888ToYUV420;
  } else if (from == FormatDType::kYUV420 && to == FormatDType::kRGBA8888) {
    return FormatConversionType::kYUV420ToRGBA8888;
  } else if (from == FormatDType::kYUV420 && to == FormatDType::kUnsigned8) {
    return FormatConversionType::kYUV420ToRGB888;
  } else if (from == FormatDType::kNV12 && to == FormatDType::kUnsigned8) {
    return FormatConversionType::kNV12ToRGB888;
  } else {
    return FormatConversionType::kUnknown;
  }
}

static constexpr FormatDType normalizeFormatDType(FormatDType dtype) {
  switch (dtype) {
    case FormatDType::kRGB888:
      return FormatDType::kUnsigned8;
    default:
      return dtype;
  }
}

static constexpr nvidia::gxf::PrimitiveType primitiveTypeFromFormatDType(FormatDType dtype) {
  switch (dtype) {
    case FormatDType::kRGB888:
    case FormatDType::kRGBA8888:
    case FormatDType::kUnsigned8:
    case FormatDType::kYUV420:
    case FormatDType::kNV12:
      return nvidia::gxf::PrimitiveType::kUnsigned8;
    case FormatDType::kFloat32:
      return nvidia::gxf::PrimitiveType::kFloat32;
    default:
      return nvidia::gxf::PrimitiveType::kCustom;
  }
}

static constexpr FormatDType FormatDTypeFromPrimitiveType(nvidia::gxf::PrimitiveType type) {
  switch (type) {
    case nvidia::gxf::PrimitiveType::kUnsigned8:
      return FormatDType::kUnsigned8;
    case nvidia::gxf::PrimitiveType::kFloat32:
      return FormatDType::kFloat32;
    default:
      return FormatDType::kUnknown;
  }
}

static gxf_result_t verifyFormatDTypeChannels(FormatDType dtype, int channel_count) {
  switch (dtype) {
    case FormatDType::kRGB888:
      if (channel_count != 3) {
        HOLOSCAN_LOG_ERROR("Invalid channel count for RGB888 {} != 3\n", channel_count);
        return GXF_FAILURE;
      }
      break;
    case FormatDType::kRGBA8888:
      if (channel_count != 4) {
        HOLOSCAN_LOG_ERROR("Invalid channel count for RGBA8888 {} != 4\n", channel_count);
        return GXF_FAILURE;
      }
      break;
    default:
      break;
  }
  return GXF_SUCCESS;
}

void FormatConverterOp::initialize() {
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

void FormatConverterOp::start() {
  resize_buffer_ = std::make_unique<nvidia::gxf::MemoryBuffer>();
  channel_buffer_ = std::make_unique<nvidia::gxf::MemoryBuffer>();
  device_scratch_buffer_ = std::make_unique<nvidia::gxf::MemoryBuffer>();
  out_dtype_ = toFormatDType(out_dtype_str_.get());
  if (out_dtype_ == FormatDType::kUnknown) {
    throw std::runtime_error(
        fmt::format("Unsupported output format dtype: {}\n", out_dtype_str_.get()));
  }
  out_primitive_type_ = primitiveTypeFromFormatDType(out_dtype_);
  if (out_primitive_type_ == nvidia::gxf::PrimitiveType::kCustom) {
    throw std::runtime_error(
        fmt::format("Unsupported output format dtype: {}\n", out_dtype_str_.get()));
  }

  if (!in_dtype_str_.get().empty()) {
    in_dtype_ = toFormatDType(in_dtype_str_.get());
    if (in_dtype_ == FormatDType::kUnknown) {
      throw std::runtime_error(
          fmt::format("Unsupported input format dtype: {}\n", in_dtype_str_.get()));
    }
    format_conversion_type_ =
        getFormatConversionType(normalizeFormatDType(in_dtype_), normalizeFormatDType(out_dtype_));
    in_primitive_type_ = primitiveTypeFromFormatDType(in_dtype_);
  }

  switch (resize_mode_) {
    case 0:
      // resize_mode_.set(NPPI_INTER_CUBIC);
      resize_mode_ = NPPI_INTER_CUBIC;
      break;
    case 1:                                // NPPI_INTER_NN
    case 2:                                // NPPI_INTER_LINEAR
    case 4:                                // NPPI_INTER_CUBIC
    case 5:                                // NPPI_INTER_CUBIC2P_BSPLINE
    case 6:                                // NPPI_INTER_CUBIC2P_CATMULLROM
    case 7:                                // NPPI_INTER_CUBIC2P_B05C03
    case 8:                                // NPPI_INTER_SUPER
    case 16:                               // NPPI_INTER_LANCZOS
    case 17:                               // NPPI_INTER_LANCZOS3_ADVANCED
    case static_cast<int32_t>(0x8000000):  // NPPI_SMOOTH_EDGE
      break;
    default:
      throw std::runtime_error(fmt::format("Unsupported resize mode: {}\n", resize_mode_.get()));
  }
}

void FormatConverterOp::stop() {
  // Destroy the memory buffers here because GXF context is needed to destroy these buffers,
  // and GXF context would not be available after stop(). GXF context would not be available
  // when the FormatConverterOp destructor is being called as well.
  resize_buffer_.reset();
  channel_buffer_.reset();
  device_scratch_buffer_.reset();
}

void FormatConverterOp::compute(InputContext& op_input, OutputContext& op_output,
                                ExecutionContext& context) {
  // Process input message
  auto in_message = op_input.receive<gxf::Entity>("source_video").value();

  // get the CUDA stream from the input message
  gxf_result_t stream_handler_result =
      cuda_stream_handler_.from_message(context.context(), in_message);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
  }

  // assign the CUDA stream to the NPP stream context
  npp_stream_ctx_.hStream = cuda_stream_handler_.get_cuda_stream(context.context());

  nvidia::gxf::Shape out_shape{0, 0, 0};
  void* in_tensor_data = nullptr;
  nvidia::gxf::PrimitiveType in_primitive_type = nvidia::gxf::PrimitiveType::kCustom;
  nvidia::gxf::MemoryStorageType in_memory_storage_type = nvidia::gxf::MemoryStorageType::kHost;
  int32_t rows = 0;
  int32_t columns = 0;
  int16_t in_channels = 0;
  int16_t out_channels = 0;

  const std::vector<int32_t> out_img_size = output_img_size_;
  const std::vector<int32_t> src_img_roi = src_roi_rect_;

  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto pool =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

  // Get either the Tensor or VideoBuffer attached to the message
  bool is_video_buffer;
  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> video_buffer;
  try {
    video_buffer = holoscan::gxf::get_videobuffer(in_message);
    is_video_buffer = true;
  } catch (const std::runtime_error& r_) {
    HOLOSCAN_LOG_TRACE("Failed to read VideoBuffer with error: {}", std::string(r_.what()));
    is_video_buffer = false;
  }

  if (is_video_buffer) {
    // Convert VideoBuffer to Tensor
    auto frame = video_buffer.get();

    // NOTE: VideoBuffer::moveToTensor() converts VideoBuffer instance to the Tensor instance
    // with an unexpected shape:  [width, height] or [width, height, num_planes].
    // And, if we use moveToTensor() to convert VideoBuffer to Tensor, we may lose the the original
    // video buffer when the VideoBuffer instance is used in other places. For that reason, we
    // directly access internal data of VideoBuffer instance to access Tensor data.
    const auto& buffer_info = frame->video_frame_info();
    switch (buffer_info.color_format) {
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA:
        in_primitive_type = nvidia::gxf::PrimitiveType::kUnsigned8;
        in_channels = 4;  // RGBA
        out_channels = in_channels;
        out_shape = nvidia::gxf::Shape{
            static_cast<int32_t>(buffer_info.height), static_cast<int32_t>(buffer_info.width), 4};
        break;
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB:
        in_primitive_type = nvidia::gxf::PrimitiveType::kUnsigned8;
        in_channels = 3;  // RGB
        out_channels = in_channels;
        out_shape = nvidia::gxf::Shape{
            static_cast<int32_t>(buffer_info.height), static_cast<int32_t>(buffer_info.width), 3};
        break;
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12:
        in_primitive_type = nvidia::gxf::PrimitiveType::kUnsigned8;
        in_channels = buffer_info.color_planes.size();
        out_channels = 3;
        switch (out_dtype_) {
          case FormatDType::kRGB888:
            out_channels = 3;
            out_shape = nvidia::gxf::Shape{static_cast<int32_t>(buffer_info.height),
                                           static_cast<int32_t>(buffer_info.width),
                                           3};
            break;
          default:
            throw std::runtime_error(fmt::format("Unsupported format conversion: {} -> {}\n",
                                                 in_dtype_str_.get(),
                                                 out_dtype_str_.get()));
            break;
        }
        break;
      default:
        throw std::runtime_error(fmt::format("Unsupported input format: {}\n",
                                             static_cast<int>(buffer_info.color_format)));
    }

    // Get needed information from the tensor
    in_memory_storage_type = frame->storage_type();
    out_shape = nvidia::gxf::Shape{
        static_cast<int32_t>(buffer_info.height), static_cast<int32_t>(buffer_info.width), 4};
    in_tensor_data = frame->pointer();
    rows = buffer_info.height;
    columns = buffer_info.width;

    // If the buffer is in host memory, copy it to a device (GPU) buffer
    // as needed for the NPP resize/convert operations.
    if (in_memory_storage_type == nvidia::gxf::MemoryStorageType::kHost) {
      size_t buffer_size = rows * columns * in_channels;
      if (buffer_size > device_scratch_buffer_->size()) {
        device_scratch_buffer_->resize(
            pool.value(), buffer_size, nvidia::gxf::MemoryStorageType::kDevice);
        if (!device_scratch_buffer_->pointer()) {
          throw std::runtime_error(
              fmt::format("Failed to allocate device scratch buffer ({} bytes)", buffer_size));
        }
      }
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpy(device_scratch_buffer_->pointer(),
                                                frame->pointer(),
                                                buffer_size,
                                                cudaMemcpyHostToDevice),
                                     "Failed to copy input data to device scratch buffer");
      in_tensor_data = device_scratch_buffer_->pointer();
      in_memory_storage_type = nvidia::gxf::MemoryStorageType::kDevice;
    }
  } else {
    const auto maybe_tensor = in_message.get<Tensor>(in_tensor_name_.get().c_str());
    if (!maybe_tensor) {
      throw std::runtime_error(
          fmt::format("Tensor '{}' not found in message.\n", in_tensor_name_.get()));
    }
    // Tensor in_tensor;
    auto in_tensor = maybe_tensor;

    // Get needed information from the tensor
#if GXF_HAS_DLPACK_SUPPORT
    // cast Holoscan::Tensor to nvidia::gxf::Tensor to use its APIs directly
    nvidia::gxf::Tensor in_tensor_gxf{in_tensor->dl_ctx()};
#else
    // cast Holoscan::Tensor to GXFTensor so attribute access code can remain as-is
    holoscan::gxf::GXFTensor in_tensor_gxf{in_tensor->dl_ctx()};
#endif
    out_shape = in_tensor_gxf.shape();
    in_tensor_data = in_tensor_gxf.pointer();
    if (in_tensor_data == nullptr) {
      // This should never happen, but just in case...
      HOLOSCAN_LOG_ERROR("Unable to get tensor data pointer. nullptr returned.");
    }
    in_primitive_type = in_tensor_gxf.element_type();
    in_memory_storage_type = in_tensor_gxf.storage_type();
    rows = in_tensor_gxf.shape().dimension(0);
    columns = in_tensor_gxf.shape().dimension(1);
    in_channels = in_tensor_gxf.shape().dimension(2);
    out_channels = in_channels;
  }

  if (in_memory_storage_type != nvidia::gxf::MemoryStorageType::kDevice) {
    throw std::runtime_error(fmt::format(
        "Tensor('{}') or VideoBuffer is not allocated on device.\n", in_tensor_name_.get()));
  }

  if (in_dtype_ == FormatDType::kUnknown) {
    in_primitive_type_ = in_primitive_type;
    in_dtype_ = FormatDTypeFromPrimitiveType(in_primitive_type_);
    format_conversion_type_ =
        getFormatConversionType(normalizeFormatDType(in_dtype_), normalizeFormatDType(out_dtype_));
  }

  // Check if input tensor is consistent over all the frames
  if (in_primitive_type != in_primitive_type_) {
    throw std::runtime_error("Input tensor element type is inconsistent over all the frames.\n");
  }

  // Check if the input/output tensor is compatible with the format conversion
  if (format_conversion_type_ == FormatConversionType::kUnknown) {
    throw std::runtime_error(fmt::format("Unsupported format conversion: {} ({}) -> {}\n",
                                         in_dtype_str_.get(),
                                         static_cast<uint32_t>(in_dtype_),
                                         out_dtype_str_.get()));
  }

  // Check that if the format requires a specific number of channels they are consistent.
  // Some formats (e.g. float32) are agnostic to channel count, while others (e.g. RGB888) have a
  // specific channel count.
  if (GXF_SUCCESS != verifyFormatDTypeChannels(in_dtype_, in_channels)) {
    throw std::runtime_error(
        fmt::format("Failed to verify the channels for the expected input dtype [{}]: {}.",
                    static_cast<uint32_t>(in_dtype_),
                    in_channels));
  }

  // Resize the input image before converting data type
  if (out_img_size[0] > 0 && out_img_size[1] > 0) {
    auto resize_result = resizeImage(
        in_tensor_data, rows, columns, in_channels, in_primitive_type, out_img_size, src_img_roi);
    if (!resize_result) {
      throw std::runtime_error("Failed to resize image.\n");
    }

    // Update the tensor pointer and shape
    out_shape = nvidia::gxf::Shape{out_img_size[1], out_img_size[0], in_channels};
    in_tensor_data = resize_result.value();
    rows = out_img_size[1];
    columns = out_img_size[0];
  }

  // Create output message
  const uint32_t dst_typesize = nvidia::gxf::PrimitiveTypeSize(out_primitive_type_);

  // Adjust output shape if the conversion involves the change in the channel dimension
  switch (format_conversion_type_) {
    case FormatConversionType::kRGB888ToRGBA8888: {
      out_channels = 4;
      out_shape = nvidia::gxf::Shape{out_shape.dimension(0), out_shape.dimension(1), out_channels};
      break;
    }
    case FormatConversionType::kRGBA8888ToRGB888:
    case FormatConversionType::kNV12ToRGB888:
    case FormatConversionType::kYUV420ToRGB888:
    case FormatConversionType::kRGBA8888ToFloat32: {
      out_channels = 3;
      out_shape = nvidia::gxf::Shape{out_shape.dimension(0), out_shape.dimension(1), out_channels};
      break;
    }
    default:
      break;
  }

  // Check that if the format requires a specific number of channels they are consistent.
  // Some formats (e.g. float32) are agnostic to channel count, while others (e.g. RGB888) have a
  // specific channel count.
  if (GXF_SUCCESS != verifyFormatDTypeChannels(out_dtype_, out_channels)) {
    throw std::runtime_error(
        fmt::format("Failed to verify the channels for the expected output dtype [{}]: {}.",
                    static_cast<int32_t>(out_dtype_),
                    out_channels));
  }

  nvidia::gxf::Expected<nvidia::gxf::Entity> out_message =
      CreateTensorMap(context.context(),
                      pool.value(),
                      {{out_tensor_name_.get(),
                        nvidia::gxf::MemoryStorageType::kDevice,
                        out_shape,
                        out_primitive_type_,
                        0,
                        nvidia::gxf::ComputeTrivialStrides(out_shape, dst_typesize)}},
                      false);

  if (!out_message) {
    std::runtime_error("failed to create out_message");
  }
  const auto out_tensor = out_message.value().get<nvidia::gxf::Tensor>();
  if (!out_tensor) {
    std::runtime_error("failed to create out_tensor");
  }

  // Set tensor to constant using NPP
  if (in_channels == 2 || in_channels == 3 || in_channels == 4) {
    // gxf_result_t convert_result = convertTensorFormat(
    convertTensorFormat(
        in_tensor_data, out_tensor.value()->pointer(), rows, columns, in_channels, out_channels);

  } else {
    throw std::runtime_error("Only support 3 or 4 channel input tensor");
  }

  // pass the CUDA stream to the output message
  stream_handler_result = cuda_stream_handler_.to_message(out_message);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to add the CUDA stream to the outgoing messages");
  }

  // Emit the tensor
  auto result = gxf::Entity(std::move(out_message.value()));
  op_output.emit(result);
}

nvidia::gxf::Expected<void*> FormatConverterOp::resizeImage(
    const void* in_tensor_data, const int32_t rows, const int32_t columns, const int16_t channels,
    const nvidia::gxf::PrimitiveType primitive_type, const std::vector<int32_t>& output_img_size,
    const std::vector<int32_t>& src_img_rect) {
  if (resize_buffer_->size() == 0) {
    auto frag = fragment();

    // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto pool = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(frag->executor().context(),
                                                                    allocator_->gxf_cid());

    uint64_t buffer_size = output_img_size[0] * output_img_size[1] * channels;
    resize_buffer_->resize(pool.value(), buffer_size, nvidia::gxf::MemoryStorageType::kDevice);
  }

  const auto converted_tensor_ptr = resize_buffer_->pointer();
  if (converted_tensor_ptr == nullptr) {
    HOLOSCAN_LOG_ERROR("Failed to allocate memory for the resizing image");
    return nvidia::gxf::ExpectedOrCode(GXF_FAILURE, nullptr);
  }

  // Resize image
  NppStatus status = NPP_ERROR;
  const NppiSize src_size = {static_cast<int>(columns), static_cast<int>(rows)};
  const NppiRect src_roi = {src_img_rect[0], src_img_rect[1], src_img_rect[2], src_img_rect[3]};
  const NppiSize dst_size = {output_img_size[0], output_img_size[1]};
  const NppiRect dst_roi = {0, 0, output_img_size[0], output_img_size[1]};

  switch (channels) {
    case 3:
      switch (primitive_type) {
        case nvidia::gxf::PrimitiveType::kUnsigned8:
          status = nppiResize_8u_C3R_Ctx(static_cast<const Npp8u*>(in_tensor_data),
                                         columns * channels,
                                         src_size,
                                         src_roi,
                                         converted_tensor_ptr,
                                         output_img_size[0] * channels,
                                         dst_size,
                                         dst_roi,
                                         resize_mode_,
                                         npp_stream_ctx_);
          break;
        default:
          HOLOSCAN_LOG_ERROR("Unsupported input primitive type for resizing image");
          return nvidia::gxf::ExpectedOrCode(GXF_FAILURE, nullptr);
      }
      break;
    case 4:
      switch (primitive_type) {
        case nvidia::gxf::PrimitiveType::kUnsigned8:
          status = nppiResize_8u_C4R_Ctx(static_cast<const Npp8u*>(in_tensor_data),
                                         columns * channels,
                                         src_size,
                                         src_roi,
                                         converted_tensor_ptr,
                                         output_img_size[0] * channels,
                                         dst_size,
                                         dst_roi,
                                         resize_mode_,
                                         npp_stream_ctx_);
          break;
        default:
          HOLOSCAN_LOG_ERROR("Unsupported input primitive type for resizing image");
          return nvidia::gxf::ExpectedOrCode(GXF_FAILURE, nullptr);
      }
      break;
    default:
      HOLOSCAN_LOG_ERROR("Unsupported input primitive type for resizing image (%d, %d)",
                         channels,
                         static_cast<int32_t>(primitive_type));
      return nvidia::gxf::ExpectedOrCode(GXF_FAILURE, nullptr);
      break;
  }

  if (status != NPP_SUCCESS) {
    return nvidia::gxf::ExpectedOrCode(GXF_FAILURE, nullptr);
  }

  return nvidia::gxf::ExpectedOrCode(GXF_SUCCESS, converted_tensor_ptr);
}

// gxf_result_t FormatConverterOp::convertTensorFormat(const void* in_tensor_data, void*
// out_tensor_data,
void FormatConverterOp::convertTensorFormat(const void* in_tensor_data, void* out_tensor_data,
                                            const int32_t rows, const int32_t columns,
                                            const int16_t in_channels, const int16_t out_channels) {
  const uint32_t src_typesize = nvidia::gxf::PrimitiveTypeSize(in_primitive_type_);
  const uint32_t dst_typesize = nvidia::gxf::PrimitiveTypeSize(out_primitive_type_);

  const int32_t src_step = columns * in_channels * src_typesize;
  const int32_t dst_step = columns * out_channels * dst_typesize;

  const auto& out_channel_order = out_channel_order_.get();

  NppStatus status = NPP_ERROR;
  const NppiSize roi = {static_cast<int>(columns), static_cast<int>(rows)};

  switch (format_conversion_type_) {
    case FormatConversionType::kNone: {
      const auto in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      auto out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);

      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpyAsync(out_tensor_ptr,
                                                     in_tensor_ptr,
                                                     src_step * rows,
                                                     cudaMemcpyDeviceToDevice,
                                                     npp_stream_ctx_.hStream),
                                     "Failed to copy GPU data to GPU memory.");
      status = NPP_SUCCESS;
      break;
    }
    case FormatConversionType::kUnsigned8ToFloat32: {
      const auto in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      const auto out_tensor_ptr = static_cast<float*>(out_tensor_data);
      status = nppiScale_8u32f_C3R_Ctx(in_tensor_ptr,
                                       src_step,
                                       out_tensor_ptr,
                                       dst_step,
                                       roi,
                                       scale_min_,
                                       scale_max_,
                                       npp_stream_ctx_);
      break;
    }
    case FormatConversionType::kFloat32ToUnsigned8: {
      const auto in_tensor_ptr = static_cast<const float*>(in_tensor_data);
      const auto out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      status = nppiScale_32f8u_C3R_Ctx(in_tensor_ptr,
                                       src_step,
                                       out_tensor_ptr,
                                       dst_step,
                                       roi,
                                       scale_min_,
                                       scale_max_,
                                       npp_stream_ctx_);
      break;
    }
    case FormatConversionType::kRGB888ToRGBA8888: {
      const auto in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      const auto out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      // Convert RGB888 to RGBA8888 (3 channels -> 4 channels, uint8_t)
      int dst_order[4]{0, 1, 2, 3};
      if (!out_channel_order.empty()) {
        if (out_channel_order.size() != 4) {
          throw std::runtime_error("Invalid channel order for RGBA8888.");
        }
        for (int i = 0; i < 4; i++) { dst_order[i] = out_channel_order[i]; }
      }
      status = nppiSwapChannels_8u_C3C4R_Ctx(in_tensor_ptr,
                                             src_step,
                                             out_tensor_ptr,
                                             out_channels * dst_typesize * columns,
                                             roi,
                                             dst_order,
                                             alpha_value_.get(),
                                             npp_stream_ctx_);
      break;
    }
    case FormatConversionType::kRGBA8888ToRGB888: {
      const auto in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      const auto out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      // Convert RGBA8888 to RGB888 (4 channels -> 3 channels, uint8_t)
      int dst_order[3]{0, 1, 2};
      if (!out_channel_order.empty()) {
        if (out_channel_order.size() != 3) {
          throw std::runtime_error("Invalid channel order for RGB888.");
        }
        for (int i = 0; i < 3; i++) { dst_order[i] = out_channel_order[i]; }
      }
      status = nppiSwapChannels_8u_C4C3R_Ctx(in_tensor_ptr,
                                             src_step,
                                             out_tensor_ptr,
                                             out_channels * dst_typesize * columns,
                                             roi,
                                             dst_order,
                                             npp_stream_ctx_);
      break;
    }
    case FormatConversionType::kRGBA8888ToFloat32: {
      const auto in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      const auto out_tensor_ptr = static_cast<float*>(out_tensor_data);

      if (channel_buffer_->size() == 0) {
        auto frag = fragment();

        // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
        auto pool = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(frag->executor().context(),
                                                                        allocator_->gxf_cid());

        uint64_t buffer_size = rows * columns * 3;  // 4 channels -> 3 channels
        channel_buffer_->resize(pool.value(), buffer_size, nvidia::gxf::MemoryStorageType::kDevice);
      }

      const auto converted_tensor_ptr = channel_buffer_->pointer();
      if (converted_tensor_ptr == nullptr) {
        throw std::runtime_error("Failed to allocate memory for the channel conversion");
      }

      int dst_order[3]{0, 1, 2};
      if (!out_channel_order.empty()) {
        if (out_channel_order.size() != 3) {
          throw std::runtime_error("Invalid channel order for RGB888");
        }
        for (int i = 0; i < 3; i++) { dst_order[i] = out_channel_order[i]; }
      }

      status = nppiSwapChannels_8u_C4C3R_Ctx(in_tensor_ptr,
                                             src_step,
                                             converted_tensor_ptr,
                                             out_channels * src_typesize * columns,
                                             roi,
                                             dst_order,
                                             npp_stream_ctx_);

      if (status == NPP_SUCCESS) {
        const int32_t new_src_step = columns * out_channels * src_typesize;
        status = nppiScale_8u32f_C3R_Ctx(converted_tensor_ptr,
                                         new_src_step,
                                         out_tensor_ptr,
                                         dst_step,
                                         roi,
                                         scale_min_,
                                         scale_max_,
                                         npp_stream_ctx_);
      } else {
        throw std::runtime_error(fmt::format("Failed to convert channel order (NPP error code: {})",
                                             static_cast<int>(status)));
      }
      break;
    }
    case FormatConversionType::kRGB888ToYUV420: {
      nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420> color_format;
      auto color_planes = color_format.getDefaultColorPlanes(columns, rows);

      const auto in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);

      const auto out_y_ptr = static_cast<uint8_t*>(out_tensor_data);
      const auto out_u_ptr = out_y_ptr + color_planes[0].size;
      const auto out_v_ptr = out_u_ptr + color_planes[1].size;
      uint8_t* out_yuv_ptrs[3] = {out_y_ptr, out_u_ptr, out_v_ptr};

      const int32_t out_y_step = color_planes[0].stride;
      const int32_t out_u_step = color_planes[1].stride;
      const int32_t out_v_step = color_planes[2].stride;
      int32_t out_yuv_steps[3] = {out_y_step, out_u_step, out_v_step};

      status = nppiRGBToYUV420_8u_C3P3R_Ctx(
          in_tensor_ptr, src_step, out_yuv_ptrs, out_yuv_steps, roi, npp_stream_ctx_);
      if (status != NPP_SUCCESS) {
        throw std::runtime_error(fmt::format(
            "rgb888 to yuv420 conversion failed (NPP error code: {})", static_cast<int>(status)));
      }
      break;
    }
    case FormatConversionType::kYUV420ToRGBA8888: {
      nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420> color_format;
      auto color_planes = color_format.getDefaultColorPlanes(columns, rows);
      const auto in_y_ptr = static_cast<const uint8_t*>(in_tensor_data);
      const auto in_u_ptr = in_y_ptr + color_planes[0].size;
      const auto in_v_ptr = in_u_ptr + color_planes[1].size;
      const uint8_t* in_yuv_ptrs[3] = {in_y_ptr, in_u_ptr, in_v_ptr};

      const int32_t in_y_step = color_planes[0].stride;
      const int32_t in_u_step = color_planes[1].stride;
      const int32_t in_v_step = color_planes[2].stride;
      int32_t in_yuv_steps[3] = {in_y_step, in_u_step, in_v_step};

      const auto out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);

      status = nppiYUV420ToRGB_8u_P3AC4R_Ctx(
          in_yuv_ptrs, in_yuv_steps, out_tensor_ptr, dst_step, roi, npp_stream_ctx_);
      if (status != NPP_SUCCESS) {
        throw std::runtime_error(fmt::format(
            "yuv420 to rgba8888 conversion failed (NPP error code: {})", static_cast<int>(status)));
      }
      break;
    }
    case FormatConversionType::kYUV420ToRGB888: {
      nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420> color_format;
      auto color_planes = color_format.getDefaultColorPlanes(columns, rows);
      const auto in_y_ptr = static_cast<const uint8_t*>(in_tensor_data);
      const auto in_u_ptr = in_y_ptr + color_planes[0].size;
      const auto in_v_ptr = in_u_ptr + color_planes[1].size;
      const uint8_t* in_yuv_ptrs[3] = {in_y_ptr, in_u_ptr, in_v_ptr};

      const int32_t in_y_step = color_planes[0].stride;
      const int32_t in_u_step = color_planes[1].stride;
      const int32_t in_v_step = color_planes[2].stride;
      int32_t in_yuv_steps[3] = {in_y_step, in_u_step, in_v_step};

      const auto out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);

      status = nppiYUV420ToRGB_8u_P3C3R_Ctx(
          in_yuv_ptrs, in_yuv_steps, out_tensor_ptr, dst_step, roi, npp_stream_ctx_);
      if (status != NPP_SUCCESS) {
        throw std::runtime_error(fmt::format(
            "yuv420 to rgb888 conversion failed (NPP error code: {})", static_cast<int>(status)));
      }
      break;
    }
    case FormatConversionType::kNV12ToRGB888: {
      nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12> color_format;
      const auto in_y_ptr = static_cast<const uint8_t*>(in_tensor_data);
      auto color_planes = color_format.getDefaultColorPlanes(columns, rows);
      const auto in_uv_ptr = in_y_ptr + color_planes[0].size;
      const uint8_t* in_y_uv_ptrs[2] = {in_y_ptr, in_uv_ptr};

      const int32_t in_y_uv_step = color_planes[0].stride;

      const auto out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);

      status = nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(
          in_y_uv_ptrs, in_y_uv_step, out_tensor_ptr, dst_step, roi, npp_stream_ctx_);
      if (status != NPP_SUCCESS) {
        throw std::runtime_error(fmt::format(
            "NV12 to rgb888 conversion failed (NPP error code: {})", static_cast<int>(status)));
      }
      break;
    }
    default:
      throw std::runtime_error(fmt::format("Unsupported format conversion: {} ({}) -> {}\n",
                                           in_dtype_str_.get(),
                                           static_cast<uint32_t>(in_dtype_),
                                           out_dtype_str_.get()));
  }

  // Reorder channels in the output tensor (inplace) if needed.
  switch (format_conversion_type_) {
    case FormatConversionType::kNone:
    case FormatConversionType::kUnsigned8ToFloat32:
    case FormatConversionType::kFloat32ToUnsigned8: {
      if (!out_channel_order.empty()) {
        switch (out_channels) {
          case 3: {
            int dst_order[3]{0, 1, 2};
            if (out_channel_order.size() != 3) {
              throw std::runtime_error(
                  fmt::format("Invalid channel order for {}", out_dtype_str_.get()));
            }
            for (int i = 0; i < 3; i++) { dst_order[i] = out_channel_order[i]; }
            switch (out_primitive_type_) {
              case nvidia::gxf::PrimitiveType::kUnsigned8: {
                auto out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
                status = nppiSwapChannels_8u_C3IR_Ctx(
                    out_tensor_ptr, dst_step, roi, dst_order, npp_stream_ctx_);
                break;
              }
              case nvidia::gxf::PrimitiveType::kFloat32: {
                auto out_tensor_ptr = static_cast<float*>(out_tensor_data);
                status = nppiSwapChannels_32f_C3IR_Ctx(
                    out_tensor_ptr, dst_step, roi, dst_order, npp_stream_ctx_);
                break;
              }
              default:
                throw std::runtime_error(
                    fmt::format("Unsupported output data type for reordering channels: {}",
                                out_dtype_str_.get()));
            }
            break;
          }
          case 4: {
            int dst_order[4]{0, 1, 2, 3};
            if (out_channel_order.size() != 4) {
              throw std::runtime_error(
                  fmt::format("Invalid channel order for {}", out_dtype_str_.get()));
            }
            for (int i = 0; i < 4; i++) { dst_order[i] = out_channel_order[i]; }
            switch (out_primitive_type_) {
              case nvidia::gxf::PrimitiveType::kUnsigned8: {
                auto out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
                status = nppiSwapChannels_8u_C4IR_Ctx(
                    out_tensor_ptr, dst_step, roi, dst_order, npp_stream_ctx_);
                break;
              }
              case nvidia::gxf::PrimitiveType::kFloat32: {
                auto out_tensor_ptr = static_cast<float*>(out_tensor_data);
                status = nppiSwapChannels_32f_C4IR_Ctx(
                    out_tensor_ptr, dst_step, roi, dst_order, npp_stream_ctx_);
                break;
              }
              default:
                throw std::runtime_error(
                    fmt::format("Unsupported output data type for reordering channels: {}",
                                out_dtype_str_.get()));
            }
            break;
          }
        }
        if (status != NPP_SUCCESS) {
          throw std::runtime_error("Failed to convert channel order");
        }
      }
    }
    default:
      break;
  }
}

void FormatConverterOp::setup(OperatorSpec& spec) {
  auto& in_tensor = spec.input<gxf::Entity>("source_video");
  auto& out_tensor = spec.output<gxf::Entity>("tensor");

  spec.param(in_, "in", "Input", "Input channel.", &in_tensor);
  spec.param(out_, "out", "Output", "Output channel.", &out_tensor);

  spec.param(in_tensor_name_,
             "in_tensor_name",
             "InputTensorName",
             "Name of the input tensor.",
             std::string(""));
  spec.param(in_dtype_str_, "in_dtype", "InputDataType", "Source data type.", std::string(""));
  spec.param(out_tensor_name_,
             "out_tensor_name",
             "OutputTensorName",
             "Name of the output tensor.",
             std::string(""));
  spec.param(out_dtype_str_, "out_dtype", "OutputDataType", "Destination data type.");
  spec.param(scale_min_, "scale_min", "Scale min", "Minimum value of the scale.", 0.f);
  spec.param(scale_max_, "scale_max", "Scale max", "Maximum value of the scale.", 1.f);
  spec.param(alpha_value_,
             "alpha_value",
             "Alpha value",
             "Alpha value that can be used to fill the alpha channel when "
             "converting RGB888 to RGBA8888.",
             static_cast<uint8_t>(255));

  spec.param(src_roi_rect_,
             "src_roi_rect",
             "src image roi rect",
             "Src image roi rect [x, y, width, height]",
             std::vector<int32_t>{0, 0, 1920, 1080});

  spec.param(output_img_size_,
             "output_img_size",
             "Output image size after resize",
             "Output image size [ width, height ] after resize",
             std::vector<int32_t>({0, 0}));

  spec.param(resize_mode_,
             "resize_mode",
             "Resize mode",
             "Mode for resize. 4 (NPPI_INTER_CUBIC) if this value is zero.",
             0);
  spec.param(out_channel_order_,
             "out_channel_order",
             "Output channel order",
             "Host memory integer array describing how channel values are permutated.",
             std::vector<int>{});

  spec.param(allocator_, "allocator", "Allocator", "Output Allocator");

  cuda_stream_handler_.define_params(spec);

  // TODO (gbae): spec object holds an information about errors
  // TODO (gbae): incorporate std::expected to not throw exceptions
}

}  // namespace holoscan::ops::orsi
