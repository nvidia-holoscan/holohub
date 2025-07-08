/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nv_video_decoder.hpp"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <cuda.h>
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/gxf/entity.hpp"

#include "gxf/core/entity.hpp"    // nvidia::gxf::Entity::Shared
#include "gxf/std/allocator.hpp"  // nvidia::gxf::Allocator, nvidia::gxf::MemoryStorageType
#include "gxf/std/tensor.hpp"     // nvidia::gxf::Tensor etc.
#include "gxf/std/timestamp.hpp"  // nvidia::gxf::Timestamp

#include "../common/utils.h"

namespace holoscan::ops {
void NvVideoDecoderOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("input");

  spec.output<holoscan::gxf::Entity>("output");

  spec.param(cuda_device_ordinal_,
             "cuda_device_ordinal",
             "CudaDeviceOrdinal",
             "Device to use for CUDA operations",
             ParameterFlag::kOptional);
  spec.param(allocator_, "allocator", "Allocator", "Allocator for output buffers.");
  spec.param(verbose_, "verbose", "Verbose", "Print detailed decoder information", false);

  cuda_stream_handler_.define_params(spec);
}

void NvVideoDecoderOp::initialize() {
  Operator::initialize();

  // Initialize CUDA
  CudaCheck(cuInit(0));

  // Get the CUDA device
  CUdevice cu_device;
  CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal_.get()));
  cu_device_ = cu_device;

  // Retain the primary context for the device
  CudaCheck(cuDevicePrimaryCtxRetain(&cu_context_, cu_device_));

  // Initialize NVIDIA decoder with CUDA context
  file_data_provider_ = std::make_unique<StreamDataProvider>();
}

void NvVideoDecoderOp::compute(InputContext& op_input, OutputContext& op_output,
                               ExecutionContext& context) {
  auto enter_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now().time_since_epoch())
                             .count();

  // Get input tensor
  auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
  if (!maybe_entity) {
    throw std::runtime_error("Failed to receive input entity");
  }

  // Get the tensor from the input message
  auto tensor = maybe_entity.value().get<Tensor>("");
  if (!tensor) {
    throw std::runtime_error("Failed to get tensor from input message");
  }

  auto data_size = tensor->size();
  auto data_ptr = tensor->data();

  file_data_provider_->SetData(static_cast<uint8_t*>(data_ptr), data_size);

  if (demuxer_ == nullptr) {
    // Set the current context
    CudaCheck(cuCtxPushCurrent(cu_context_));

    try {
      demuxer_ = std::make_unique<FFmpegDemuxer>(file_data_provider_.get());
      decoder_ = std::make_unique<NvDecoder>(cu_context_,
                                             true,
                                             FFmpeg2NvCodecId(demuxer_->GetVideoCodec()),
                                             true,
                                             false,
                                             nullptr,
                                             nullptr,
                                             false,
                                             0,
                                             0,
                                             1000,
                                             true);
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Failed to initialize decoder: {}", e.what());
      // Reset the demuxer and decoder for potential retry
      demuxer_.reset();
      decoder_.reset();
      // Pop the context to avoid context stack issues
      CudaCheck(cuCtxPopCurrent(nullptr));
      throw std::runtime_error(
          "Decoder initialization failed. Please check video format and codec support.");
    }
  }

  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

  auto output = nvidia::gxf::Entity::New(context.context());
  if (!output) {
    throw std::runtime_error("Failed to allocate message for output");
  }

  auto maybe_video_buffer = output.value().add<nvidia::gxf::VideoBuffer>();
  if (!maybe_video_buffer) {
    throw std::runtime_error("Failed to allocate video buffer");
  }

  auto video_buffer = maybe_video_buffer.value();
  nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12> video_type;
  nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12> color_format;

  int nFrame = 0;
  uint8_t* pVideo = NULL;
  int nVideoBytes = 0;
  uint8_t* pFrame;
  int nFrameReturned = 0;

  do {
    demuxer_->Demux(&pVideo, &nVideoBytes);

    try {
      nFrameReturned = decoder_->Decode(pVideo, nVideoBytes);
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Decode failed: {}", e.what());
      // Skip this frame and continue
      continue;
    }

    if (nFrameReturned == 0) {
      // emit_empty_frame(op_output, context);
      continue;
    }

    nFrame += nFrameReturned;

    // HOLOSCAN_LOG_INFO("Decoded {} frames, total {} frames", nFrameReturned, nFrame);
    if (nFrameReturned == 0) {
      continue;
    }
    if (nFrameReturned > 1) {
      HOLOSCAN_LOG_WARN("More than one frame returned from decoder: {}. Processing all frames.",
                        nFrameReturned);
    }

    for (int i = 0; i < nFrameReturned; i++) {
      pFrame = decoder_->GetLockedFrame();

      auto width = decoder_->GetWidth();
      auto height = decoder_->GetHeight();
      auto color_planes = color_format.getDefaultColorPlanes(width, height, false);
      nvidia::gxf::VideoBufferInfo video_buffer_info{
          static_cast<uint32_t>(width),
          static_cast<uint32_t>(height),
          video_type.value,
          std::move(color_planes),
          nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
      video_buffer_info.color_planes[0].offset = 0;
      video_buffer_info.color_planes[1].offset = decoder_->GetLumaPlaneSize();

      auto result = video_buffer->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12>(
          static_cast<uint32_t>(width),
          static_cast<uint32_t>(height),
          nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
          nvidia::gxf::MemoryStorageType::kDevice,
          allocator.value(),
          false);

      if (!result) {
        throw std::runtime_error("Failed to resize video buffer");
      }
      size_t src_pitch = decoder_->GetDeviceFramePitch();
      size_t dst_pitch = video_buffer_info.color_planes[0].stride;

      // Log video buffer and decoder info for debugging
      if (verbose_.get()) {
        HOLOSCAN_LOG_INFO("---- Video Buffer & Decoder Debug Info ----");
        HOLOSCAN_LOG_INFO("video_buffer_info.color_planes[0].stride (Y): {}",
                          video_buffer_info.color_planes[0].stride);
        HOLOSCAN_LOG_INFO("video_buffer_info.color_planes[1].stride (UV): {}",
                          video_buffer_info.color_planes[1].stride);
        HOLOSCAN_LOG_INFO("video_buffer_info.color_planes[0].offset (Y): {}",
                          video_buffer_info.color_planes[0].offset);
        HOLOSCAN_LOG_INFO("video_buffer_info.color_planes[1].offset (UV): {}",
                          video_buffer_info.color_planes[1].offset);
        HOLOSCAN_LOG_INFO("decoder_->GetVideoInfo(): {}", decoder_->GetVideoInfo());
        HOLOSCAN_LOG_INFO("decoder_->GetWidth(): {}, decoder_->GetHeight(): {}",
                          decoder_->GetWidth(),
                          decoder_->GetHeight());
        HOLOSCAN_LOG_INFO("decoder_->GetDecodeWidth(): {}", decoder_->GetDecodeWidth());
        HOLOSCAN_LOG_INFO("decoder_->GetFrameSize(): {}", decoder_->GetFrameSize());
        HOLOSCAN_LOG_INFO("decoder_->GetLumaPlaneSize(): {}", decoder_->GetLumaPlaneSize());
        HOLOSCAN_LOG_INFO("decoder_->GetChromaPlaneSize(): {}", decoder_->GetChromaPlaneSize());
        HOLOSCAN_LOG_INFO("decoder_->GetOutputFormat(): {}",
                          static_cast<int>(decoder_->GetOutputFormat()));
        HOLOSCAN_LOG_INFO("decoder_->GetDeviceFramePitch(): {}",
                          static_cast<int>(decoder_->GetDeviceFramePitch()));
        HOLOSCAN_LOG_INFO("decoder_->GetChromaHeight(): {}",
                          static_cast<int>(decoder_->GetChromaHeight()));
        HOLOSCAN_LOG_INFO("decoder_->GetNumChromaPlanes(): {}",
                          static_cast<int>(decoder_->GetNumChromaPlanes()));
        HOLOSCAN_LOG_INFO("decoder_->GetBitDepth(): {}", static_cast<int>(decoder_->GetBitDepth()));
        HOLOSCAN_LOG_INFO("decoder_->GetBPP(): {}", static_cast<int>(decoder_->GetBPP()));
        HOLOSCAN_LOG_INFO("pFrame address: {}", static_cast<void*>(pFrame));
        HOLOSCAN_LOG_INFO("video_buffer->pointer() address: {}",
                          static_cast<void*>(video_buffer->pointer()));
        HOLOSCAN_LOG_INFO("------------------------------------------");
      }

      CUDA_TRY(cudaMemcpy2D(video_buffer->pointer() + video_buffer_info.color_planes[0].offset,
                            video_buffer_info.color_planes[0].stride,
                            pFrame,
                            decoder_->GetDeviceFramePitch(),
                            width,  // width in bytes for Y plane
                            height,
                            cudaMemcpyDeviceToDevice));

      CUDA_TRY(cudaMemcpy2D(video_buffer->pointer() + video_buffer_info.color_planes[1].offset,
                            video_buffer_info.color_planes[1].stride,
                            pFrame + decoder_->GetLumaPlaneSize(),
                            decoder_->GetDeviceFramePitch(),
                            width,
                            height / 2,
                            cudaMemcpyDeviceToDevice));

      // After copying Y plane
      size_t pad = video_buffer_info.color_planes[0].stride - width;
      if (pad > 0) {
        HOLOSCAN_LOG_INFO("Padding Y plane with {} bytes", pad);
        for (int y = 0; y < height; ++y) {
          uint8_t* row_start = video_buffer->pointer() + video_buffer_info.color_planes[0].offset +
                               y * video_buffer_info.color_planes[0].stride;
          CUDA_TRY(cudaMemset(row_start + width, 0, pad));
        }
      }

      // After copying UV plane
      pad = video_buffer_info.color_planes[1].stride - width;
      if (pad > 0) {
        HOLOSCAN_LOG_INFO("Padding UV plane with {} bytes", pad);
        for (int y = 0; y < height / 2; ++y) {
          uint8_t* row_start = video_buffer->pointer() + video_buffer_info.color_planes[1].offset +
                               y * video_buffer_info.color_planes[1].stride;
          CUDA_TRY(cudaMemset(row_start + width, 0, pad));
        }
      }

      decoder_->UnlockFrame(&pFrame);
    }
  } while (nVideoBytes);

  if (nFrame > 0) {
    auto emit_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();
    auto meta = metadata();
    auto decode_latency_ms = (emit_timestamp - enter_timestamp) / 1000000.0;
    meta->set("video_decoder_decode_latency_ms"s, decode_latency_ms);
    meta->set("jitter_time"s, emit_timestamp - last_emit_timestamp_);
    meta->set("fps"s,
              last_emit_timestamp_ == 0
                  ? 0
                  : static_cast<uint64_t>(1e9 / (emit_timestamp - last_emit_timestamp_)));
    auto result = gxf::Entity(std::move(output.value()));
    op_output.emit(result, "output");
    last_emit_timestamp_ = emit_timestamp;
  }
}

void NvVideoDecoderOp::stop() {
  // Cleanup resources in reverse order of creation
  // Release the primary context for the device if it was created by this operator
  if (cu_context_) {
    // Ensure the context is not active before releasing it
    CUcontext current_ctx;
    CUresult result = cuCtxGetCurrent(&current_ctx);
    if (result == CUDA_SUCCESS && current_ctx == cu_context_) {
      CudaCheck(cuCtxPopCurrent(nullptr));
    }

    CudaCheck(cuDevicePrimaryCtxRelease(cu_device_));
    cu_context_ = nullptr;
  }
}
}  // namespace holoscan::ops
