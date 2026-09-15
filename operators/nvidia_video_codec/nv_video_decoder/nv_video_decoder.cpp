/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <cuda.h>
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/resources/gxf/receiver.hpp"

#include "gxf/core/entity.hpp"    // nvidia::gxf::Entity::Shared
#include "gxf/std/allocator.hpp"  // nvidia::gxf::Allocator, nvidia::gxf::MemoryStorageType
#include "gxf/std/tensor.hpp"     // nvidia::gxf::Tensor etc.
#include "gxf/std/timestamp.hpp"  // nvidia::gxf::Timestamp

#include "../common/utils.h"

namespace holoscan::ops {

namespace {

class InputOrPendingCondition : public Condition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS(InputOrPendingCondition)
  InputOrPendingCondition() = default;

  void initialize() override {
    Condition::initialize();
    if (pending_frame_count_ == nullptr) {
      throw std::runtime_error("InputOrPendingCondition has no pending-frame counter");
    }
  }

  void setup(ComponentSpec& spec) override {
    spec.param(receiver_,
               "receiver",
               "Receiver",
               "Input receiver checked together with the decoder's pending-frame queue.");
  }

  void check([[maybe_unused]] int64_t timestamp, SchedulingStatusType* status_type,
             int64_t* target_timestamp) const override {
    if (status_type == nullptr || target_timestamp == nullptr) {
      throw std::runtime_error("InputOrPendingCondition received a null output argument");
    }
    *status_type = current_state_;
    *target_timestamp = last_state_change_;
  }

  void on_execute(int64_t timestamp) override { update_state(timestamp); }

  void update_state(int64_t timestamp) override {
    const auto receiver = receiver_.get();
    if (receiver == nullptr) {
      throw std::runtime_error("InputOrPendingCondition has no input receiver");
    }
    const bool input_available = receiver->back_size() + receiver->size() > 0;
    const bool frame_pending = pending_frame_count_->load(std::memory_order_acquire) > 0;
    const auto next_state = input_available || frame_pending ? SchedulingStatusType::kReady
                                                             : SchedulingStatusType::kWait;
    if (next_state != current_state_) {
      current_state_ = next_state;
      last_state_change_ = timestamp;
    }
  }

  void pending_frame_count(const std::atomic<std::size_t>* pending_frame_count) {
    pending_frame_count_ = pending_frame_count;
  }

 private:
  Parameter<std::shared_ptr<Receiver>> receiver_;
  const std::atomic<std::size_t>* pending_frame_count_ = nullptr;
  SchedulingStatusType current_state_ = SchedulingStatusType::kWait;
  int64_t last_state_change_ = 0;
};

}  // namespace

void NvVideoDecoderOp::setup(OperatorSpec& spec) {
  // InputOrPendingCondition below replaces the default input-only condition so
  // queued decoded frames can be emitted after the final input has been consumed.
  spec.input<holoscan::gxf::Entity>("input").condition(ConditionType::kNone);

  spec.output<holoscan::gxf::Entity>("output");

  spec.param(cuda_device_ordinal_,
             "cuda_device_ordinal",
             "CudaDeviceOrdinal",
             "Device to use for CUDA operations",
             ParameterFlag::kOptional);
  spec.param(allocator_, "allocator", "Allocator", "Allocator for output buffers.");
  spec.param(verbose_, "verbose", "Verbose", "Print detailed decoder information", false);
  spec.param(codec_,
             "codec",
             "Codec",
             "Optional codec for packetized Annex-B input. Set H264 or HEVC to bypass "
             "the FFmpeg demuxer and feed each input tensor directly to NVDEC. "
             "Packetized input tensors must use host-accessible kHost or kSystem storage. "
             "Only bitstreams decoded to 8-bit 4:2:0 NV12 are supported. Set the "
             "end_of_stream metadata field on the final tensor to drain delayed frames.",
             std::string(""));
  spec.param(packetized_input_mode_,
             "packetized_input_mode",
             "PacketizedInputMode",
             "Framing of packetized codec input: 'stream' for arbitrary byte-stream chunks, "
             "or 'access_unit' when every input tensor contains exactly one complete encoded "
             "access unit. 'access_unit' enables CUVID_PKT_ENDOFPICTURE.",
             std::string("stream"));
  spec.param(packetized_low_latency_,
             "packetized_low_latency",
             "PacketizedLowLatency",
             "Decoder display policy for packetized input. False preserves normal display "
             "reordering, including B-frames. True reduces display delay and requires a "
             "bitstream without B-frames.",
             false);

  cuda_stream_handler_.define_params(spec);
}

void NvVideoDecoderOp::initialize() {
  if (input_or_pending_condition_ == nullptr) {
    auto input_or_pending_condition = fragment()->make_condition<InputOrPendingCondition>(
        name() + "_input_or_pending", Arg("receiver", "input"));
    input_or_pending_condition->pending_frame_count(&pending_frame_count_);
    input_or_pending_condition_ = input_or_pending_condition;
    add_arg(input_or_pending_condition_);
  }

  Operator::initialize();

  const std::string& packetized_input_mode = packetized_input_mode_.get();
  if (packetized_input_mode != "stream" && packetized_input_mode != "access_unit") {
    throw std::invalid_argument("Unsupported packetized_input_mode: " + packetized_input_mode +
                                ". Expected 'stream' or 'access_unit'.");
  }

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
  // Preserve decoder output ordering and respect the downstream-affordability
  // condition by emitting at most one queued frame per operator execution.
  if (!pending_frames_.empty()) {
    emit_pending_frame(op_output, context);
    return;
  }

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

  if (verbose_.get()) {
    HOLOSCAN_LOG_INFO("Tensor received: {} bytes", data_size);
  }

  auto meta = metadata();
  bool is_from_reader = (meta->get<std::string>("source", "") == "nv_video_reader");
  bool is_packetized_stream = !codec_.get().empty();
  bool direct_packet_decode = is_from_reader || is_packetized_stream;
  const bool packetized_end_of_stream =
      is_packetized_stream && meta->get<bool>("end_of_stream", false);

  if (is_packetized_stream) {
    auto gxf_tensor = static_cast<nvidia::gxf::Entity&>(maybe_entity.value())
                          .get<nvidia::gxf::Tensor>("");
    if (!gxf_tensor) {
      throw std::runtime_error("Failed to get GXF tensor from packetized input message");
    }
    const auto storage_type = gxf_tensor.value()->storage_type();
    if (storage_type != nvidia::gxf::MemoryStorageType::kHost &&
        storage_type != nvidia::gxf::MemoryStorageType::kSystem) {
      throw std::runtime_error(
          "Packetized input tensors must use host-accessible kHost or kSystem storage");
    }
  }

  // Handle stream reset signal for looping videos or a discontinuity in a
  // packetized stream. Packetized input recreates NVDEC so a codec/stream
  // change cannot retain references to the previous GOP.
  bool stream_reset = meta->get<bool>("stream_reset", false);
  if (direct_packet_decode && stream_reset && decoder_ != nullptr) {
    if (verbose_.get()) {
      HOLOSCAN_LOG_INFO("Stream reset detected - flushing decoder");
    }
    try {
      decoder_->Decode(nullptr, 0);
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_WARN("Failed to flush decoder on stream reset: {}", e.what());
    }
    if (is_packetized_stream) {
      decoder_.reset();
      // Balance the context pushed when this decoder was initialized before
      // init_decoder_for_packetized_stream() pushes it again.
      CudaCheck(cuCtxPopCurrent(nullptr));
    }
  }

  // When codec is set, the FFmpeg demuxer is intentionally bypassed and each
  // input tensor is fed directly to CUVID. packetized_input_mode determines
  // whether those tensors are arbitrary byte-stream chunks or complete access units.
  if (is_packetized_stream) {
    init_decoder_for_packetized_stream();
  } else if (is_from_reader) {
    init_decoder_for_file(meta);
  } else {
    init_decoder_for_streaming(data_ptr, data_size);
  }

  uint8_t* pVideo = NULL;
  int nVideoBytes = 0;
  int nFrameReturned = 0;

  if (direct_packet_decode) {
    // The input tensor is already encoded data ready for direct parser submission.
    pVideo = static_cast<uint8_t*>(data_ptr);
    nVideoBytes = data_size;

    try {
      // CUVID_PKT_ENDOFPICTURE is only correct when the caller guarantees that
      // each packetized input tensor contains exactly one complete access unit.
      // In stream mode, leave picture-boundary detection entirely to CUVID.
      const bool complete_access_unit =
          is_packetized_stream && packetized_input_mode_.get() == "access_unit";
      const uint32_t decode_flags = complete_access_unit ? CUVID_PKT_ENDOFPICTURE : 0;
      if (is_packetized_stream) {
        nFrameReturned = decoder_->Decode(pVideo, nVideoBytes, decode_flags);
      } else {
        nFrameReturned = decoder_->Decode(pVideo, nVideoBytes);
      }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Failed to decode packetized frame: {}", e.what());
      return;
    }
  } else {
    // Generic byte-stream data - use FFmpeg demuxer and loop until we get frames.
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
        continue;
      }
      break;  // Exit loop after successful decode
    } while (nVideoBytes);
  }

  // A zero return simply means the current parser submission produced no displayable
  // frame. Retain the legacy buffered-frame probe only for file/generic streaming.
  if (nFrameReturned == 0 && decoder_ != nullptr && !is_packetized_stream) {
    uint8_t* test_frame = decoder_->GetLockedFrame();
    if (test_frame != nullptr) {
      decoder_->UnlockFrame(&test_frame);
      nFrameReturned = 1;
      if (verbose_.get()) {
        HOLOSCAN_LOG_INFO("Retrieved buffered frame from decoder");
      }
    }
  }

  struct LockedFrames {
    explicit LockedFrames(NvDecoder* decoder) : decoder(decoder) {}
    ~LockedFrames() {
      for (auto& frame : frames) {
        if (frame != nullptr) {
          decoder->UnlockFrame(&frame);
        }
      }
    }

    NvDecoder* decoder;
    std::vector<uint8_t*> frames;
  } locked_frames(decoder_.get());

  auto collect_decoded_frames = [&](int frame_count) {
    if (frame_count > 0 && is_packetized_stream &&
        (decoder_->GetOutputFormat() != cudaVideoSurfaceFormat_NV12 ||
         decoder_->GetBitDepth() != 8 ||
         decoder_->GetOutputChromaFormat() != cudaVideoChromaFormat_420)) {
      throw std::runtime_error(
          "Unsupported packetized decoder output format; only 8-bit 4:2:0 NV12 is supported");
    }

    if (frame_count > 1 && verbose_.get()) {
      HOLOSCAN_LOG_INFO("Decoder returned {} frames. Queueing all frames.", frame_count);
    }

    for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
      uint8_t* frame = decoder_->GetLockedFrame();
      if (frame == nullptr) {
        throw std::runtime_error("Failed to get a decoded frame reported by NvDecoder");
      }
      locked_frames.frames.push_back(frame);
    }
  };

  collect_decoded_frames(nFrameReturned);

  if (packetized_end_of_stream) {
    if (verbose_.get()) {
      HOLOSCAN_LOG_INFO("Packetized end of stream received - draining decoder");
    }
    int flushed_frame_count = 0;
    try {
      flushed_frame_count = decoder_->Decode(nullptr, 0);
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Failed to drain packetized decoder at end of stream: {}", e.what());
      return;
    }
    collect_decoded_frames(flushed_frame_count);
  }

  if (locked_frames.frames.empty()) {
    if (verbose_.get()) {
      HOLOSCAN_LOG_INFO(
          "No frames decoded - this is normal for initialization/header packets");
    }
    return;
  }

  for (auto& frame : locked_frames.frames) {
    pending_frames_.push_back(PendingFrame{frame, *meta, enter_timestamp});
    frame = nullptr;
  }
  pending_frame_count_.store(pending_frames_.size(), std::memory_order_release);

  emit_pending_frame(op_output, context);
}

void NvVideoDecoderOp::emit_pending_frame(OutputContext& op_output, ExecutionContext& context) {
  if (pending_frames_.empty()) {
    throw std::runtime_error("No decoded frame is pending for emission");
  }

  auto& pending_frame = pending_frames_.front();
  uint8_t* pFrame = pending_frame.data;
  auto meta = metadata();
  *meta = pending_frame.metadata;

  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

  nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12> video_type;
  nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12> color_format;

  auto output = nvidia::gxf::Entity::New(context.context());
  if (!output) {
    throw std::runtime_error("Failed to allocate message for output");
  }

  auto maybe_video_buffer = output.value().add<nvidia::gxf::VideoBuffer>();
  if (!maybe_video_buffer) {
    throw std::runtime_error("Failed to allocate video buffer");
  }
  auto video_buffer = maybe_video_buffer.value();

  auto width = decoder_->GetWidth();
  auto height = decoder_->GetHeight();
  auto color_planes = color_format.getDefaultColorPlanes(width, height, true);
  nvidia::gxf::VideoBufferInfo video_buffer_info{
      static_cast<uint32_t>(width),
      static_cast<uint32_t>(height),
      video_type.value,
      std::move(color_planes),
      nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
  video_buffer_info.color_planes[0].offset = 0;
  // When stride=true, use the actual size of the Y plane (which includes padding)
  // instead of decoder's luma plane size
  video_buffer_info.color_planes[1].offset = video_buffer_info.color_planes[0].size;

  auto result = video_buffer->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12>(
      static_cast<uint32_t>(width),
      static_cast<uint32_t>(height),
      nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
      nvidia::gxf::MemoryStorageType::kDevice,
      allocator.value(),
      true);

  if (!result) {
    throw std::runtime_error("Failed to resize video buffer");
  }

  // Log video buffer and decoder info for debugging
  if (verbose_.get()) {
    HOLOSCAN_LOG_INFO("---- Video Buffer & Decoder Debug Info ----");
    HOLOSCAN_LOG_INFO("Processing frame: width={}, height={}", width, height);
    HOLOSCAN_LOG_INFO("video_buffer_info.color_planes[0].stride (Y): {}",
                      video_buffer_info.color_planes[0].stride);
    HOLOSCAN_LOG_INFO("video_buffer_info.color_planes[0].size (Y): {}",
                      video_buffer_info.color_planes[0].size);
    HOLOSCAN_LOG_INFO("video_buffer_info.color_planes[1].stride (UV): {}",
                      video_buffer_info.color_planes[1].stride);
    HOLOSCAN_LOG_INFO("video_buffer_info.color_planes[1].offset (UV): {}",
                      video_buffer_info.color_planes[1].offset);
    HOLOSCAN_LOG_INFO("decoder_->GetDeviceFramePitch(): {}",
                      static_cast<int>(decoder_->GetDeviceFramePitch()));
    HOLOSCAN_LOG_INFO("decoder_->GetLumaPlaneSize(): {}",
                      static_cast<int>(decoder_->GetLumaPlaneSize()));
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
    if (verbose_.get()) {
      HOLOSCAN_LOG_INFO("Padding Y plane with {} bytes", pad);
    }
    for (int y = 0; y < height; ++y) {
      uint8_t* row_start = video_buffer->pointer() + video_buffer_info.color_planes[0].offset +
                           y * video_buffer_info.color_planes[0].stride;
      CUDA_TRY(cudaMemset(row_start + width, 0, pad));
    }
  }

  // After copying UV plane
  pad = video_buffer_info.color_planes[1].stride - width;
  if (pad > 0) {
    if (verbose_.get()) {
      HOLOSCAN_LOG_INFO("Padding UV plane with {} bytes", pad);
    }
    for (int y = 0; y < height / 2; ++y) {
      uint8_t* row_start = video_buffer->pointer() + video_buffer_info.color_planes[1].offset +
                           y * video_buffer_info.color_planes[1].stride;
      CUDA_TRY(cudaMemset(row_start + width, 0, pad));
    }
  }

  decoder_->UnlockFrame(&pFrame);
  pending_frame.data = nullptr;

  auto emit_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
  auto decode_latency_ms =
      (emit_timestamp - pending_frame.decode_start_timestamp) / 1000000.0;
  // Replace metrics from the queued frame's metadata snapshot without changing
  // the policy for unrelated metadata keys.
  meta->erase("video_decoder_decode_latency_ms"s);
  meta->erase("jitter_time"s);
  meta->erase("fps"s);
  meta->set("video_decoder_decode_latency_ms"s, decode_latency_ms);
  meta->set("jitter_time"s, (emit_timestamp - last_emit_timestamp_) / 1000000.0);
  meta->set("fps"s,
            last_emit_timestamp_ == 0
                ? 0
                : static_cast<uint64_t>(1e9 / (emit_timestamp - last_emit_timestamp_)));

  auto output_result = gxf::Entity(std::move(output.value()));
  op_output.emit(output_result, "output");
  last_emit_timestamp_ = emit_timestamp;

  pending_frames_.pop_front();
  pending_frame_count_.store(pending_frames_.size(), std::memory_order_release);
}

void NvVideoDecoderOp::release_pending_frames() {
  if (decoder_ != nullptr) {
    for (auto& pending_frame : pending_frames_) {
      if (pending_frame.data != nullptr) {
        decoder_->UnlockFrame(&pending_frame.data);
      }
    }
  }
  pending_frames_.clear();
  pending_frame_count_.store(0, std::memory_order_release);
}

void NvVideoDecoderOp::init_decoder_for_streaming(void* data, size_t size) {
  file_data_provider_->SetData(static_cast<uint8_t*>(data), size);

  if (verbose_.get()) {
    HOLOSCAN_LOG_INFO("StreamDataProvider buffer size: {} bytes, offset: {}",
                      file_data_provider_->GetBufferSize(),
                      file_data_provider_->GetOffset());
  }

  if (demuxer_ == nullptr || decoder_ == nullptr) {
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
}

void NvVideoDecoderOp::init_decoder_for_file(std::shared_ptr<MetadataDictionary> meta) {
  if (decoder_ == nullptr) {
    CudaCheck(cuCtxPushCurrent(cu_context_));
    try {
      cudaVideoCodec codec = FFmpeg2NvCodecId(meta->get<AVCodecID>("codec", AV_CODEC_ID_H264));
      decoder_ =
          std::make_unique<NvDecoder>(cu_context_,
                                      true,     // bUseDeviceFrame
                                      codec,    // eCodec
                                      false,    // bLowLatency - disable for proper frame order
                                      false,    // bDeviceFramePitched
                                      nullptr,  // pCropRect
                                      nullptr,  // pResizeDim
                                      false,    // extract_user_SEI_Message
                                      0,        // maxWidth
                                      0,        // maxHeight
                                      1000,     // clkRate
                                      false);   // force_zero_latency - allow reordering
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Failed to initialize decoder for nv_video_reader: {}", e.what());
      decoder_.reset();
      CudaCheck(cuCtxPopCurrent(nullptr));
      throw std::runtime_error("Decoder initialization failed for nv_video_reader data.");
    }
  }
}

void NvVideoDecoderOp::init_decoder_for_packetized_stream() {
  if (decoder_ != nullptr) {
    return;
  }

  CudaCheck(cuCtxPushCurrent(cu_context_));
  try {
    const std::string codec_name = codec_.get();
    const std::string input_mode = packetized_input_mode_.get();
    const bool low_latency = packetized_low_latency_.get();
    cudaVideoCodec codec;
    if (codec_name == "H264" || codec_name == "h264") {
      codec = cudaVideoCodec_H264;
    } else if (codec_name == "HEVC" || codec_name == "hevc" || codec_name == "H265" ||
               codec_name == "h265") {
      codec = cudaVideoCodec_HEVC;
    } else {
      throw std::runtime_error("Unsupported packetized codec: " + codec_name);
    }

    // Framing and display latency are independent: packetized_input_mode controls
    // per-submission flags, while packetized_low_latency controls CUVID reordering.
    decoder_ = std::make_unique<NvDecoder>(cu_context_,
                                           true,   // bUseDeviceFrame
                                           codec,  // eCodec
                                           low_latency,
                                           false,  // bDeviceFramePitched
                                           nullptr,
                                           nullptr,
                                           false,
                                           0,
                                           0,
                                           1000,
                                           false);  // force_zero_latency

    if (verbose_.get()) {
      HOLOSCAN_LOG_INFO("Initialized packetized {} decoder (input mode: {}, low latency: {})",
                        codec_name,
                        input_mode,
                        low_latency);
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Failed to initialize packetized decoder: {}", e.what());
    decoder_.reset();
    CudaCheck(cuCtxPopCurrent(nullptr));
    throw;
  }
}

void NvVideoDecoderOp::stop() {
  // Destroy codec objects while the retained CUDA primary context is still valid.
  release_pending_frames();
  decoder_.reset();
  demuxer_.reset();
  file_data_provider_.reset();

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
