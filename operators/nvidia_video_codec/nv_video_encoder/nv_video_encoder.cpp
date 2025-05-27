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

#include "nv_video_encoder.hpp"

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

#include "../api/NvEncoder/NvEncoderCuda.h"
#include "../api/utils.h"

namespace holoscan::ops {

const std::map<std::string, GUID> NvVideoEncoderOp::CODEC_GUIDS{{"H264", NV_ENC_CODEC_H264_GUID},
                                                                {"HEVC", NV_ENC_CODEC_HEVC_GUID}};

const std::map<std::string, GUID> NvVideoEncoderOp::PRESET_GUIDS{{"P3", NV_ENC_PRESET_P3_GUID},
                                                                 {"P4", NV_ENC_PRESET_P4_GUID},
                                                                 {"P5", NV_ENC_PRESET_P5_GUID},
                                                                 {"P6", NV_ENC_PRESET_P6_GUID},
                                                                 {"P7", NV_ENC_PRESET_P7_GUID}};
void NvVideoEncoderOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("in");
  spec.output<holoscan::gxf::Entity>("out");

  spec.param(cuda_device_ordinal_,
             "cuda_device_ordinal",
             "CudaDeviceOrdinal",
             "Device to use for CUDA operations");
  spec.param(allocator_, "allocator", "Allocator", "Allocator for output buffers.");
  spec.param(width_, "width", "Width", "Video Frame Width", 1920u);
  spec.param(height_, "height", "Height", "Video Frame Height", 1080u);
  spec.param(preset_, "preset", "Preset", "Preset for the encoder", std::string("P3"));
  spec.param(
      codec_, "codec", "Codec", "Codec for the encoder. (H264 or HEVC)", std::string("H264"));
  spec.param(bitrate_, "bitrate", "Bitrate", "Bitrate for the encoder", 10000000u);
  spec.param(frame_rate_, "frame_rate", "Frame Rate", "Frame rate for the encoder", 60u);
  spec.param(rate_control_mode_,
             "rate_control_mode",
             "Rate Control Mode",
             "Rate control mode for the encoder",
             0u);
  spec.param(multi_pass_encoding_,
             "multi_pass_encoding",
             "Multi Pass Encoding",
             "Multi pass encoding for the encoder",
             1u);
  spec.param(
      copy_to_host_, "copy_to_host", "CopyToHost", "Copy encoded frame to host memory", false);

  cuda_stream_handler_.define_params(spec);
}

void NvVideoEncoderOp::initialize() {
  Operator::initialize();

  // Initialize CUDA
  CudaCheck(cuInit(0));

  // Get the CUDA device
  CUdevice cu_device;
  CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal_.get()));
  cu_device_ = cu_device;

  // Retain the primary context for the device
  CudaCheck(cuDevicePrimaryCtxRetain(&cu_context_, cu_device_));

  // Set the current context
  CudaCheck(cuCtxPushCurrent(cu_context_));

  // Initialize NVIDIA encoder with CUDA context
  NV_ENC_BUFFER_FORMAT input_format = NV_ENC_BUFFER_FORMAT_ABGR;
  encoder_ =
      std::make_unique<NvEncoderCuda>(cu_context_, width_.get(), height_.get(), input_format, 0);

  // Configure encoder with low latency or ultra low latency settings
  std::string upper_case = codec_.get();
  std::transform(upper_case.begin(), upper_case.end(), upper_case.begin(), ::toupper);
  GUID guidCodec = CODEC_GUIDS.at(upper_case);

  upper_case = preset_.get();
  std::transform(upper_case.begin(), upper_case.end(), upper_case.begin(), ::toupper);
  GUID guidPreset = PRESET_GUIDS.at(upper_case);

  // Initialize encoder parameters
  NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
  NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
  initializeParams.encodeConfig = &encodeConfig;

  // Create default encoder parameters
  encoder_->CreateDefaultEncoderParams(
      &initializeParams, guidCodec, guidPreset, NV_ENC_TUNING_INFO_LOW_LATENCY);

  // Set frame rate to 60 FPS
  initializeParams.frameRateNum = frame_rate_.get();
  initializeParams.frameRateDen = 1;

  // Configure GOP (Group of Pictures) settings for low latency
  // encodeConfig.gopLength = NVENC_INFINITE_GOPLENGTH;
  // encodeConfig.encodeCodecConfig.h264Config.idrPeriod = NVENC_INFINITE_GOPLENGTH;
  encodeConfig.rcParams.averageBitRate = bitrate_.get();
  encodeConfig.frameIntervalP = 1;
  switch (rate_control_mode_.get()) {
    case 0:
      encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
      break;
    case 1:
      encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
      break;
    case 2:
      encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
      break;
    default:
      HOLOSCAN_LOG_ERROR("Invalid rate control mode: {}", rate_control_mode_.get());
      throw std::runtime_error("Invalid rate control mode");
  }
  switch (multi_pass_encoding_.get()) {
    case 0:
      encodeConfig.rcParams.multiPass = NV_ENC_MULTI_PASS_DISABLED;
      break;
    case 1:
      encodeConfig.rcParams.multiPass = NV_ENC_TWO_PASS_QUARTER_RESOLUTION;
      break;
    case 2:
      encodeConfig.rcParams.multiPass = NV_ENC_TWO_PASS_FULL_RESOLUTION;
      break;
    default:
      HOLOSCAN_LOG_ERROR("Invalid multi pass encoding: {}", multi_pass_encoding_.get());
      throw std::runtime_error("Invalid multi pass encoding");
  }

  encodeConfig.rcParams.vbvBufferSize =
      (encodeConfig.rcParams.averageBitRate * initializeParams.frameRateDen /
       initializeParams.frameRateNum) *
      5;
  encodeConfig.rcParams.maxBitRate = encodeConfig.rcParams.averageBitRate;
  encodeConfig.rcParams.vbvInitialDelay = encodeConfig.rcParams.vbvBufferSize;

  // Create the encoder with configured parameters
  encoder_->CreateEncoder(&initializeParams);
  HOLOSCAN_LOG_INFO("Video encoder initialized: {}x{}", width_.get(), height_.get());
  if (copy_to_host_.get()) {
    HOLOSCAN_LOG_WARN(
        "The encoder is configured to copy encoded frame to host memory. This will increase "
        "latency.");
  }
}

void NvVideoEncoderOp::compute(InputContext& op_input, OutputContext& op_output,
                               ExecutionContext& context) {
  auto enter_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now().time_since_epoch())
                             .count();
  // Get input tensor
  auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("in");
  if (!maybe_entity) {
    throw std::runtime_error("Failed to receive input entity");
  }

  // Get the tensor from the input message
  auto tensor = maybe_entity.value().get<Tensor>("");
  if (!tensor) {
    throw std::runtime_error("Failed to get tensor from input message");
  }

  gxf_result_t stream_handler_result =
      cuda_stream_handler_.from_message(context.context(), maybe_entity.value());
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
  }
  // Initialize frame parameters
  NV_ENC_PIC_PARAMS picParams = {NV_ENC_PIC_PARAMS_VER};
  picParams.encodePicFlags =
      NV_ENC_PIC_FLAG_FORCEINTRA | NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
  int nFrameSize = encoder_->GetFrameSize();
  size_t data_size = tensor->size();

  if (data_size != nFrameSize) {
    HOLOSCAN_LOG_ERROR("Frame size mismatch: {} != {}", data_size, nFrameSize);
    return;
  }

  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
  std::vector<NvEncOutputFrame> vPacket;
  const NvEncInputFrame* encoderInputFrame = encoder_->GetNextInputFrame();

  // Copy frame data to GPU memory
  NvEncoderCuda::CopyToDeviceFrame(cu_context_,
                                   tensor->data(),
                                   0,
                                   (CUdeviceptr)encoderInputFrame->inputPtr,
                                   (int)encoderInputFrame->pitch,
                                   encoder_->GetEncodeWidth(),
                                   encoder_->GetEncodeHeight(),
                                   CU_MEMORYTYPE_HOST,
                                   encoderInputFrame->bufferFormat,
                                   encoderInputFrame->chromaOffsets,
                                   encoderInputFrame->numChromaPlanes);
  // Encode the frame
  encoder_->EncodeFrame(vPacket, &picParams);

  if (vPacket.size() == 0) {
    HOLOSCAN_LOG_ERROR("No encoded frame");
    return;
  } else if (vPacket.size() > 1) {
    HOLOSCAN_LOG_ERROR("More than one encoded frame found: {}. Using the first one.",
                       vPacket.size());
  }

  const cudaStream_t cuda_stream = cuda_stream_handler_.get_cuda_stream(context.context());
  auto out_message = nvidia::gxf::Entity::New(context.context());
  if (!out_message) {
    throw std::runtime_error("Failed to create output entity");
  }

  // Create tensor with explicit name and type
  auto out_tensor = out_message.value().add<nvidia::gxf::Tensor>("");
  if (!out_tensor) {
    throw std::runtime_error("Failed to allocate output tensor");
  }

  nvidia::gxf::Shape shape = nvidia::gxf::Shape{static_cast<int>(vPacket[0].frame.size())};

  if (copy_to_host_.get()) {
    out_tensor.value()->reshape<uint8_t>(
        shape, nvidia::gxf::MemoryStorageType::kHost, allocator.value());
    CUDA_TRY(cudaMemcpyAsync(out_tensor.value()->pointer(),
                             vPacket[0].frame.data(),
                             vPacket[0].frame.size(),
                             cudaMemcpyDeviceToHost,
                             cuda_stream));
  } else {
    out_tensor.value()->reshape<uint8_t>(
        shape, nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
    CUDA_TRY(cudaMemcpyAsync(out_tensor.value()->pointer(),
                             vPacket[0].frame.data(),
                             vPacket[0].frame.size(),
                             cudaMemcpyDeviceToDevice,
                             cuda_stream));
  }

  auto emit_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();

  auto meta = metadata();
  meta->set("video_encoder_enter_timestamp"s, enter_timestamp);
  meta->set("video_encoder_emit_timestamp"s, emit_timestamp);

  // Transmit the output message
  auto result = nvidia::gxf::Entity(std::move(out_message.value()));
  op_output.emit(result, "out");
}
}  // namespace holoscan::ops
