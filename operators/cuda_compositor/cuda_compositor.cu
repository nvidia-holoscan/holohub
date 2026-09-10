// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cuda_compositor/cuda_compositor.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/activation.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "v4l2_depth_common/cuda_device_guard.hpp"
#include "v4l2_depth_common/tensor_utils.hpp"

namespace holoscan::examples::v4l2_depth {
namespace {

[[nodiscard]] holoscan::Error cuda_error(const char* operation, cudaError_t status) {
  return holoscan::Error{
      holoscan::ErrorCode::kFailure,
      std::string(operation) + " failed: " + cudaGetErrorString(status)};
}

[[nodiscard]] std::size_t checked_bgra_bytes(std::int32_t width, std::int32_t height) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("CudaCompositorOp dimensions must be positive");
  }
  const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if (pixels > std::numeric_limits<std::size_t>::max() / 4U) {
    throw std::overflow_error("CudaCompositorOp image byte size overflows size_t");
  }
  return pixels * 4U;
}

[[nodiscard]] bool valid_bgra_tensor(const holoscan::Tensor& tensor,
                                     std::span<const std::int64_t> shape,
                                     std::size_t bytes,
                                     std::int32_t cuda_device) {
  return tensor.device().device_type == kDLCUDA &&
         tensor.device().device_id == cuda_device &&
         same_dtype(tensor.dtype(), kUInt8Dtype) &&
         shape_equals(tensor, shape) &&
         tensor.is_contiguous() &&
         tensor.nbytes() == static_cast<std::int64_t>(bytes) &&
         tensor.data_as<std::uint8_t>() != nullptr;
}

__global__ void over_blend_kernel(std::uint8_t* output,
                                  const std::uint8_t* overlay,
                                  std::size_t pixel_count) {
  const std::size_t pixel =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= pixel_count) {
    return;
  }
  const std::size_t offset = pixel * 4U;
  const int alpha = static_cast<int>(overlay[offset + 3U]);
  if (alpha == 0) {
    return;
  }
  if (alpha == 255) {
    output[offset] = overlay[offset];
    output[offset + 1U] = overlay[offset + 1U];
    output[offset + 2U] = overlay[offset + 2U];
  } else {
    const int inverse_alpha = 255 - alpha;
    output[offset] = static_cast<std::uint8_t>(
        (static_cast<int>(overlay[offset]) * alpha +
         static_cast<int>(output[offset]) * inverse_alpha) /
        255);
    output[offset + 1U] = static_cast<std::uint8_t>(
        (static_cast<int>(overlay[offset + 1U]) * alpha +
         static_cast<int>(output[offset + 1U]) * inverse_alpha) /
        255);
    output[offset + 2U] = static_cast<std::uint8_t>(
        (static_cast<int>(overlay[offset + 2U]) * alpha +
         static_cast<int>(output[offset + 2U]) * inverse_alpha) /
        255);
  }
  output[offset + 3U] = 255U;
}

}  // namespace

CudaCompositorOp::CudaCompositorOp(std::int32_t width,
                                   std::int32_t height,
                                   std::int32_t cuda_device)
    : width_(width), height_(height), cuda_device_(cuda_device) {
  static_cast<void>(checked_bgra_bytes(width_, height_));
  if (cuda_device_ < 0) {
    throw std::invalid_argument("CudaCompositorOp CUDA device must be nonnegative");
  }
}

CudaCompositorOp::~CudaCompositorOp() {
  stop();
}

void CudaCompositorOp::setup(holoscan::OperatorSpec& spec) {
  const holoscan::TensorRepresentation bgra_layout{
      .memory_kind = holoscan::MemoryKind::kCudaDevice,
      .dtype = kUInt8Dtype,
      .rank = 3U,
  };
  spec.input(base, "base")
      .queue_depth(1U)
      .expects_tensor(holoscan::TensorInputSpec{.representation = bgra_layout});
  spec.input(overlay, "overlay")
      .queue_depth(1U)
      .expects_tensor(holoscan::TensorInputSpec{.representation = bgra_layout});
  spec.output(output, "output")
      .max_emits_per_compute(1U)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = bgra_layout,
          .bounds = holoscan::tensor_bounds(checked_bgra_bytes(width_, height_)),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

void CudaCompositorOp::start() {
  if (overlay_latch_ != nullptr) {
    throw std::runtime_error("CudaCompositorOp overlay latch is already allocated");
  }
  const CudaDeviceGuard selected_device{cuda_device_};
  if (!selected_device.active()) {
    throw std::runtime_error(selected_device.error_message("CudaCompositorOp startup"));
  }
  const std::size_t bytes = checked_bgra_bytes(width_, height_);
  if (const cudaError_t status =
          cudaMalloc(reinterpret_cast<void**>(&overlay_latch_), bytes);
      status != cudaSuccess) {
    overlay_latch_ = nullptr;
    throw std::runtime_error(
        std::string("CudaCompositorOp cudaMalloc failed: ") + cudaGetErrorString(status));
  }
  if (const cudaError_t status = cudaMemset(overlay_latch_, 0, bytes);
      status != cudaSuccess) {
    static_cast<void>(cudaFree(std::exchange(overlay_latch_, nullptr)));
    throw std::runtime_error(
        std::string("CudaCompositorOp cudaMemset failed: ") + cudaGetErrorString(status));
  }
  if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) {
    static_cast<void>(cudaFree(std::exchange(overlay_latch_, nullptr)));
    throw std::runtime_error(
        std::string("CudaCompositorOp startup synchronization failed: ") +
        cudaGetErrorString(status));
  }
}

void CudaCompositorOp::stop() {
  if (overlay_latch_ == nullptr) {
    return;
  }
  const CudaDeviceGuard selected_device{cuda_device_};
  if (!selected_device.active()) {
    const std::string message = selected_device.error_message("CudaCompositorOp cleanup");
    std::fprintf(stderr, "%s\n", message.c_str());
    return;
  }
  std::uint8_t* allocation = std::exchange(overlay_latch_, nullptr);
  if (const cudaError_t status = cudaFree(allocation); status != cudaSuccess) {
    std::fprintf(stderr, "CudaCompositorOp cudaFree failed: %s\n", cudaGetErrorString(status));
  }
}

holoscan::expected<void, holoscan::Error> CudaCompositorOp::compute(
    holoscan::ExecutionContext& context) {
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kNotSupported, "CUDA stream is unavailable"});
  }
  if (overlay_latch_ == nullptr) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kNotReady, "CudaCompositorOp overlay latch is unavailable"});
  }

  const std::array<std::int64_t, 3U> bgra_shape{
      static_cast<std::int64_t>(height_),
      static_cast<std::int64_t>(width_),
      4,
  };
  const std::size_t bytes = checked_bgra_bytes(width_, height_);
  const bool overlay_selected =
      overlay.selection().state == holoscan::InputState::kSelected;
  const bool base_selected = base.selection().state == holoscan::InputState::kSelected;

  if (overlay_selected) {
    auto overlay_sample = overlay.receive();
    if (!overlay_sample) {
      return holoscan::make_unexpected(std::move(overlay_sample).error());
    }
    if (!valid_bgra_tensor(overlay_sample->data, bgra_shape, bytes, cuda_device_)) {
      return holoscan::make_unexpected(invalid_tensor(
          "CudaCompositorOp expects contiguous CUDA uint8 [height,width,4] overlay input"));
    }
    if (const cudaError_t status = cudaMemcpyAsync(overlay_latch_,
                                                   overlay_sample->data.data(),
                                                   bytes,
                                                   cudaMemcpyDeviceToDevice,
                                                   stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_error("overlay latch copy", status));
    }
  }

  // The inputful empty contract lowers to OnAnyInput. An overlay witness only
  // refreshes the latch; a base witness publishes immediately, including before
  // the first overlay because start() initialized a transparent latch.
  if (!base_selected) {
    return {};
  }

  auto base_sample = base.receive();
  if (!base_sample) {
    return holoscan::make_unexpected(std::move(base_sample).error());
  }
  if (!valid_bgra_tensor(base_sample->data, bgra_shape, bytes, cuda_device_)) {
    return holoscan::make_unexpected(invalid_tensor(
        "CudaCompositorOp expects contiguous CUDA uint8 [height,width,4] base input"));
  }

  auto loan = output.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = bgra_shape, .dtype = kUInt8Dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (writer->byte_size() != bytes || writer->data() == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "invalid compositor output allocation"});
  }
  if (const cudaError_t status = cudaMemcpyAsync(writer->data(),
                                                 base_sample->data.data(),
                                                 bytes,
                                                 cudaMemcpyDeviceToDevice,
                                                 stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_error("base-frame copy", status));
  }

  constexpr unsigned int block_size = 256U;
  const std::size_t pixel_count =
      static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
  const auto grid_size =
      static_cast<unsigned int>((pixel_count + block_size - 1U) / block_size);
  over_blend_kernel<<<grid_size, block_size, 0U, stream>>>(
      writer->data_as<std::uint8_t>(), overlay_latch_, pixel_count);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_error("compositor kernel launch", status));
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return output.emit(std::move(*loan), forwarded_emit_options(base_sample->metadata));
}

}  // namespace holoscan::examples::v4l2_depth
