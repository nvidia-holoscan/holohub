// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "yuyv_to_bgra/yuyv_to_bgra.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "v4l2_depth_common/tensor_utils.hpp"

namespace holoscan::examples::v4l2_depth {
namespace {

[[nodiscard]] holoscan::Error cuda_error(const char* operation, cudaError_t status) {
  return holoscan::Error{
      holoscan::ErrorCode::kFailure,
      std::string(operation) + " failed: " + cudaGetErrorString(status)};
}

[[nodiscard]] std::size_t checked_image_bytes(std::int32_t width,
                                              std::int32_t height,
                                              std::size_t bytes_per_pixel) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("YuyvToBgraOp dimensions must be positive");
  }
  const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if (pixels > std::numeric_limits<std::size_t>::max() / bytes_per_pixel) {
    throw std::overflow_error("YuyvToBgraOp image byte size overflows size_t");
  }
  return pixels * bytes_per_pixel;
}

__device__ __forceinline__ std::uint8_t clamp_byte(int value) {
  const int nonnegative = value < 0 ? 0 : value;
  return static_cast<std::uint8_t>(nonnegative > 255 ? 255 : nonnegative);
}

__global__ void yuyv_to_bgra_kernel(const std::uint8_t* input,
                                    std::uint8_t* output,
                                    std::int32_t width,
                                    std::int32_t height) {
  const std::int32_t pair_x =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const std::int32_t y =
      static_cast<std::int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  if (pair_x >= width / 2 || y >= height) {
    return;
  }

  const std::size_t pair_index =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(width / 2) +
      static_cast<std::size_t>(pair_x);
  const std::size_t input_offset = pair_index * 4U;
  const int y0 = static_cast<int>(input[input_offset]);
  const int u = static_cast<int>(input[input_offset + 1U]) - 128;
  const int y1 = static_cast<int>(input[input_offset + 2U]);
  const int v = static_cast<int>(input[input_offset + 3U]) - 128;

  const auto write_pixel = [=] __device__(int luma, std::size_t output_offset) {
    const int c = luma > 16 ? luma - 16 : 0;
    const int red = (298 * c + 409 * v + 128) >> 8;
    const int green = (298 * c - 100 * u - 208 * v + 128) >> 8;
    const int blue = (298 * c + 516 * u + 128) >> 8;
    output[output_offset] = clamp_byte(blue);
    output[output_offset + 1U] = clamp_byte(green);
    output[output_offset + 2U] = clamp_byte(red);
    output[output_offset + 3U] = 255U;
  };  // NOLINT(readability/braces): CUDA device lambda declarations require a semicolon.

  const std::size_t first_pixel =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
      static_cast<std::size_t>(pair_x * 2);
  write_pixel(y0, first_pixel * 4U);
  write_pixel(y1, (first_pixel + 1U) * 4U);
}

}  // namespace

YuyvToBgraOp::YuyvToBgraOp(std::int32_t width, std::int32_t height)
    : width_(width), height_(height) {
  static_cast<void>(checked_image_bytes(width_, height_, 4U));
  if ((width_ & 1) != 0) {
    throw std::invalid_argument("YuyvToBgraOp width must be even for packed YUYV");
  }
}

void YuyvToBgraOp::setup(holoscan::OperatorSpec& spec) {
  const holoscan::TensorPortLayout input_layout{
      .memory_kind = holoscan::MemoryKind::kCudaDevice,
      .dtype = kUInt8Dtype,
      .rank = 3U,
  };
  const holoscan::TensorPortLayout output_layout{
      .memory_kind = holoscan::MemoryKind::kCudaDevice,
      .dtype = kUInt8Dtype,
      .rank = 3U,
  };
  spec.input(input, "input").queue_depth(1U).expects_tensor(input_layout);
  spec.output(output, "output")
      .max_emits_per_compute(1U)
      .produces_tensor(output_layout)
      .tensor_allocation(holoscan::TensorAllocationBounds{
          .max_rank = 3U,
          .max_byte_span = checked_image_bytes(width_, height_, 4U),
      });
}

holoscan::expected<void, holoscan::Error> YuyvToBgraOp::compute(
    holoscan::ExecutionContext& context) {
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kNotSupported, "CUDA stream is unavailable"});
  }

  auto sample = input.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }

  const std::array<std::int64_t, 3U> input_shape{
      static_cast<std::int64_t>(height_),
      static_cast<std::int64_t>(width_),
      2,
  };
  const std::size_t input_bytes = checked_image_bytes(width_, height_, 2U);
  const holoscan::Tensor& tensor = sample->data;
  const std::uint8_t* input_data = tensor.data_as<std::uint8_t>();
  if (tensor.device().device_type != kDLCUDA ||
      !same_dtype(tensor.dtype(), kUInt8Dtype) ||
      !shape_equals(tensor, input_shape) ||
      !tensor.is_contiguous() ||
      tensor.nbytes() != static_cast<std::int64_t>(input_bytes) ||
      input_data == nullptr) {
    return holoscan::make_unexpected(invalid_tensor(
        "YuyvToBgraOp expects contiguous CUDA uint8 [height,width,2] input"));
  }

  const std::array<std::int64_t, 3U> output_shape{
      static_cast<std::int64_t>(height_),
      static_cast<std::int64_t>(width_),
      4,
  };
  const std::size_t output_bytes = checked_image_bytes(width_, height_, 4U);
  auto loan = output.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = output_shape, .dtype = kUInt8Dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (writer->byte_size() != output_bytes || writer->data() == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "invalid BGRA output allocation"});
  }

  constexpr dim3 block{16U, 16U, 1U};
  const dim3 grid{
      static_cast<unsigned int>((width_ / 2 + static_cast<std::int32_t>(block.x) - 1) /
                                static_cast<std::int32_t>(block.x)),
      static_cast<unsigned int>((height_ + static_cast<std::int32_t>(block.y) - 1) /
                                static_cast<std::int32_t>(block.y)),
      1U,
  };
  yuyv_to_bgra_kernel<<<grid, block, 0U, stream>>>(
      input_data, writer->data_as<std::uint8_t>(), width_, height_);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_error("YUYV-to-BGRA kernel launch", status));
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return output.emit(std::move(*loan), forwarded_emit_options(sample->metadata));
}

}  // namespace holoscan::examples::v4l2_depth
