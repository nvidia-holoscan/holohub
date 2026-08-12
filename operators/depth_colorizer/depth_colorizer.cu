// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "depth_colorizer/depth_colorizer.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "v4l2_depth_common/cuda_device_guard.hpp"
#include "v4l2_depth_common/tensor_utils.hpp"

namespace holoscan::examples::v4l2_depth {
namespace {

constexpr unsigned int kReductionThreads = 256U;
constexpr unsigned int kMaximumReductionBlocks = 1024U;

[[nodiscard]] holoscan::Error cuda_error(const char* operation, cudaError_t status) {
  return holoscan::Error{
      holoscan::ErrorCode::kFailure,
      std::string(operation) + " failed: " + cudaGetErrorString(status)};
}

[[nodiscard]] std::size_t checked_image_bytes(std::int32_t width,
                                              std::int32_t height,
                                              std::size_t channels,
                                              std::size_t element_bytes,
                                              const char* description) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument(std::string(description) + " dimensions must be positive");
  }
  const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if (pixels > std::numeric_limits<std::size_t>::max() / channels ||
      pixels * channels > std::numeric_limits<std::size_t>::max() / element_bytes) {
    throw std::overflow_error(std::string(description) + " byte size overflows size_t");
  }
  return pixels * channels * element_bytes;
}

__device__ void atomic_min_float(float* address, float value) {
  auto* bits = reinterpret_cast<int*>(address);
  int old = *bits;
  while (value < __int_as_float(old)) {
    const int assumed = old;
    old = atomicCAS(bits, assumed, __float_as_int(value));
    if (old == assumed) {
      break;
    }
  }
}

__device__ void atomic_max_float(float* address, float value) {
  auto* bits = reinterpret_cast<int*>(address);
  int old = *bits;
  while (value > __int_as_float(old)) {
    const int assumed = old;
    old = atomicCAS(bits, assumed, __float_as_int(value));
    if (old == assumed) {
      break;
    }
  }
}

__global__ void initialize_min_max_kernel(float* min_max) {
  if (threadIdx.x == 0U && blockIdx.x == 0U) {
    min_max[0] = CUDART_INF_F;
    min_max[1] = -CUDART_INF_F;
  }
}

__global__ void finite_min_max_kernel(const float* depth,
                                      std::size_t element_count,
                                      float* min_max) {
  __shared__ float block_min[kReductionThreads];
  __shared__ float block_max[kReductionThreads];

  float local_min = CUDART_INF_F;
  float local_max = -CUDART_INF_F;
  const std::size_t first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = first; index < element_count; index += stride) {
    const float value = depth[index];
    if (isfinite(value)) {
      local_min = fminf(local_min, value);
      local_max = fmaxf(local_max, value);
    }
  }
  block_min[threadIdx.x] = local_min;
  block_max[threadIdx.x] = local_max;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2U; offset > 0U; offset >>= 1U) {
    if (threadIdx.x < offset) {
      block_min[threadIdx.x] =
          fminf(block_min[threadIdx.x], block_min[threadIdx.x + offset]);
      block_max[threadIdx.x] =
          fmaxf(block_max[threadIdx.x], block_max[threadIdx.x + offset]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0U) {
    if (isfinite(block_min[0])) {
      atomic_min_float(&min_max[0], block_min[0]);
    }
    if (isfinite(block_max[0])) {
      atomic_max_float(&min_max[1], block_max[0]);
    }
  }
}

__global__ void depth_colorize_kernel(const float* depth,
                                      std::int32_t depth_width,
                                      std::int32_t depth_height,
                                      const float* min_max,
                                      std::uint8_t* bgra,
                                      std::int32_t output_width,
                                      std::int32_t output_height,
                                      std::uint8_t alpha) {
  const std::int32_t output_x =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const std::int32_t output_y =
      static_cast<std::int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  if (output_x >= output_width || output_y >= output_height) {
    return;
  }

  const std::int32_t source_x =
      static_cast<std::int32_t>((static_cast<std::int64_t>(output_x) * depth_width) /
                                output_width);
  const std::int32_t source_y =
      static_cast<std::int32_t>((static_cast<std::int64_t>(output_y) * depth_height) /
                                output_height);
  const float value =
      depth[static_cast<std::size_t>(source_y) * static_cast<std::size_t>(depth_width) +
            static_cast<std::size_t>(source_x)];
  const float minimum = min_max[0];
  const float maximum = min_max[1];

  float normalized = 0.0F;
  if (isfinite(value) && isfinite(minimum) && isfinite(maximum) && maximum > minimum) {
    normalized = fminf(fmaxf((value - minimum) / (maximum - minimum), 0.0F), 1.0F);
  }
  const float red =
      fminf(fmaxf(1.5F - fabsf(4.0F * normalized - 3.0F), 0.0F), 1.0F);
  const float green =
      fminf(fmaxf(1.5F - fabsf(4.0F * normalized - 2.0F), 0.0F), 1.0F);
  const float blue =
      fminf(fmaxf(1.5F - fabsf(4.0F * normalized - 1.0F), 0.0F), 1.0F);

  const std::size_t output_offset =
      (static_cast<std::size_t>(output_y) * static_cast<std::size_t>(output_width) +
       static_cast<std::size_t>(output_x)) *
      4U;
  bgra[output_offset] = static_cast<std::uint8_t>(255.0F * blue);
  bgra[output_offset + 1U] = static_cast<std::uint8_t>(255.0F * green);
  bgra[output_offset + 2U] = static_cast<std::uint8_t>(255.0F * red);
  bgra[output_offset + 3U] = alpha;
}

}  // namespace

DepthColorizerOp::DepthColorizerOp(std::int32_t output_width,
                                   std::int32_t output_height,
                                   std::int32_t depth_width,
                                   std::int32_t depth_height,
                                   std::uint8_t alpha,
                                   std::int32_t cuda_device)
    : output_width_(output_width),
      output_height_(output_height),
      depth_width_(depth_width),
      depth_height_(depth_height),
      alpha_(alpha),
      cuda_device_(cuda_device) {
  static_cast<void>(
      checked_image_bytes(output_width_, output_height_, 4U, 1U, "colorized image"));
  static_cast<void>(
      checked_image_bytes(depth_width_, depth_height_, 1U, sizeof(float), "depth map"));
  if (cuda_device_ < 0) {
    throw std::invalid_argument("DepthColorizerOp CUDA device must be nonnegative");
  }
}

DepthColorizerOp::~DepthColorizerOp() {
  stop();
}

void DepthColorizerOp::setup(holoscan::OperatorSpec& spec) {
  const holoscan::TensorPortLayout input_layout{
      .memory_kind = holoscan::MemoryKind::kCudaDevice,
      .dtype = kFloat32Dtype,
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
          .max_byte_span =
              checked_image_bytes(output_width_, output_height_, 4U, 1U, "colorized image"),
      });
}

void DepthColorizerOp::start() {
  if (device_min_max_ != nullptr) {
    throw std::runtime_error("DepthColorizerOp device reduction state is already allocated");
  }
  const CudaDeviceGuard selected_device{cuda_device_};
  if (!selected_device.active()) {
    throw std::runtime_error(selected_device.error_message("DepthColorizerOp startup"));
  }
  if (const cudaError_t status =
          cudaMalloc(reinterpret_cast<void**>(&device_min_max_), 2U * sizeof(float));
      status != cudaSuccess) {
    device_min_max_ = nullptr;
    throw std::runtime_error(
        std::string("DepthColorizerOp cudaMalloc failed: ") + cudaGetErrorString(status));
  }
}

void DepthColorizerOp::stop() {
  if (device_min_max_ == nullptr) {
    return;
  }
  const CudaDeviceGuard selected_device{cuda_device_};
  if (!selected_device.active()) {
    const std::string message = selected_device.error_message("DepthColorizerOp cleanup");
    std::fprintf(stderr, "%s\n", message.c_str());
    return;
  }
  float* allocation = std::exchange(device_min_max_, nullptr);
  if (const cudaError_t status = cudaFree(allocation); status != cudaSuccess) {
    std::fprintf(
        stderr, "DepthColorizerOp cudaFree failed: %s\n", cudaGetErrorString(status));
  }
}

holoscan::expected<void, holoscan::Error> DepthColorizerOp::compute(
    holoscan::ExecutionContext& context) {
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kNotSupported, "CUDA stream is unavailable"});
  }
  if (device_min_max_ == nullptr) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kNotReady, "DepthColorizerOp reduction state is unavailable"});
  }

  auto sample = input.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  const holoscan::Tensor& tensor = sample->data;
  const auto shape = tensor.shape_span();
  bool shape_valid = shape.size() >= 2U &&
                     shape[shape.size() - 2U] == static_cast<std::int64_t>(depth_height_) &&
                     shape.back() == static_cast<std::int64_t>(depth_width_);
  for (std::size_t index = 0U; shape_valid && index + 2U < shape.size(); ++index) {
    shape_valid = shape[index] == 1;
  }
  const std::size_t depth_bytes =
      checked_image_bytes(depth_width_, depth_height_, 1U, sizeof(float), "depth map");
  const float* depth = tensor.data_as<float>();
  if (tensor.device().device_type != kDLCUDA ||
      tensor.device().device_id != cuda_device_ ||
      !same_dtype(tensor.dtype(), kFloat32Dtype) ||
      !shape_valid ||
      !tensor.is_contiguous() ||
      tensor.nbytes() != static_cast<std::int64_t>(depth_bytes) ||
      depth == nullptr) {
    return holoscan::make_unexpected(invalid_tensor(
        "DepthColorizerOp expects contiguous CUDA float32 depth with singleton leading axes "
        "and configured trailing [height,width]"));
  }

  const std::array<std::int64_t, 3U> output_shape{
      static_cast<std::int64_t>(output_height_),
      static_cast<std::int64_t>(output_width_),
      4,
  };
  const std::size_t output_bytes =
      checked_image_bytes(output_width_, output_height_, 4U, 1U, "colorized image");
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
        holoscan::Error{holoscan::ErrorCode::kFailure, "invalid colorized output allocation"});
  }

  initialize_min_max_kernel<<<1U, 1U, 0U, stream>>>(device_min_max_);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_error("min/max initialization kernel launch", status));
  }

  const std::size_t depth_elements =
      static_cast<std::size_t>(depth_width_) * static_cast<std::size_t>(depth_height_);
  const std::size_t requested_blocks =
      (depth_elements + kReductionThreads - 1U) / kReductionThreads;
  const auto reduction_blocks = static_cast<unsigned int>(
      requested_blocks < kMaximumReductionBlocks ? requested_blocks : kMaximumReductionBlocks);
  finite_min_max_kernel<<<reduction_blocks, kReductionThreads, 0U, stream>>>(
      depth, depth_elements, device_min_max_);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_error("finite min/max kernel launch", status));
  }

  constexpr dim3 color_block{16U, 16U, 1U};
  const dim3 color_grid{
      static_cast<unsigned int>((output_width_ + static_cast<std::int32_t>(color_block.x) - 1) /
                                static_cast<std::int32_t>(color_block.x)),
      static_cast<unsigned int>((output_height_ + static_cast<std::int32_t>(color_block.y) - 1) /
                                static_cast<std::int32_t>(color_block.y)),
      1U,
  };
  depth_colorize_kernel<<<color_grid, color_block, 0U, stream>>>(
      depth,
      depth_width_,
      depth_height_,
      device_min_max_,
      writer->data_as<std::uint8_t>(),
      output_width_,
      output_height_,
      alpha_);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_error("depth-colorize kernel launch", status));
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return output.emit(std::move(*loan), forwarded_emit_options(sample->metadata));
}

}  // namespace holoscan::examples::v4l2_depth
