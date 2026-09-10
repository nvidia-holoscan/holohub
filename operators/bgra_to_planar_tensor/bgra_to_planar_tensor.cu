// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "bgra_to_planar_tensor/bgra_to_planar_tensor.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
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

[[nodiscard]] std::size_t checked_element_bytes(std::int32_t width,
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

__global__ void bgra_to_planar_tensor_kernel(const std::uint8_t* bgra,
                                             std::int32_t source_width,
                                             std::int32_t source_height,
                                             float* output,
                                             std::int32_t network_width,
                                             std::int32_t network_height,
                                             float mean_r,
                                             float mean_g,
                                             float mean_b,
                                             float std_r,
                                             float std_g,
                                             float std_b) {
  const std::int32_t output_x =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const std::int32_t output_y =
      static_cast<std::int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  if (output_x >= network_width || output_y >= network_height) {
    return;
  }

  const std::int32_t source_x =
      static_cast<std::int32_t>((static_cast<std::int64_t>(output_x) * source_width) /
                                network_width);
  const std::int32_t source_y =
      static_cast<std::int32_t>((static_cast<std::int64_t>(output_y) * source_height) /
                                network_height);
  const std::size_t source_offset =
      (static_cast<std::size_t>(source_y) * static_cast<std::size_t>(source_width) +
       static_cast<std::size_t>(source_x)) *
      4U;
  const std::size_t plane =
      static_cast<std::size_t>(network_width) * static_cast<std::size_t>(network_height);
  const std::size_t output_offset =
      static_cast<std::size_t>(output_y) * static_cast<std::size_t>(network_width) +
      static_cast<std::size_t>(output_x);

  output[output_offset] =
      (static_cast<float>(bgra[source_offset + 2U]) / 255.0F - mean_r) / std_r;
  output[plane + output_offset] =
      (static_cast<float>(bgra[source_offset + 1U]) / 255.0F - mean_g) / std_g;
  output[2U * plane + output_offset] =
      (static_cast<float>(bgra[source_offset]) / 255.0F - mean_b) / std_b;
}

}  // namespace

BgraToPlanarTensorOp::BgraToPlanarTensorOp(std::int32_t source_width,
                                           std::int32_t source_height,
                                           std::int32_t network_width,
                                           std::int32_t network_height,
                                           std::array<float, 3> mean,
                                           std::array<float, 3> standard_deviation)
    : source_width_(source_width),
      source_height_(source_height),
      network_width_(network_width),
      network_height_(network_height),
      mean_(mean),
      standard_deviation_(standard_deviation) {
  static_cast<void>(
      checked_element_bytes(source_width_, source_height_, 4U, 1U, "source image"));
  static_cast<void>(
      checked_element_bytes(network_width_, network_height_, 3U, sizeof(float), "model input"));
  for (std::size_t channel = 0U; channel < mean_.size(); ++channel) {
    if (!std::isfinite(mean_[channel]) ||
        !std::isfinite(standard_deviation_[channel]) ||
        standard_deviation_[channel] <= 0.0F) {
      throw std::invalid_argument(
          "BgraToPlanarTensorOp normalization values must be finite and standard deviations positive");
    }
  }
}

void BgraToPlanarTensorOp::setup(holoscan::OperatorSpec& spec) {
  const holoscan::TensorRepresentation input_layout{
      .memory_kind = holoscan::MemoryKind::kCudaDevice,
      .dtype = kUInt8Dtype,
      .rank = 3U,
  };
  const holoscan::TensorRepresentation output_layout{
      .memory_kind = holoscan::MemoryKind::kCudaDevice,
      .dtype = kFloat32Dtype,
      .rank = 4U,
  };
  spec.input(input, "input")
      .queue_depth(1U)
      .expects_tensor(holoscan::TensorInputSpec{.representation = input_layout});
  spec.output(output, "output")
      .max_emits_per_compute(1U)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = output_layout,
          .bounds = holoscan::tensor_bounds(checked_element_bytes(
              network_width_, network_height_, 3U, sizeof(float), "model input")),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

holoscan::expected<void, holoscan::Error> BgraToPlanarTensorOp::compute(
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
      static_cast<std::int64_t>(source_height_),
      static_cast<std::int64_t>(source_width_),
      4,
  };
  const std::size_t input_bytes =
      checked_element_bytes(source_width_, source_height_, 4U, 1U, "source image");
  const holoscan::Tensor& tensor = sample->data;
  const std::uint8_t* input_data = tensor.data_as<std::uint8_t>();
  if (tensor.device().device_type != kDLCUDA ||
      !same_dtype(tensor.dtype(), kUInt8Dtype) ||
      !shape_equals(tensor, input_shape) ||
      !tensor.is_contiguous() ||
      tensor.nbytes() != static_cast<std::int64_t>(input_bytes) ||
      input_data == nullptr) {
    return holoscan::make_unexpected(invalid_tensor(
        "BgraToPlanarTensorOp expects contiguous CUDA uint8 "
        "[source_height,source_width,4] input"));
  }

  const std::array<std::int64_t, 4U> output_shape{
      1,
      3,
      static_cast<std::int64_t>(network_height_),
      static_cast<std::int64_t>(network_width_),
  };
  const std::size_t output_bytes =
      checked_element_bytes(network_width_, network_height_, 3U, sizeof(float), "model input");
  auto loan = output.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = output_shape, .dtype = kFloat32Dtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (writer->byte_size() != output_bytes || writer->data() == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "invalid model-input output allocation"});
  }

  constexpr dim3 block{16U, 16U, 1U};
  const dim3 grid{
      static_cast<unsigned int>((network_width_ + static_cast<std::int32_t>(block.x) - 1) /
                                static_cast<std::int32_t>(block.x)),
      static_cast<unsigned int>((network_height_ + static_cast<std::int32_t>(block.y) - 1) /
                                static_cast<std::int32_t>(block.y)),
      1U,
  };
  bgra_to_planar_tensor_kernel<<<grid, block, 0U, stream>>>(
      input_data,
      source_width_,
      source_height_,
      writer->data_as<float>(),
      network_width_,
      network_height_,
      mean_[0],
      mean_[1],
      mean_[2],
      standard_deviation_[0],
      standard_deviation_[1],
      standard_deviation_[2]);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_error("image-to-tensor kernel launch", status));
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return output.emit(std::move(*loan), forwarded_emit_options(sample->metadata));
}

}  // namespace holoscan::examples::v4l2_depth
