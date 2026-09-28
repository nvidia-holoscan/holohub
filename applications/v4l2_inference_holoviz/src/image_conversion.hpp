// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace camera_inference {

// Reject a classifier/detector tensor even if its element count happens to match the image.
inline std::size_t validate_rgb_layout(std::span<const std::int64_t> shape,
                                       std::span<const std::int64_t> strides, int width,
                                       int height) {
  if (width <= 0 || height <= 0 || width > 4096 || height > 4096 || shape.size() != 4 ||
      shape[0] != 1 || shape[1] != 3 || shape[2] != height || shape[3] != width) {
    throw std::invalid_argument("Model output must have shape [1, 3, model-height, model-width]");
  }
  std::int64_t expected_stride = sizeof(float);
  if (!strides.empty()) {
    if (strides.size() != shape.size()) {
      throw std::invalid_argument("Model output stride rank does not match its shape");
    }
    for (int dim = 3; dim >= 0; --dim) {
      if (shape[dim] > 1 && strides[dim] != expected_stride) {
        throw std::invalid_argument("Model output must be contiguous NCHW float32");
      }
      expected_stride *= shape[dim];
    }
  }
  return static_cast<std::size_t>(width) * height * 3;
}

// Interleave the model's three channel planes into the RGB8 image Holoviz expects.
inline void planar_rgb_float_to_u8(std::span<const float> input, std::span<std::uint8_t> output) {
  if (input.size() != output.size() || input.size() % 3 != 0) {
    throw std::invalid_argument("RGB buffers must have matching sizes divisible by three");
  }
  const std::size_t plane_size = input.size() / 3;
  for (std::size_t pixel = 0; pixel < plane_size; ++pixel) {
    for (std::size_t channel = 0; channel < 3; ++channel) {
      const float value = input[channel * plane_size + pixel];
      if (!std::isfinite(value)) {
        throw std::invalid_argument("Model output contains a non-finite RGB value");
      }
      output[pixel * 3 + channel] =
          static_cast<std::uint8_t>(std::clamp(value, 0.0F, 1.0F) * 255.0F + 0.5F);
    }
  }
}

}  // namespace camera_inference
