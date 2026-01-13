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

#include "pattern_generator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

namespace holoscan {

const char* get_pattern_name(int pattern) {
  const char* pattern_names[] = {"animated gradient", "animated checkerboard", "color bars"};
  return (pattern >= 0 && pattern <= 2) ? pattern_names[pattern] : "unknown";
}

const char* get_pattern_name(PatternType pattern) {
  return get_pattern_name(static_cast<int>(pattern));
}

holoscan::gxf::Entity PatternEntityGenerator::generate(int width, int height,
                                                       nvidia::gxf::MemoryStorageType storage_type,
                                                       holoscan::Allocator* allocator) {
  HOLOSCAN_LOG_DEBUG("Generating {}x{} pattern entity (storage: {})",
                     width,
                     height,
                     storage_type == nvidia::gxf::MemoryStorageType::kDevice ? "device" : "host");

  // Validate inputs
  if (width <= 0 || height <= 0) {
    HOLOSCAN_LOG_ERROR("Invalid dimensions: {}x{} (must be positive)", width, height);
    return holoscan::gxf::Entity();
  }

  if (!allocator) {
    HOLOSCAN_LOG_ERROR("Allocator is null");
    return holoscan::gxf::Entity();
  }

  gxf_context_t context = allocator->gxf_context();

  // Create allocator handle
  auto allocator_handle =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context, allocator->gxf_cid());
  if (!allocator_handle) {
    HOLOSCAN_LOG_ERROR("Failed to create allocator handle");
    return holoscan::gxf::Entity();
  }

  // Create GXF entity with tensor using CreateTensorMap
  auto gxf_entity = nvidia::gxf::CreateTensorMap(
      context,
      allocator_handle.value(),
      {{"video_frame",
        storage_type,
        nvidia::gxf::Shape{static_cast<int32_t>(height),
                           static_cast<int32_t>(width),
                           static_cast<int32_t>(RGBA_CHANNELS)},
        nvidia::gxf::PrimitiveType::kUnsigned8,
        0,
        nvidia::gxf::ComputeTrivialStrides(nvidia::gxf::Shape{static_cast<int32_t>(height),
                                                              static_cast<int32_t>(width),
                                                              static_cast<int32_t>(RGBA_CHANNELS)},
                                           sizeof(uint8_t))}},
      false);

  if (!gxf_entity) {
    HOLOSCAN_LOG_ERROR("Failed to create GXF entity. Error code: {}",
                       static_cast<int>(gxf_entity.error()));
    return holoscan::gxf::Entity();
  }

  // Get the tensor from the entity
  auto maybe_tensor = gxf_entity.value().get<nvidia::gxf::Tensor>("video_frame");
  if (!maybe_tensor) {
    HOLOSCAN_LOG_ERROR("Failed to get tensor from entity");
    return holoscan::gxf::Entity();
  }

  // Calculate buffer size with overflow protection
  const auto w = static_cast<size_t>(width);
  const auto h = static_cast<size_t>(height);
  const auto c = static_cast<size_t>(RGBA_CHANNELS);
  if (h != 0 && w > std::numeric_limits<size_t>::max() / (h * c)) {
    HOLOSCAN_LOG_ERROR("Requested frame size is too large: {}x{}", width, height);
    return holoscan::gxf::Entity();
  }
  size_t buffer_size = w * h * c;

  // For device memory, generate pattern in host buffer first, then copy to device
  if (storage_type == nvidia::gxf::MemoryStorageType::kDevice) {
    // Allocate temporary host buffer
    std::vector<uint8_t> host_buffer(buffer_size);

    // Generate pattern in host buffer (call derived class implementation)
    generate_pattern_data(host_buffer.data(), width, height);

    // Get device pointer and validate
    void* device_ptr = maybe_tensor.value()->pointer();
    if (!device_ptr) {
      HOLOSCAN_LOG_ERROR("Failed to get device tensor data pointer");
      return holoscan::gxf::Entity();
    }

    // Copy from host to device
    cudaError_t cuda_result =
        cudaMemcpy(device_ptr, host_buffer.data(), buffer_size, cudaMemcpyHostToDevice);

    if (cuda_result != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Failed to copy pattern to device memory: {}",
                         cudaGetErrorString(cuda_result));
      return holoscan::gxf::Entity();
    }
  } else {
    // Host memory - generate pattern directly
    uint8_t* data = static_cast<uint8_t*>(maybe_tensor.value()->pointer());
    if (!data) {
      HOLOSCAN_LOG_ERROR("Failed to get tensor data pointer");
      return holoscan::gxf::Entity();
    }

    // Generate pattern (call derived class implementation)
    generate_pattern_data(data, width, height);
  }

  // Wrap the GXF entity in a Holoscan entity and return
  return holoscan::gxf::Entity(std::move(gxf_entity.value()));
}

void GradientPatternGenerator::generate_pattern_data(uint8_t* data, int width, int height) {
  time_offset_ += GRADIENT_TIME_STEP;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * RGBA_CHANNELS;
      // Animated color gradient - std::sin/cos are overloaded for float in C++
      uint8_t r = static_cast<uint8_t>(128 + 127 * std::sin(x * 0.01f + time_offset_));
      uint8_t g = static_cast<uint8_t>(128 + 127 * std::sin(y * 0.01f + time_offset_));
      uint8_t b = static_cast<uint8_t>(128 + 127 * std::cos((x + y) * 0.005f + time_offset_));
      set_rgba_pixel(data, idx, r, g, b);
    }
  }
}

void CheckerboardPatternGenerator::generate_pattern_data(uint8_t* data, int width, int height) {
  animation_time_ += CHECKERBOARD_TIME_STEP;

  // Calculate square size with safety against division by zero
  int square_size =
      CHECKERBOARD_BASE_SIZE + static_cast<int>(CHECKERBOARD_VARIATION * std::sin(animation_time_));
  square_size = std::max(1, square_size);  // Ensure at least 1 to avoid division by zero

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * RGBA_CHANNELS;
      bool is_white = ((x / square_size) + (y / square_size)) % 2 == 0;
      uint8_t color = is_white ? ALPHA_OPAQUE : 0;
      set_rgba_pixel(data, idx, color, color, color);
    }
  }
}

void ColorBarsPatternGenerator::generate_pattern_data(uint8_t* data, int width, int height) {
  // SMPTE color bars (7 bars) - static to avoid recreating on each call
  static constexpr uint8_t colors[SMPTE_COLOR_BARS][RGB_CHANNELS] = {
      {255, 255, 255},  // White
      {255, 255, 0},    // Yellow
      {0, 255, 255},    // Cyan
      {0, 255, 0},      // Green
      {255, 0, 255},    // Magenta
      {255, 0, 0},      // Red
      {0, 0, 255}       // Blue
  };

  int bar_width = width / SMPTE_COLOR_BARS;
  if (bar_width == 0)
    bar_width = 1;  // Prevent division by zero for very small widths
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * RGBA_CHANNELS;
      int bar_idx = x / bar_width;
      if (bar_idx >= SMPTE_COLOR_BARS)
        bar_idx = SMPTE_COLOR_BARS - 1;

      set_rgba_pixel(data,
                     idx,
                     colors[bar_idx][0],   // R
                     colors[bar_idx][1],   // G
                     colors[bar_idx][2]);  // B
    }
  }
}

void PatternGenOperator::setup(OperatorSpec& spec) {
  spec.output<gxf::Entity>("output");

  spec.param(allocator_, "allocator", "Allocator", "Memory allocator for tensor allocation");
  spec.param(width_, "width", "Width", "Frame width in pixels", 1920);
  spec.param(height_, "height", "Height", "Frame height in pixels", 1080);
  spec.param(
      pattern_, "pattern", "Pattern", "Pattern type: 0=gradient, 1=checkerboard, 2=color bars", 0);
  spec.param(
      storage_type_, "storage_type", "StorageType", "Memory storage type: 0=host, 1=device", 0);
}

void PatternGenOperator::start() {
  // Create the appropriate pattern generator based on the pattern type
  int pattern = pattern_.get();
  switch (static_cast<PatternType>(pattern)) {
    case PatternType::Gradient:
      generator_ = std::make_unique<GradientPatternGenerator>();
      HOLOSCAN_LOG_INFO("Using animated gradient pattern generator");
      break;
    case PatternType::Checkerboard:
      generator_ = std::make_unique<CheckerboardPatternGenerator>();
      HOLOSCAN_LOG_INFO("Using animated checkerboard pattern generator");
      break;
    case PatternType::ColorBars:
      generator_ = std::make_unique<ColorBarsPatternGenerator>();
      HOLOSCAN_LOG_INFO("Using SMPTE color bars pattern generator");
      break;
    default:
      generator_ = std::make_unique<GradientPatternGenerator>();
      HOLOSCAN_LOG_WARN("Invalid pattern type {}, defaulting to gradient", pattern);
      break;
  }
}

void PatternGenOperator::compute(InputContext& input, OutputContext& output,
                                 ExecutionContext& context) {
  HOLOSCAN_LOG_DEBUG("Generating pattern");

  auto allocator_ptr = allocator_.get().get();
  if (!allocator_ptr) {
    HOLOSCAN_LOG_ERROR("Allocator parameter is not set");
    return;
  }

  // Convert storage_type parameter (0=host, 1=device) to MemoryStorageType enum
  nvidia::gxf::MemoryStorageType storage = (storage_type_.get() == 1)
                                               ? nvidia::gxf::MemoryStorageType::kDevice
                                               : nvidia::gxf::MemoryStorageType::kHost;

  // Generate a pattern entity with tensors using the polymorphic generator
  auto entity = generator_->generate(width_.get(), height_.get(), storage, allocator_ptr);
  if (!entity) {
    HOLOSCAN_LOG_ERROR("Failed to generate pattern entity");
    return;
  }

  HOLOSCAN_LOG_DEBUG("Pattern entity generated, emitting to output");

  // Emit the entity to the output port
  output.emit(entity, "output");
  HOLOSCAN_LOG_DEBUG("Pattern entity emitted");
}

}  // namespace holoscan
