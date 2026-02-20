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

#ifndef APPLICATIONS_GSTREAMER_COMMON_PATTERN_GENERATOR_HPP
#define APPLICATIONS_GSTREAMER_COMMON_PATTERN_GENERATOR_HPP

#include <cstdint>
#include <memory>

#include <holoscan/holoscan.hpp>

namespace holoscan {

// Pattern types
enum class PatternType {
  Gradient = 0,
  Checkerboard = 1,
  ColorBars = 2
};

// Video format constants
constexpr int RGBA_CHANNELS = 4;
constexpr int RGB_CHANNELS = 3;
constexpr uint8_t ALPHA_OPAQUE = 255;

// Pattern-specific constants
constexpr int CHECKERBOARD_BASE_SIZE = 64;
constexpr int CHECKERBOARD_VARIATION = 32;
constexpr int SMPTE_COLOR_BARS = 7;

// Animation constants
constexpr float GRADIENT_TIME_STEP = 0.02f;
constexpr float CHECKERBOARD_TIME_STEP = 0.05f;

/**
 * @brief Get pattern name from pattern type enum
 * @param pattern Pattern type enum
 * @return Human-readable pattern name
 */
const char* get_pattern_name(PatternType pattern);

/**
 * @brief Get pattern name from pattern type integer
 * @param pattern Pattern type as integer
 * @return Human-readable pattern name
 */
const char* get_pattern_name(int pattern);

/**
 * @brief Helper function to set RGBA pixel values
 *
 * @param data Pointer to pixel data buffer
 * @param idx Index of the pixel (calculated as (y * width + x) * RGBA_CHANNELS)
 * @param r Red component (0-255)
 * @param g Green component (0-255)
 * @param b Blue component (0-255)
 * @param a Alpha component (0-255, default is opaque)
 */
inline void set_rgba_pixel(uint8_t* data, int idx, uint8_t r, uint8_t g, uint8_t b,
                           uint8_t a = ALPHA_OPAQUE) {
  data[idx + 0] = r;
  data[idx + 1] = g;
  data[idx + 2] = b;
  data[idx + 3] = a;
}

/**
 * @brief Abstract base class for pattern entity generation
 *
 * This class provides a template method pattern for generating video frame patterns.
 * Derived classes implement specific pattern generation logic.
 */
class PatternEntityGenerator {
 public:
  virtual ~PatternEntityGenerator() = default;

  /**
   * @brief Generate a pattern entity with animated content
   *
   * @param width Frame width in pixels
   * @param height Frame height in pixels
   * @param storage_type Memory storage type (host or device)
   * @param allocator Holoscan allocator for tensor memory
   * @return GXF entity containing the pattern tensor
   */
  holoscan::gxf::Entity generate(int width, int height,
                                 nvidia::gxf::MemoryStorageType storage_type,
                                 holoscan::Allocator* allocator);

 protected:
  /**
   * @brief Pure virtual function to generate pattern data
   *
   * Derived classes must implement this to provide specific pattern generation logic.
   *
   * @param data Pointer to the buffer where pattern should be written (RGBA format)
   * @param width Frame width in pixels
   * @param height Frame height in pixels
   */
  virtual void generate_pattern_data(uint8_t* data, int width, int height) = 0;
};

/**
 * @brief Gradient pattern generator with animated sine wave colors
 */
class GradientPatternGenerator : public PatternEntityGenerator {
 public:
  GradientPatternGenerator() : time_offset_(0.0f) {}

 protected:
  void generate_pattern_data(uint8_t* data, int width, int height) override;

 private:
  float time_offset_;  // Animation state for gradient
};

/**
 * @brief Checkerboard pattern generator with animated square size
 */
class CheckerboardPatternGenerator : public PatternEntityGenerator {
 public:
  CheckerboardPatternGenerator() : animation_time_(0.0f) {}

 protected:
  void generate_pattern_data(uint8_t* data, int width, int height) override;

 private:
  float animation_time_;  // Animation state for checkerboard
};

/**
 * @brief SMPTE color bars pattern generator (static pattern)
 */
class ColorBarsPatternGenerator : public PatternEntityGenerator {
 protected:
  void generate_pattern_data(uint8_t* data, int width, int height) override;
};

/**
 * PatternGenOperator - Generates pattern data as GXF entities with tensors
 */
class PatternGenOperator : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PatternGenOperator)

  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<int> width_;
  Parameter<int> height_;
  Parameter<int> pattern_;
  Parameter<int> storage_type_;
  std::unique_ptr<PatternEntityGenerator> generator_;  // Polymorphic pattern generator
};

}  // namespace holoscan

#endif  // APPLICATIONS_GSTREAMER_COMMON_PATTERN_GENERATOR_HPP
