// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "image_conversion.hpp"
#include "options.hpp"

namespace {

void check(bool value, const char* message) {
  if (!value) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("Expected invalid_argument");
}

void test_options() {
  const char* defaults[]{"app"};
  const auto default_options = camera_inference::parse_options(1, defaults);
  check(default_options.model.ends_with("/identity_model.onnx") &&
            default_options.model_width == 256 && default_options.model_height == 256,
        "Defaults must match the SDK identity model");
  const char* valid[]{"app",
                      "--device",
                      "/dev/video2",
                      "--width",
                      "1280",
                      "--height",
                      "720",
                      "--model-width",
                      "16",
                      "--model-height",
                      "8",
                      "--frames",
                      "5",
                      "--validate"};
  const auto options = camera_inference::parse_options(14, valid);
  check(options.device == "/dev/video2" && options.width == 1280 && options.height == 720 &&
            options.model_width == 16 && options.model_height == 8 && options.frames == 5 &&
            options.validate,
        "Options were not applied");
  for (const char* value : {"0", "-1", "1junk", "999999999999999999999", "abc"}) {
    const char* args[]{"app", "--frames", value};
    rejects([&] { camera_inference::parse_options(3, args); });
  }
  for (const char* flag : {"--width", "--height", "--model", "--device", "--unknown"}) {
    const char* args[]{"app", flag};
    rejects([&] { camera_inference::parse_options(2, args); });
  }
  const char* odd[]{"app", "--width", "641"};
  rejects([&] { camera_inference::parse_options(3, odd); });
  const char* missing[]{"app", "--model", "--validate"};
  rejects([&] { camera_inference::parse_options(3, missing); });
}

void test_conversion() {
  // Distinct RGB channels catch channel permutation, rounding, and clipping.
  const std::array<float, 6> rgb{-1.0F, 1.0F, 0.0F, 2.0F, 0.5F, 0.25F};
  std::array<std::uint8_t, 6> bytes{};
  camera_inference::planar_rgb_float_to_u8(rgb, bytes);
  check(bytes == std::array<std::uint8_t, 6>{0, 0, 128, 255, 255, 64}, "RGB conversion failed");
  for (float invalid :
       {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
    for (std::size_t channel = 0; channel < 3; ++channel) {
      std::array<float, 3> values{};
      values[channel] = invalid;
      std::array<std::uint8_t, 3> output{};
      rejects([&] { camera_inference::planar_rgb_float_to_u8(values, output); });
    }
  }
  rejects([&] { camera_inference::planar_rgb_float_to_u8(rgb, std::span{bytes}.first(3)); });
  rejects([&] {
    camera_inference::planar_rgb_float_to_u8(std::span{rgb}.first(2), std::span{bytes}.first(2));
  });

  const std::array<std::int64_t, 4> nchw{1, 3, 2, 4};
  const std::array<std::int64_t, 4> contiguous{96, 32, 16, 4};
  check(camera_inference::validate_rgb_layout(nchw, contiguous, 4, 2) == 24, "NCHW rejected");
  check(camera_inference::validate_rgb_layout(nchw, {}, 4, 2) == 24, "Packed NCHW rejected");
  const std::array<std::int64_t, 4> nhwc{1, 2, 4, 3};
  rejects([&] { camera_inference::validate_rgb_layout(nhwc, {}, 4, 2); });
  const std::array<std::int64_t, 4> padded{192, 64, 32, 4};
  rejects([&] { camera_inference::validate_rgb_layout(nchw, padded, 4, 2); });
  rejects([&] { camera_inference::validate_rgb_layout(nchw, contiguous, 2, 4); });
}

}  // namespace

int main() {
  try {
    test_options();
    test_conversion();
    std::cout << "Options and image conversion checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
