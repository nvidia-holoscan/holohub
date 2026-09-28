// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace camera_inference {

inline constexpr std::string_view kUsage = R"(Usage: v4l2_inference_holoviz [options]
  --model PATH          ONNX model: one FP32 NCHW input/output, RGB values in [0,1]
  --device PATH         V4L2 YUYV camera (default /dev/video0)
  --width N             Camera width, even (default 640)
  --height N            Camera height (default 480)
  --fps N               Camera frame rate (default 30)
  --model-width N       Model input/output width (default 256)
  --model-height N      Model input/output height (default 256)
  --frames N            Inferred images to emit toward Holoviz (default 120)
  --timeout N           Run deadline in seconds, including startup (default 120)
  --validate            Compile the graph without opening the camera/model/window
  --help                Show this message
)";

struct Options {
  std::string model{"data/v4l2_inference_holoviz/identity_model.onnx"};
  std::string device{"/dev/video0"};
  int width{640};
  int height{480};
  int fps{30};
  int model_width{256};
  int model_height{256};
  int frames{120};
  int timeout{120};
  bool validate{false};
  bool help{false};
};

inline int positive_integer(std::string_view text, std::string_view option, int maximum) {
  int value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size() || value <= 0 || value > maximum) {
    throw std::invalid_argument(std::string(option) + " expects an integer in [1," +
                                std::to_string(maximum) + "]");
  }
  return value;
}

inline Options parse_options(int argc, const char* const* argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    if (arg == "--help") {
      options.help = true;
    } else if (arg == "--validate") {
      options.validate = true;
    } else {
      if (i + 1 == argc) {
        throw std::invalid_argument("Missing value for " + std::string{arg});
      }
      const std::string_view value{argv[++i]};
      if (value.empty() || value.starts_with("--")) {
        throw std::invalid_argument("Missing value for " + std::string{arg});
      }
      if (arg == "--model") {
        options.model = value;
      } else if (arg == "--device") {
        options.device = value;
      } else if (arg == "--width") {
        options.width = positive_integer(value, arg, 8192);
      } else if (arg == "--height") {
        options.height = positive_integer(value, arg, 8192);
      } else if (arg == "--fps") {
        options.fps = positive_integer(value, arg, 1000);
      } else if (arg == "--model-width") {
        options.model_width = positive_integer(value, arg, 4096);
      } else if (arg == "--model-height") {
        options.model_height = positive_integer(value, arg, 4096);
      } else if (arg == "--frames") {
        options.frames = positive_integer(value, arg, 1000000);
      } else if (arg == "--timeout") {
        options.timeout = positive_integer(value, arg, 86400);
      } else {
        throw std::invalid_argument("Unknown option: " + std::string{arg});
      }
    }
  }
  if (options.width % 2 != 0) {
    throw std::invalid_argument("--width must be even for packed YUYV capture");
  }
  return options;
}

}  // namespace camera_inference
