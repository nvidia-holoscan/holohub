/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <string>

#include <getopt.h>

namespace holoscan::ops {

class SourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SourceOp);

  void initialize() override {
    const int32_t width = 64, height = 64;
    shape_ = nvidia::gxf::Shape{width, height, 4};
    element_type_ = nvidia::gxf::PrimitiveType::kFloat32;
    element_size_ = nvidia::gxf::PrimitiveTypeSize(element_type_);
    strides_ = nvidia::gxf::ComputeTrivialStrides(shape_, element_size_);

    data_.resize(strides_[0] * shape_.dimension(0));

    // create an RGB image with smooth color transitions
    for (size_t y = 0; y < shape_.dimension(0); ++y) {
      for (size_t x = 0; x < shape_.dimension(1); ++x) {
        float rgb[3];
        for (size_t component = 0; component < 3; ++component) {
          switch (component) {
            case 0:
              rgb[component] = float(x) / shape_.dimension(1);
              break;
            case 1:
              rgb[component] = float(y) / shape_.dimension(0);
              break;
            case 2:
              rgb[component] = 1.f - (float(x) / shape_.dimension(1));
              break;
          }

          // create two regions, the top region has 100 nits
          // the bottom region starts at 100 nits and ends at 500 nits
          constexpr float max_luminance = 10000.f;
          if (y < height / 2) {
            rgb[component] *= 100.f / max_luminance;
          } else {
            rgb[component] *= (100.f + (float(x) / shape_.dimension(1)) * 500.f) / max_luminance;
          }
        }

        // use the RGB data to generate data in HDR10 (BT2020 color space) with SMPTE ST2084
        // Perceptual Quantizer (PQ) EOTF

        float rgb_2020[3];
        // linear to BT2020 color space conversion
        // https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#PRIMARIES_BT2020
        rgb_2020[0] = std::clamp(
            (0.636958f * rgb[0]) + (0.144617f * rgb[1]) + (0.168881f * rgb[2]), 0.f, 1.f);
        rgb_2020[1] = std::clamp(
            (0.262700f * rgb[0]) + (0.677998f * rgb[1]) + (0.059302f * rgb[2]), 0.f, 1.f);
        rgb_2020[2] = std::clamp(
            (0.000000f * rgb[0]) + (0.028073f * rgb[1]) + (1.060985f * rgb[2]), 0.f, 1.f);

        // apply inverse SMPTE ST2084 Perceptual Quantizer (PQ) EOTF
        constexpr float m1 = 2610.f / 16384.f;
        constexpr float m2 = 2523.f / 4096.f * 128.f;
        constexpr float c2 = 2413.f / 4096.f * 32.f;
        constexpr float c3 = 2392.f / 4096.f * 32.f;
        constexpr float c1 = c3 - c2 + 1.f;

        for (size_t component = 0; component < 3; ++component) {
          float lp = std::pow(rgb_2020[component], m1);
          float value = std::pow((c1 + c2 * lp) / (1.f + c3 * lp), m2);

          *reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(data_.data()) + y * strides_[0] +
                                    x * strides_[1] + component * strides_[2]) = value;
        }
        // alpha
        *reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(data_.data()) + y * strides_[0] +
                                  x * strides_[1] + 3 * strides_[2]) = 1.f;
      }
    }

    Operator::initialize();
  }

  void setup(OperatorSpec& spec) override { spec.output<holoscan::gxf::Entity>("output"); }

  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override {
    auto entity = holoscan::gxf::Entity::New(&context);
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>("image");
    tensor.value()->wrapMemory(shape_,
                               element_type_,
                               element_size_,
                               strides_,
                               nvidia::gxf::MemoryStorageType::kSystem,
                               data_.data(),
                               nullptr);
    output.emit(entity, "output");
  }

 private:
  nvidia::gxf::Shape shape_;
  nvidia::gxf::PrimitiveType element_type_;
  uint64_t element_size_;
  nvidia::gxf::Tensor::stride_array_t strides_;
  std::vector<float> data_;
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  explicit App(int count) : count_(count) {}
  App() = delete;

  void compose() override {
    using namespace holoscan;

    auto source =
        make_operator<ops::SourceOp>("source",
                                     // stop application count
                                     make_condition<CountCondition>("count-condition", count_));

    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
        // select the HDR10 ST2084 display color space
        Arg("display_color_space", ops::HolovizOp::ColorSpace::HDR10_ST2084),
        Arg("window_title", std::string("Holoviz HDR")),
        Arg("cuda_stream_pool", make_resource<CudaStreamPool>("cuda_stream_pool", 0, 0, 0, 1, 5)));

    add_flow(source, holoviz, {{"output", "receivers"}});
  }

 private:
  const int count_;
};

int main(int argc, char** argv) {
  int count = -1;

  struct option long_options[] = {
      {"help", no_argument, 0, 'h'}, {"count", optional_argument, 0, 'c'}, {0, 0, 0, 0}};

  // parse options
  while (true) {
    int option_index = 0;

    const int c = getopt_long(argc, argv, "hc:", long_options, &option_index);

    if (c == -1) {
      break;
    }

    const std::string argument(optarg ? optarg : "");
    switch (c) {
      case 'h':
        std::cout << "Holoviz HDR" << std::endl
                  << "Usage: " << argv[0] << " [options]" << std::endl
                  << "Options:" << std::endl
                  << "  -h, --help                    Display this information" << std::endl
                  << "  -c <COUNT>, --count <COUNT>   execute operators <COUNT> times (default "
                     "'-1' for unlimited)"
                  << std::endl;
        return 0;

      case 'c':
        count = stoi(argument);
        break;

      case '?':
        // unknown option, error already printed by getop_long
        break;
      default:
        holoscan::log_error("Unhandled option '{}'", static_cast<char>(c));
    }
  }

  auto app = holoscan::make_application<App>(count);
  app->run();

  holoscan::log_info("Application has finished running.");
  return 0;
}
