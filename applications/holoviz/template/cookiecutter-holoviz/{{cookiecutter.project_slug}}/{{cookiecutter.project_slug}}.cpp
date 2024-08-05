/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
{%- set example = namespace(print_fps = false) %}
{%- if cookiecutter.example == "vsync" %}
  {% set example.print_fps = true %}
{%- endif %}


#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

{%- if cookiecutter.example == "vsync" %}
#include <chrono>
{%- endif %}
#include <string>

#include <getopt.h>

namespace holoscan::ops {

class SourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SourceOp);

  void initialize() override {
    shape_ = nvidia::gxf::Shape{64, 64, 3};
    element_type_ = nvidia::gxf::PrimitiveType::kUnsigned8;
    element_size_ = nvidia::gxf::PrimitiveTypeSize(element_type_);
    strides_ = nvidia::gxf::ComputeTrivialStrides(shape_, element_size_);

    data_.resize(strides_[0] * shape_.dimension(0));

    // create an RGB image with random values
    for (size_t y = 0; y < shape_.dimension(0); ++y) {
      for (size_t x = 0; x < shape_.dimension(1); ++x) {
        for (size_t component = 0; component < shape_.dimension(2); ++component) {
          float value;
          switch (component) {
            case 0:
              value = float(x) / shape_.dimension(1);
              break;
            case 1:
              value = float(y) / shape_.dimension(0);
              break;
            case 2:
              value = 1.f - (float(x) / shape_.dimension(1));
              break;
            default:
              value = 1.f;
              break;
          }
{%- if cookiecutter.example == "sRGB" %}

          // inverse sRGB EOTF conversion from linear to non-linear
          // https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html
          if (value < 0.04045f) {
            value /= 12.92f;
          } else {
            value = std::pow(((value + 0.055f) / 1.055f), 2.4f);
          }
{%- endif %}

          data_[y * strides_[0] + x * strides_[1] + component] = uint8_t((value * 255.f) + 0.5f);
        }
      }
    }

    Operator::initialize();
  }

  void setup(OperatorSpec& spec) override { spec.output<holoscan::gxf::Entity>("output"); }

  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override {
{%- if example.print_fps %}
    if (start_.time_since_epoch().count() == 0) {
      start_ = std::chrono::steady_clock::now();
    }

{% endif -%}

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
{%- if example.print_fps %}

    iterations_++;
    const std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_);
    if (elapsed.count() > 1000) {
      const float fps =
          static_cast<float>(iterations_) / (static_cast<float>(elapsed.count()) / 1000.f);
      HOLOSCAN_LOG_INFO("Frames per second {}", fps);
      start_ = std::chrono::steady_clock::now();
      iterations_ = 0;
    }
{%- endif %}
  }

 private:
  nvidia::gxf::Shape shape_;
  nvidia::gxf::PrimitiveType element_type_;
  uint64_t element_size_;
  nvidia::gxf::Tensor::stride_array_t strides_;
  std::vector<uint8_t> data_;

{%- if example.print_fps %}
  std::chrono::steady_clock::time_point start_;
  uint32_t iterations_ = 0;
{%- endif %}
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
{%- if cookiecutter.example == "sRGB" %}

    // By default the image format is auto detected. Auto detection assumes linear color space,
    // but we provide an sRGB encoded image. Create an input spec and change the image format to
    // sRGB.
    ops::HolovizOp::InputSpec input_spec("image", ops::HolovizOp::InputType::COLOR);
    input_spec.image_format_ = ops::HolovizOp::ImageFormat::R8G8B8_SRGB;
{%- endif %}

    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
{%- if cookiecutter.example == "sRGB" %}
        Arg("tensors", std::vector<ops::HolovizOp::InputSpec>{input_spec}),
        // enable the sRGB frame buffer
        Arg("framebuffer_srgb", true),
{%- endif %}
{%- if cookiecutter.example == "vsync" %}
        // enable synchronization to vertical blank
        Arg("vsync", true),
{%- endif %}
        Arg("window_title", std::string("{{ cookiecutter.project_name }}")),
        Arg("cuda_stream_pool", make_resource<CudaStreamPool>("cuda_stream_pool", 0, 0, 0, 1, 5)));
{%- raw %}

    add_flow(source, holoviz, {{"output", "receivers"}});
{%- endraw %}
  }

  const int count_;
};

int main(int argc, char** argv) {
  int count = -1;

{%- raw %}

  struct option long_options[] = {
      {"help", no_argument, 0, 'h'}, {"count", optional_argument, 0, 'c'}, {0, 0, 0, 0}};
{%- endraw %}

  // parse options
  while (true) {
    int option_index = 0;

    const int c = getopt_long(argc, argv, "hc:", long_options, &option_index);

    if (c == -1) { break; }

    const std::string argument(optarg ? optarg : "");
    switch (c) {
      case 'h':
        std::cout << "Holoscan ClaraViz volume renderer."
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
