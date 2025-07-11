/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <imgui.h>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoviz/holoviz.hpp>
#include <string>

#include <getopt.h>

namespace holoscan::ops {

class SourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SourceOp);

  void initialize() override {
    const int32_t width = 64, height = 64;
    shape_ = nvidia::gxf::Shape{width, height, 3};
    element_type_ = nvidia::gxf::PrimitiveType::kUnsigned8;
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
          data_[y * strides_[0] + x * strides_[1] + component] =
              uint8_t((rgb[component] * 255.f) + 0.5f);
        }
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
  std::vector<uint8_t> data_;
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
        // set the layer callback to execute a member function of the App class.
        Arg("layer_callback",
            ops::HolovizOp::LayerCallbackFunction(
                std::bind(&App::layer_callback, this, std::placeholders::_1))),
        Arg("window_title", std::string("Holoviz UI")),
        Arg("cuda_stream_pool", make_resource<CudaStreamPool>("cuda_stream_pool", 0, 0, 0, 1, 5)));

    add_flow(source, holoviz, {{"output", "receivers"}});
  }
  void layer_callback(const std::vector<holoscan::gxf::Entity>& inputs) {
    using namespace holoscan;

    // The layer callback is executed after the Holoviz operator has finished drawing all layers. We
    // now can add our own layers after that.
    // For more information on Holoviz layers see
    // https://docs.nvidia.com/holoscan/sdk-user-guide/visualization.html#layers.

    // Add a simple UI, Holoviz supports a `Dear ImGui` layer. For more information on `Dear ImGui`
    // check https://github.com/ocornut/imgui.
    viz::BeginImGuiLayer();
    ImGui::Begin("UI", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Checkbox("Checkbox", &checkbox_selected_);
    static const char* combo_items[] = {
        "Item 1",
        "Item 2",
        "Item 3",
    };
    ImGui::Combo("Combo", &combo_item_, combo_items, IM_ARRAYSIZE(combo_items));
    ImGui::SliderFloat("Slider Float", &slider_float_value_, 0.F, 1.F);
    ImGui::SliderInt("Slider Int", &slider_int_value_, -10, 10);
    ImGui::Separator();
    ImGui::ColorEdit4("Color", color_value_, ImGuiColorEditFlags_DefaultOptions_);
    viz::EndLayer();

    // Now create a geometry layer using the values of the UI.
    viz::BeginGeometryLayer();
    // Draw the text from combo with the color set by the user and the position set by the sliders.
    viz::Color(color_value_[0], color_value_[1], color_value_[2], color_value_[3]);
    viz::Text(slider_float_value_,
              float(slider_int_value_ + 10) / 20.F,
              checkbox_selected_ ? 0.1F : 0.05F,
              combo_items[combo_item_]);
    viz::EndLayer();
  }

 private:
  const int count_;
  bool checkbox_selected_ = false;
  int combo_item_ = 0;
  float slider_float_value_ = 0.F;
  int slider_int_value_ = 0;
  float color_value_[4]{1.F, 1.F, 1.F, 1.F};
};

int main(int argc, char** argv) {
  int count = -1;

  struct option long_options[] = {
      {"help", no_argument, 0, 'h'}, {"count", optional_argument, 0, 'c'}, {0, 0, 0, 0}};

  // parse options
  while (true) {
    int option_index = 0;

    const int c = getopt_long(argc, argv, "hc:", long_options, &option_index);

    if (c == -1) { break; }

    const std::string argument(optarg ? optarg : "");
    switch (c) {
      case 'h':
        std::cout << "Holoviz UI" << std::endl
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
