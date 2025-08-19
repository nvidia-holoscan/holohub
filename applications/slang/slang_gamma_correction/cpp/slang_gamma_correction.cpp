/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <holoscan/utils/cuda_macros.hpp>

#include <gamma_correction/gamma_correction.hpp>

#include <getopt.h>

namespace holoscan::ops {

/**
 * This operator serves as a data source in the Holoscan pipeline, emitting
 * a 64x64x3 image with smooth color transitions.
 */
class SourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SourceOp);

  SourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(
        allocator_,
        "allocator",
        "Allocator for output buffers.",
        "Allocator for output buffers.",
        std::static_pointer_cast<Allocator>(fragment()->make_resource<RMMAllocator>("allocator")));
    spec.output<gxf::Entity>("output");
  }

  void initialize() override {
    // Add the allocator to the operator so that it is initialized
    add_arg(allocator_.default_value());

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

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto entity = gxf::Entity::New(&context);
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>().value();
    // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_.get()->gxf_cid());
    tensor->reshape<uint8_t>(
        shape_, nvidia::gxf::MemoryStorageType::kDevice, gxf_allocator.value());

    HOLOSCAN_CUDA_CALL(
        cudaMemcpy(tensor->pointer(), data_.data(), data_.size(), cudaMemcpyHostToDevice));

    op_output.emit(entity, "output");
  }

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  nvidia::gxf::Shape shape_;
  nvidia::gxf::PrimitiveType element_type_;
  uint64_t element_size_;
  nvidia::gxf::Tensor::stride_array_t strides_;
  std::vector<uint8_t> data_;
};

}  // namespace holoscan::ops

/**
 * @brief Application class for demonstrating gamma correction using SLANG shaders
 *
 * This application creates a workflow that:
 * 1. Generates a test image with smooth color transitions
 * 2. Applies gamma correction using a SLANG shader
 * 3. Displays the result in HoloViz with proper sRGB color space handling
 */
class SlangGammaCorrectionApp : public holoscan::Application {
 public:
  explicit SlangGammaCorrectionApp(int count) : count_(count) {}
  SlangGammaCorrectionApp() = delete;

  void compose() override {
    using namespace holoscan;

    auto allocator = make_resource<UnboundedAllocator>("pool");

    auto source =
        make_operator<ops::SourceOp>("source",
                                     // stop application count
                                     make_condition<CountCondition>("count-condition", count_));

    auto gamma_correction = make_operator<ops::GammaCorrectionOp>(
        "GammaCorrection", Arg("data_type", std::string("uint8_t")), Arg("component_count", 3));

    // By default the image format is auto detected. Auto detection assumes linear color space,
    // but we provide an sRGB encoded image. Create an input spec and change the image format to
    // sRGB.
    ops::HolovizOp::InputSpec input_spec("", ops::HolovizOp::InputType::COLOR);
    input_spec.image_format_ = ops::HolovizOp::ImageFormat::R8G8B8_SRGB;

    auto sink = make_operator<ops::HolovizOp>(
        "holoviz",
        Arg("window_title", std::string("Gamma Correction")),
        Arg("tensors", std::vector<ops::HolovizOp::InputSpec>{input_spec}),
        Arg("framebuffer_srgb", true));

    // Define the workflow
    add_flow(source, gamma_correction);
    add_flow(gamma_correction, sink, {{"output", "receivers"}});
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
        std::cout << "Slang Gamma Correction" << std::endl
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

  auto app = holoscan::make_application<SlangGammaCorrectionApp>(count);
  app->run();

  return 0;
}
