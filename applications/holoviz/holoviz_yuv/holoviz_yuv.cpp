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

    // use the RGB data to generate YUV 420 BT.601 extended range data

    // setup the video buffer info with the YUV color planes
    video_buffer_info_.width = width;
    video_buffer_info_.height = height;
    video_buffer_info_.color_format = nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_ER;
    video_buffer_info_.surface_layout = nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR;
    video_buffer_info_.color_planes =
        nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_ER>()
            .getDefaultColorPlanes(width, height, false /*stride_align*/);

    const nvidia::gxf::ColorPlane& y_color_plane = video_buffer_info_.color_planes[0];
    const nvidia::gxf::ColorPlane& uv_color_plane = video_buffer_info_.color_planes[1];

    yuv_data_.resize(y_color_plane.size + uv_color_plane.size);

    // color model conversion from RGB to YUV as defined in BT.601
    const float Kr = 0.299f;
    const float Kb = 0.114f;
    const float Kg = 1.f - Kb - Kr;

    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        const float r = data_[y * strides_[0] + x * strides_[1] + 0] / 255.f;
        const float g = data_[y * strides_[0] + x * strides_[1] + 1] / 255.f;
        const float b = data_[y * strides_[0] + x * strides_[1] + 2] / 255.f;

        float luma = Kr * r + Kg * g + Kb * b;  // 0 ... 1
        float u = (b - luma) / (1.f - Kb);      // -1 ... 1
        float v = (r - luma) / (1.f - Kr);      // -1 ... 1

        // ITU “full range” quantization rule
        u = u * 0.5f + 0.5f;
        v = v * 0.5f + 0.5f;

        yuv_data_[y * y_color_plane.stride + x] = uint8_t(luma * 255.f + 0.5f);
        if (((x & 1) == 0) && ((y & 1) == 0)) {
          yuv_data_[uv_color_plane.offset + (y / 2) * uv_color_plane.stride +
                    (x / 2) * uv_color_plane.bytes_per_pixel + 0] = uint8_t(u * 255.f + 0.5f);
          yuv_data_[uv_color_plane.offset + (y / 2) * uv_color_plane.stride +
                    (x / 2) * uv_color_plane.bytes_per_pixel + 1] = uint8_t(v * 255.f + 0.5f);
        }
      }
    }

    Operator::initialize();
  }

  void setup(OperatorSpec& spec) override { spec.output<holoscan::gxf::Entity>("output"); }

  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override {
    auto entity = holoscan::gxf::Entity::New(&context);
    auto video_buffer =
        static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::VideoBuffer>("image");
    video_buffer.value()->wrapMemory(video_buffer_info_,
                                     yuv_data_.size(),
                                     nvidia::gxf::MemoryStorageType::kSystem,
                                     yuv_data_.data(),
                                     nullptr);
    output.emit(entity, "output");
  }

 private:
  nvidia::gxf::Shape shape_;
  nvidia::gxf::PrimitiveType element_type_;
  uint64_t element_size_;
  nvidia::gxf::Tensor::stride_array_t strides_;
  std::vector<uint8_t> data_;
  std::vector<uint8_t> yuv_data_;
  nvidia::gxf::VideoBufferInfo video_buffer_info_{};
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

    ops::HolovizOp::InputSpec input_spec("image", ops::HolovizOp::InputType::COLOR);

    // Set the YUV image format, model conversion and range for the input tensor.
    input_spec.image_format_ = ops::HolovizOp::ImageFormat::Y8_U8V8_2PLANE_420_UNORM;
    input_spec.yuv_model_conversion_ = ops::HolovizOp::YuvModelConversion::YUV_601;
    input_spec.yuv_range_ = ops::HolovizOp::YuvRange::ITU_FULL;

    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
        Arg("tensors", std::vector<ops::HolovizOp::InputSpec>{input_spec}),
        Arg("window_title", std::string("Holoviz YUV")),
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
        std::cout << "Holoviz YUV" << std::endl
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
