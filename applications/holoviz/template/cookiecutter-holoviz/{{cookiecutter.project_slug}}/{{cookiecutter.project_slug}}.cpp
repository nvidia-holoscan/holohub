/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
{%- set example = namespace(print_fps = false, input_spec = false) %}
{%- if cookiecutter.example == "sRGB" %}
  {% set example.input_spec = true %}
{%- elif cookiecutter.example == "vsync" %}
  {% set example.print_fps = true %}
{%- elif cookiecutter.example == "YUV" %}
  {% set example.input_spec = true %}
{%- endif %}


#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

{%- if example.print_fps %}
#include <chrono>
{%- endif %}
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

{%- if cookiecutter.example == "YUV" %}

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
          yuv_data_[uv_color_plane.offset +
                  (y / 2) * uv_color_plane.stride + (x / 2) * uv_color_plane.bytes_per_pixel + 0] =
              uint8_t(u * 255.f + 0.5f);
          yuv_data_[uv_color_plane.offset +
                  (y / 2) * uv_color_plane.stride + (x / 2) * uv_color_plane.bytes_per_pixel + 1] =
              uint8_t(v * 255.f + 0.5f);
        }
      }
    }

{%- endif %}

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
{%- if cookiecutter.example == "YUV" %}
    auto video_buffer =
        static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::VideoBuffer>("image");
    video_buffer.value()->wrapMemory(video_buffer_info_,
                                    yuv_data_.size(),
                                    nvidia::gxf::MemoryStorageType::kSystem,
                                    yuv_data_.data(),
                                    nullptr);
{% else -%}
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>("image");
    tensor.value()->wrapMemory(shape_,
                               element_type_,
                               element_size_,
                               strides_,
                               nvidia::gxf::MemoryStorageType::kSystem,
                               data_.data(),
                               nullptr);
{% endif -%}
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
{%- if cookiecutter.example == "YUV" %}
  std::vector<uint8_t> yuv_data_;
  nvidia::gxf::VideoBufferInfo video_buffer_info_{};
{%- endif %}

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
{%- if example.input_spec %}

    ops::HolovizOp::InputSpec input_spec("image", ops::HolovizOp::InputType::COLOR);
{%- endif %}
{%- if cookiecutter.example == "sRGB" %}

    // By default the image format is auto detected. Auto detection assumes linear color space,
    // but we provide an sRGB encoded image. Create an input spec and change the image format to
    // sRGB.
    input_spec.image_format_ = ops::HolovizOp::ImageFormat::R8G8B8_SRGB;
{%- elif cookiecutter.example == "YUV" %}

    // Set the YUV image format, model conversion and range for the input tensor.
    input_spec.image_format_ = ops::HolovizOp::ImageFormat::Y8_U8V8_2PLANE_420_UNORM;
    input_spec.yuv_model_conversion_ = ops::HolovizOp::YuvModelConversion::YUV_601;
    input_spec.yuv_range_ = ops::HolovizOp::YuvRange::ITU_FULL;

{%- endif %}

    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
{%- if example.input_spec %}
        Arg("tensors", std::vector<ops::HolovizOp::InputSpec>{input_spec}),
{%- endif %}
{%- if cookiecutter.example == "sRGB" %}
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
        std::cout << "{{ cookiecutter.project_name }}" << std::endl
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
