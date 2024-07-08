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

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <string>

#include <getopt.h>

#include "volume_loader.hpp"
#include "volume_renderer.hpp"

/**
 * Dummy YAML convert function for shared data type
 */
template <>
struct YAML::convert<std::shared_ptr<std::array<float, 16>>> {
  static Node encode(const std::shared_ptr<std::array<float, 16>>& data) {
    holoscan::log_error("YAML conversion not supported");
    return Node();
  }

  static bool decode(const Node& node, std::shared_ptr<std::array<float, 16>>& data) {
    holoscan::log_error("YAML conversion not supported");
    return false;
  }
};

namespace holoscan::ops {

/**
 * Source Op for typed shared data.
 */
template <typename T>
class SharedDataSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SharedDataSourceOp);

  void initialize() override {
    register_converter<T>();
    Operator::initialize();
  }

  void setup(OperatorSpec& spec) override {
    spec.param(shared_, "shared", "shared", "Shared Type");
    spec.output<T>("out");
  }

  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override {
    output.emit(shared_.get(), "out");
  }

 private:
  Parameter<T> shared_;
};

/**
 * Sink Op for typed shared data.
 */
template <typename T>
class SharedDataSinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SharedDataSinkOp);

  void setup(OperatorSpec& spec) override {
    spec.param(shared_, "shared", "shared", "Shared Type");
    spec.input<T>("in");
  }

  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override {
    auto message = input.receive<T>("in");
    *shared_.get() = *message.value();
  }

 private:
  Parameter<T> shared_;
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  App(const std::string& render_config_file, const std::string& density_volume_file,
      const std::string& mask_volume_file, int count)
      : render_config_file_(render_config_file),
        density_volume_file_(density_volume_file),
        mask_volume_file_(mask_volume_file),
        count_(count) {}
  App() = delete;

  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Resource> allocator = make_resource<UnboundedAllocator>("allocator");

    auto density_volume_loader =
        make_operator<ops::VolumeLoaderOp>("density_volume_loader",
                                           Arg("file_name", density_volume_file_),
                                           Arg("allocator", allocator),
                                           // the loader will executed only once to load the volume
                                           make_condition<CountCondition>("count-condition", 1));

    std::shared_ptr<ops::VolumeLoaderOp> mask_volume_loader;
    if (!mask_volume_file_.empty()) {
      mask_volume_loader = make_operator<ops::VolumeLoaderOp>(
          "mask_volume_loader",
          Arg("file_name", mask_volume_file_),
          Arg("allocator", allocator),
          // the loader will executed only once to load the volume
          make_condition<CountCondition>("count-condition", 1));
    }

    auto volume_renderer =
        make_operator<ops::VolumeRendererOp>("volume_renderer",
                                             Arg("config_file", render_config_file_),
                                             Arg("allocator", allocator),
                                             Arg("alloc_width", 1024u),
                                             Arg("alloc_height", 768u));

    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
        // stop application after short duration when testing
        make_condition<CountCondition>(count_),
        Arg("window_title", std::string("Volume Rendering with ClaraViz")),
        Arg("enable_camera_pose_output", true));

    // volume data loader
    add_flow(density_volume_loader,
             volume_renderer,
             {
                 {"volume", "density_volume"},
                 {"spacing", "density_spacing"},
                 {"permute_axis", "density_permute_axis"},
                 {"flip_axes", "density_flip_axes"},
             });

    if (mask_volume_loader) {
      add_flow(mask_volume_loader,
               volume_renderer,
               {
                   {"volume", "mask_volume"},
                   {"spacing", "mask_spacing"},
                   {"permute_axis", "mask_permute_axis"},
                   {"flip_axes", "mask_flip_axes"},
               });
    }

    add_flow(volume_renderer, holoviz, {{"color_buffer_out", "receivers"}});

    // Holoscan currently does not support graphs with cycles. The cycle is created by outputting
    // the rendered frame from the volume renderer to Holoviz and outputting the camera pose from
    // Holoviz to the volume renderer. To break the cycle create a shared camera pose. The
    // SharedDataSourceOp outputs the camera pose to the volume renderer and the SharedDataSinkOp
    // updates the shared camera pose with the data coming from Holoviz.
    auto camera_pose = std::make_shared<std::array<float, 16>>();
    for (uint32_t row = 0; row < 4; ++row) {
      for (uint32_t col = 0; col < 4; ++col) {
        camera_pose->at(col + row * 4) = (row == col) ? 1.f : 0.f;
      }
    }
    auto shared_source =
        make_operator<ops::SharedDataSourceOp<std::shared_ptr<std::array<float, 16>>>>(
            "camera_source", Arg("shared", camera_pose));
    auto shared_sink = make_operator<ops::SharedDataSinkOp<std::shared_ptr<std::array<float, 16>>>>(
        "camera_sink", Arg("shared", camera_pose));

    add_flow(shared_source, volume_renderer, {{"out", "camera_matrix"}});
    add_flow(holoviz, shared_sink, {{"camera_pose_output", "in"}});
  }

  const std::string render_config_file_;
  const std::string density_volume_file_;
  const std::string mask_volume_file_;
  const int count_;
};

int main(int argc, char** argv) {
  const std::string render_config_file_default("../../../data/volume_rendering/config.json");
  const std::string density_volume_file_default("../../../data/volume_rendering/highResCT.mhd");
  const std::string mask_volume_file_default("../../../data/volume_rendering/smoothmasks.seg.mhd");

  std::string render_config_file;
  std::string density_volume_file;
  std::string mask_volume_file;
  int count = -1;

  struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                  {"usages", no_argument, 0, 'u'},
                                  {"config", required_argument, 0, 'c'},
                                  {"density", required_argument, 0, 'd'},
                                  {"mask", required_argument, 0, 'm'},
                                  {"count", optional_argument, 0, 'n'},
                                  {0, 0, 0, 0}};

  // parse options
  while (true) {
    int option_index = 0;

    const int c = getopt_long(argc, argv, "huc:d:m:n:", long_options, &option_index);

    if (c == -1) { break; }

    const std::string argument(optarg ? optarg : "");
    switch (c) {
      case 'h':
      case 'u':
        std::cout << "Holoscan ClaraViz volume renderer."
                  << "Usage: " << argv[0] << " [options]" << std::endl
                  << "Options:" << std::endl
                  << "  -h,-u, --help, --usages               Display this information" << std::endl
                  << "  -c <FILENAME>, --config <FILENAME>    Name of the renderer JSON "
                     "configuration file to load (default '"
                  << render_config_file_default << "')" << std::endl
                  << "  -d <FILENAME>, --density <FILENAME>   Name of density volume file to load "
                     "(default '"
                  << density_volume_file_default << "')" << std::endl
                  << "  -m <FILENAME>, --mask <FILENAME>      Name of mask volume file to load "
                     "(default '"
                  << mask_volume_file_default << "')" << std::endl
                  << "  -n <COUNT>, --count <COUNT>           Duration to run application "
                     "(default '-1' for unlimited duration)"
                  << std::endl;
        return 0;

      case 'c':
        render_config_file = argument;
        break;

      case 'd':
        density_volume_file = argument;
        break;

      case 'm':
        mask_volume_file = argument;
        break;

      case 'n':
        count = stoi(argument);
        break;

      case '?':
        // unknown option, error already printed by getop_long
        break;
      default:
        holoscan::log_error("Unhandled option '{}'", static_cast<char>(c));
    }
  }

  if (render_config_file.empty()) { render_config_file = render_config_file_default; }
  if (density_volume_file.empty()) {
    density_volume_file = density_volume_file_default;
    mask_volume_file = mask_volume_file_default;
  }

  auto app = holoscan::make_application<App>(
      render_config_file, density_volume_file, mask_volume_file, count);
  app->run();

  holoscan::log_info("Application has finished running.");
  return 0;
}
