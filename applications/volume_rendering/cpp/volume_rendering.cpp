/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include "json_loader.hpp"
#include "volume_loader.hpp"
#include "volume_renderer.hpp"

class App : public holoscan::Application {
 public:
  App(const std::string& render_config_file, const std::vector<std::string>& render_preset_files,
      const std::string& write_config_file, const std::string& density_volume_file,
      const std::optional<float>& density_min, const std::optional<float>& density_max,
      const std::string& mask_volume_file, int count)
      : render_config_file_(render_config_file),
        render_preset_files_(render_preset_files),
        write_config_file_(write_config_file),
        density_volume_file_(density_volume_file),
        density_min_(density_min),
        density_max_(density_max),
        mask_volume_file_(mask_volume_file),
        count_(count) {}
  App() = delete;

  void compose() override {
    using namespace holoscan;

    const std::shared_ptr<Resource> allocator = make_resource<UnboundedAllocator>("allocator");
    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

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

    std::shared_ptr<ops::JsonLoaderOp> preset_loader;
    if (!render_preset_files_.empty()) {
      preset_loader =
          make_operator<ops::JsonLoaderOp>("preset_loader",
                                           Arg("file_names", render_preset_files_),
                                           // the loader will executed only once to load the presets
                                           make_condition<CountCondition>("count-condition", 1));
    }

    ArgList volume_renderer_optional_args;
    if (density_min_.has_value()) {
      volume_renderer_optional_args.add(Arg("density_min", density_min_.value()));
    }
    if (density_max_.has_value()) {
      volume_renderer_optional_args.add(Arg("density_max", density_max_.value()));
    }
    auto volume_renderer =
        make_operator<ops::VolumeRendererOp>("volume_renderer",
                                             Arg("config_file", render_config_file_),
                                             Arg("write_config_file", write_config_file_),
                                             Arg("allocator", allocator),
                                             Arg("alloc_width", 1024u),
                                             Arg("alloc_height", 768u),
                                             Arg("cuda_stream_pool", cuda_stream_pool),
                                             volume_renderer_optional_args);

    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
        // stop application after short duration when testing
        make_condition<CountCondition>(count_),
        Arg("window_title", std::string("Volume Rendering with ClaraViz")),
        Arg("enable_camera_pose_output", true),
        Arg("cuda_stream_pool", cuda_stream_pool));

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

    if (preset_loader) {
      add_flow(preset_loader, volume_renderer, {{"json", "merge_settings"}});
      // Since the preset_loader is only triggered once we have to set the input condition of
      // the merge_settings ports to ConditionType::kNone.
      // Currently, there is no API to set the condition of the receivers so we have to do this
      // after connecting the ports
      auto& inputs = volume_renderer->spec()->inputs();
      auto input = inputs.find("merge_settings:0");
      if (input == inputs.end()) {
        throw std::runtime_error("Could not find `merge_settings:0` input");
      }
      input->second->condition(ConditionType::kNone);
    }

    add_flow(volume_renderer, holoviz, {{"color_buffer_out", "receivers"}});
    add_flow(holoviz, volume_renderer, {{"camera_pose_output", "camera_pose"}});
  }

  const std::string render_config_file_;
  const std::vector<std::string> render_preset_files_;
  const std::string write_config_file_;
  const std::string density_volume_file_;
  const std::optional<float> density_min_;
  const std::optional<float> density_max_;
  const std::string mask_volume_file_;
  const int count_;
};

int main(int argc, char** argv) {
  const std::string render_config_file_default("../../../data/volume_rendering/config.json");
  const std::string density_volume_file_default("../../../data/volume_rendering/highResCT.mhd");
  const std::string mask_volume_file_default("../../../data/volume_rendering/smoothmasks.seg.mhd");

  std::string render_config_file(render_config_file_default);
  std::vector<std::string> render_preset_files;
  std::string write_config_file;
  std::string density_volume_file;
  std::optional<float> density_min;
  std::optional<float> density_max;
  std::string mask_volume_file;
  int count = -1;

  struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                  {"usages", no_argument, 0, 'u'},
                                  {"config", required_argument, 0, 'c'},
                                  {"preset", required_argument, 0, 'p'},
                                  {"write_config", required_argument, 0, 'w'},
                                  {"density", required_argument, 0, 'd'},
                                  {"density_min", optional_argument, 0, 'i'},
                                  {"density_max", optional_argument, 0, 'a'},
                                  {"mask", required_argument, 0, 'm'},
                                  {"count", optional_argument, 0, 'n'},
                                  {0, 0, 0, 0}};

  // parse options
  while (true) {
    int option_index = 0;

    const int c = getopt_long(argc, argv, "huc:p:w:d:i:a:m:e", long_options, &option_index);

    if (c == -1) { break; }

    const std::string argument(optarg ? optarg : "");
    switch (c) {
      case 'h':
      case 'u':
        std::cout
            << "Holoscan ClaraViz volume renderer." << std::endl
            << "Usage: " << argv[0] << " [options]" << std::endl
            << "Options:" << std::endl
            << "  -h,-u, --help, --usages               Display this information" << std::endl
            << "  -c <FILENAME>, --config <FILENAME>    Name of the renderer JSON "
               "configuration file to load (default '"
            << render_config_file_default << "')" << std::endl
            << "  -p <FILENAME>, --preset <FILENAME>    Name of the renderer JSON "
               "preset file to load. This will be merged into the settings loaded from the "
               "configuration file. Multiple presets can be specified."
            << std::endl
            << "  -w <FILENAME>, --write_config <FILENAME> Name of the renderer JSON "
               "configuration file to write to (default '')"
            << std::endl
            << "  -d <FILENAME>, --density <FILENAME>   Name of density volume file to load "
               "(default '"
            << density_volume_file_default << "')" << std::endl
            << "  -i <MIN>, --density_min <MIN>         Set the minimum of the density element "
               "values. If not set this is calculated from the volume data. In practice CT "
               "volumes have a minimum value of -1024 which corresponds to the lower value "
               "of the Hounsfield scale range usually used."
            << std::endl
            << "  -a <MAX>, --density_max <MAX>         Set the maximum of the density element "
               "values. If not set this is calculated from the volume data. In practice CT "
               "volumes have a maximum value of 3071 which corresponds to the upper value "
               "of the Hounsfield scale range usually used."
            << std::endl
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

      case 'p':
        render_preset_files.push_back(argument);
        break;

      case 'w':
        write_config_file = argument;
        break;

      case 'd':
        density_volume_file = argument;
        break;

      case 'i':
        density_min = std::stof(argument);
        break;

      case 'a':
        density_max = std::stof(argument);
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

  if (density_volume_file.empty()) {
    density_volume_file = density_volume_file_default;
    mask_volume_file = mask_volume_file_default;
  }

  auto app = holoscan::make_application<App>(render_config_file,
                                             render_preset_files,
                                             write_config_file,
                                             density_volume_file,
                                             density_min,
                                             density_max,
                                             mask_volume_file,
                                             count);
  app->run();

  holoscan::log_info("Application has finished running.");
  return 0;
}
