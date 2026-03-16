/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/holoscan.hpp"

#include <cstdint>
#include <filesystem>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <iostream>
#include <string>

#include "pva_radar.hpp"  // from holoscan::pva_radar operator
#include "pva_radar_graphics.hpp"
#include "raw_radar_cube_source.hpp"

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto source = make_operator<ops::RawRadarCubeSourceOp>(
        "source", from_config("radar_source"), from_config("radar_pipeline"));

    auto pva_radar_pipeline =
        make_operator<ops::PVARadarPipelineOp>("pva_radar_pipeline", from_config("radar_pipeline"));

    auto pva_radar_graphics = make_operator<ops::PVARadarGraphicsOp>(
        "pva_radar_graphics",
        from_config("pva_radar_graphics"),
        Arg("allocator") = make_resource<holoscan::UnboundedAllocator>("graphics_allocator"));

    auto image_viz = make_operator<ops::HolovizOp>("image_viz", from_config("holoviz_rgb"));
    auto cloud_viz = make_operator<ops::HolovizOp>("cloud_viz", from_config("holoviz_xyz"));

    add_flow(source, pva_radar_pipeline, {{"output", "input"}});
    add_flow(pva_radar_pipeline,
             pva_radar_graphics,
             {{"output_nci", "nci"}, {"output_peak_count", "peak_count"}, {"output_doa", "doa"}});
    add_flow(pva_radar_graphics, image_viz, {{"output_image", "receivers"}});
    add_flow(pva_radar_graphics, cloud_viz, {{"output_xyz", "receivers"}});
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/main.yaml";
  app->config(config_path);

  app->run();

  return 0;
}
