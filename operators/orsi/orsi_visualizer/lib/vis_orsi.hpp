/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "vis_intf.hpp"

#include <cuda_gl_interop.h>

#include <string>
#include <vector>

// GXF 4.0 moved parameter_parser_std.hpp from std->core
#if __has_include("gxf/core/parameter_parser_std.hpp")
  #include "gxf/core/parameter_parser_std.hpp"
#else
  #include "gxf/std/parameter_parser_std.hpp"
#endif
#include "gxf/std/codelet.hpp"
#include "gxf/std/memory_buffer.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/scheduling_terms.hpp"

#include "video_frame.hpp"
#include "vtk_view.hpp"

struct cudaGraphicsResource;
namespace holoscan::orsi {

/// @brief Visualization Codelet for Instrument Tracking and Overlay
///
/// This visualizer uses OpenGL/CUDA interopt for quickly passing data from the output of inference
/// to an OpenGL context for rendering.
/// The visualizer renders the location and text of an instrument and optionally displays the
/// model's confidence score.
class OrsiVis : public holoscan::orsi::vis::VisIntf {
 public:
  static holoscan::orsi::vis::VisIntf* Create() { return new OrsiVis(); }

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void compute(const std::unordered_map<std::string,
                                        holoscan::orsi::vis::BufferInfo>& input_buffers) override;
  void stop() override;


  // GLFW window related callbacks
  void onFramebufferSizeCallback(GLFWwindow* wnd, int width, int height) override;
  void onChar(GLFWwindow* wnd, unsigned int codepoint);
  void onEnter(GLFWwindow* wnd, int entered);
  void onMouseMove(GLFWwindow* wnd, double x, double y);
  void onMouseButtonCallback(GLFWwindow* wnd, int button, int action, int mods);
  void onScrollCallback(GLFWwindow* wnd, double x, double y);
  void onKeyCallback(GLFWwindow* wnd, int key, int scancode, int action, int mods);

 private:
  // Videoframe Vis related members
  // --------------------------------------------------------------------
  bool use_cuda_opengl_interop_ = true;
  VideoFrame video_frame_vis_;

  Parameter<std::string> stl_file_path_;
  Parameter<std::string> registration_params_path_;
  Parameter<std::vector<std::string>> stl_names_;
  Parameter<std::vector<std::vector<int32_t>>> stl_colors_;
  Parameter<std::vector<int>> stl_keys_;
  Parameter<bool> swizzle_video_;

  int32_t video_frame_width_ = 0;
  int32_t video_frame_height_ = 0;
  int32_t video_frame_channels_ = 0;

  bool apply_tool_overlay_effect_ = true;
  bool apply_anonymization_effect_ = true;
  bool toggle_anonymization_ = true;

  void resizeVideoBufferResources(int width, int height, int channels);

  GLuint video_frame_tex_ = 0;
  cudaGraphicsResource* cuda_video_frame_tex_resource_ = nullptr;
  std::vector<unsigned char> video_frame_buffer_host_;

  // Segmentation Mask  related members
  // --------------------------------------------------------------------

  int32_t seg_mask_width_ = 0;
  int32_t seg_mask_height_ = 0;

  GLuint seg_mask_tex_ = 0;
  cudaGraphicsResource* cuda_seg_mask_tex_resource_ = nullptr;
  void resizeSegmentationMaskResources(int width, int height);

  // VTK  related members
  // --------------------------------------------------------------------

  VtkView vtk_view_;
  bool enable_model_manip_ = false;
};

}  // namespace holoscan::orsi
