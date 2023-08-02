/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <GL/glew.h>

#include <string>

#include "gxf/std/codelet.hpp"

namespace holoscan::orsi {

struct VideoFrame {
  // owned internally
  GLuint vao_ = 0;
  GLuint vertex_shader_ = 0, fragment_shader_ = 0, program_ = 0;

  GLuint surgical_video_sampler_ = 0;
  GLuint preop_mesh_sampler_ = 0;
  GLuint surgical_tool_mask_sampler_ = 0;

  VideoFrame() {}

  VideoFrame(const VideoFrame&) = delete;
  VideoFrame& operator=(const VideoFrame&) = delete;

  gxf_result_t start(bool swizzleVideo);
  gxf_result_t tick(GLuint video_tex, GLenum video_tex_filter, GLuint preop_mesh_tex,
                    GLenum preop_mesh_filter, GLuint tool_mask_tex, GLenum tool_mask_filter,
                    bool applyInstrumentOverlayEffect, bool applyAnonymizationEffect);
  gxf_result_t stop();
};

}  // namespace holoscan::orsi
