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
#include "video_frame.hpp"
#include <holoscan/logger/logger.hpp>

#include <string>

#include "opengl_utils.hpp"

const std::string vertex_shader_source = // NOLINT
#include "glsl/viewport_filling_triangle.vert"
    ; // NOLINT

const std::string fragment_shader_source = // NOLINT
#include "glsl/video_frame.frag"
    ; // NOLINT

namespace holoscan::orsi {

gxf_result_t VideoFrame::start(bool swizzleVideo) {
  glGenVertexArrays(1, &vao_);

  glCreateSamplers(1, &surgical_video_sampler_);
  glSamplerParameteri(surgical_video_sampler_, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glSamplerParameteri(surgical_video_sampler_, GL_TEXTURE_WRAP_T, GL_REPEAT);

  glCreateSamplers(1, &preop_mesh_sampler_);
  glSamplerParameteri(preop_mesh_sampler_, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glSamplerParameteri(preop_mesh_sampler_, GL_TEXTURE_WRAP_T, GL_REPEAT);

  glCreateSamplers(1, &surgical_tool_mask_sampler_);
  glSamplerParameteri(surgical_tool_mask_sampler_, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glSamplerParameteri(surgical_tool_mask_sampler_, GL_TEXTURE_WRAP_T, GL_REPEAT);

  if (!createGLSLShader(GL_VERTEX_SHADER, vertex_shader_, vertex_shader_source.c_str())) {
    HOLOSCAN_LOG_ERROR("Failed to create GLSLvertex shader");
    return GXF_FAILURE;
  }

  if (!createGLSLShader(GL_FRAGMENT_SHADER, fragment_shader_, fragment_shader_source.c_str())) {
    HOLOSCAN_LOG_ERROR("Failed to create GLSL fragment shader");
    return GXF_FAILURE;
  }

  if (!linkGLSLProgram(vertex_shader_, fragment_shader_, program_)) {
    HOLOSCAN_LOG_ERROR("Failed to link GLSL program.");
    return GXF_FAILURE;
  }

  // apply swizzleVideo from at config time
  glProgramUniform1i(program_, 1, swizzleVideo);

  HOLOSCAN_LOG_INFO("Build GLSL shaders and program successfully");
  return GXF_SUCCESS;
}

gxf_result_t VideoFrame::tick(GLuint video_tex, GLenum video_tex_filter, GLuint preop_mesh_tex,
                              GLenum preop_mesh_filter, GLuint tool_mask_tex,
                              GLenum tool_mask_filter, bool applyInstrumentOverlayEffect,
                              bool applyAnonymizationEffect) {
  glActiveTexture(GL_TEXTURE0);

  glBindSampler(0, surgical_video_sampler_);
  glSamplerParameteri(surgical_video_sampler_, GL_TEXTURE_MIN_FILTER, video_tex_filter);
  glSamplerParameteri(surgical_video_sampler_, GL_TEXTURE_MAG_FILTER, video_tex_filter);
  glBindTexture(GL_TEXTURE_2D, video_tex);

  glActiveTexture(GL_TEXTURE1);
  glBindSampler(1, preop_mesh_sampler_);
  glSamplerParameteri(preop_mesh_sampler_, GL_TEXTURE_MIN_FILTER, preop_mesh_filter);
  glSamplerParameteri(preop_mesh_sampler_, GL_TEXTURE_MAG_FILTER, preop_mesh_filter);
  glBindTexture(GL_TEXTURE_2D, preop_mesh_tex);

  glActiveTexture(GL_TEXTURE2);
  glBindSampler(2, surgical_tool_mask_sampler_);
  glSamplerParameteri(surgical_tool_mask_sampler_, GL_TEXTURE_MIN_FILTER, tool_mask_filter);
  glSamplerParameteri(surgical_tool_mask_sampler_, GL_TEXTURE_MAG_FILTER, tool_mask_filter);
  glBindTexture(GL_TEXTURE_2D, tool_mask_tex);

  glUseProgram(program_);
  glProgramUniform1i(program_, 2, applyInstrumentOverlayEffect);
  glProgramUniform1i(program_, 3, applyAnonymizationEffect);

  glBindVertexArray(vao_);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 3);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, 0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, 0);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, 0);

  return GXF_SUCCESS;
}

gxf_result_t VideoFrame::stop() {
  return GXF_SUCCESS;
}

}  // namespace holoscan::orsi
