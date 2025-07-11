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
#include "opengl_utils.hpp"

#include <fstream>
#include <iterator>
#include <string>

#include "common/assert.hpp"
#include <holoscan/logger/logger.hpp>

#ifndef UNUSED
#define UNUSED(NAME) (void)(NAME)
#endif

// --------------------------------------------------------------------------------------
//
// OpenGL Debug Output Helpers
//

const char* glDebugSource2Str(GLenum source) {
  switch (source) {
    default:
      break;
    case GL_DEBUG_SOURCE_API:
      return "API";
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
      return "Window Sys";
    case GL_DEBUG_SOURCE_SHADER_COMPILER:
      return "Shader Compiler";
    case GL_DEBUG_SOURCE_THIRD_PARTY:
      return "3rdparty";
    case GL_DEBUG_SOURCE_APPLICATION:
      return "App";
    case GL_DEBUG_SOURCE_OTHER:
      return "Other";
  }
  return "Unknown";
}

const char* glDebugType2Str(GLenum type) {
  switch (type) {
    default:
      break;
    case GL_DEBUG_TYPE_ERROR:
      return "Error";
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
      return "Deprecated Behavior";
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
      return "Undefined Behavior";
    case GL_DEBUG_TYPE_PORTABILITY:
      return "Portability";
    case GL_DEBUG_TYPE_PERFORMANCE:
      return "Performance";
    case GL_DEBUG_TYPE_OTHER:
      return "Other";
  }
  return "Unknown";
}

const char* glDebugSeverity2Str(GLenum severity) {
  switch (severity) {
    default:
      break;
    case GL_DEBUG_SEVERITY_HIGH:
      return "High";
    case GL_DEBUG_SEVERITY_MEDIUM:
      return "Medium";
    case GL_DEBUG_SEVERITY_LOW:
      return "Low";
    case GL_DEBUG_SEVERITY_NOTIFICATION:
      return "Notification";
  }
  return "Unknown";
}

void OpenGLDebugMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
                                GLsizei length, const GLchar* message, const void* userParam) {
  UNUSED(id);
  UNUSED(length);
  UNUSED(userParam);

  const char* source_str = glDebugSource2Str(source);
  const char* type_str = glDebugType2Str(type);
  const char* severity_str = glDebugSeverity2Str(severity);

  if (severity == GL_DEBUG_TYPE_ERROR) {
    HOLOSCAN_LOG_ERROR("GL CALLBACK: source = %s, type = %s, severity = %s, message = %s\n",
                  source_str,
                  type_str,
                  severity_str,
                  message);
  } else {
    HOLOSCAN_LOG_INFO("GL CALLBACK: source = %s, type = %s, severity = %s, message = %s\n",
                 source_str,
                 type_str,
                 severity_str,
                 message);
  }
}

bool createGLSLShader(GLenum shader_type, GLuint& shader, const char* shader_src) {
  shader = glCreateShader(shader_type);
  glShaderSource(shader, 1, &shader_src, NULL);
  glCompileShader(shader);

  GLint isCompiled = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);

  if (isCompiled == GL_FALSE) {
    GLint maxLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

    // The maxLength includes the NULL character
    std::string compile_log;
    compile_log.resize(maxLength);
    glGetShaderInfoLog(shader, maxLength, NULL, &compile_log[0]);

    HOLOSCAN_LOG_ERROR("Shader compilation failed:  %s ", compile_log.c_str());
    return false;
  }
  return true;
}

bool linkGLSLProgram(const GLuint vertex_shader, const GLuint fragment_shader, GLuint& program) {
  program = glCreateProgram();
  glAttachShader(program, vertex_shader);
  glAttachShader(program, fragment_shader);
  glLinkProgram(program);

  GLint isLinked = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &isLinked);
  if (isLinked == GL_FALSE) {
    GLint maxLength = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

    // The maxLength includes the NULL character
    std::string link_log;
    link_log.resize(maxLength);
    glGetProgramInfoLog(program, maxLength, &maxLength, &link_log[0]);
    HOLOSCAN_LOG_ERROR("Failed to link GLSL program. Log: %s", link_log.c_str());
    // The program is useless now. So delete it.
    glDeleteProgram(program);
    // Provide the infolog in whatever manner you deem best.
    // Exit with failure.
    return false;
  }
  return true;
}
