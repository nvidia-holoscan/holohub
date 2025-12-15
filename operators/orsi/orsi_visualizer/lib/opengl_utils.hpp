/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

void GLAPIENTRY OpenGLDebugMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
                                           GLsizei length, const GLchar* message,
                                           const void* userParam);

// FBO

struct Tex2DFBO {
    GLuint  fbo_ = 0;
    GLuint  tex_ = 0;
    int32_t width_, height_;

    void resize(const int32_t width, const int32_t height)  {
        if (fbo_ != 0 && width_ == width && height_ == height) {
          return;
        }

        width_ = width;
        height_ = height;

        if (!fbo_) {
          glGenFramebuffers(1, &fbo_);
        }

        GLuint tex2delete(tex_);
        glCreateTextures(GL_TEXTURE_2D, 1, &tex_);
        if (tex2delete) {
            glDeleteTextures(1, &tex2delete);
        }
        // allocate 2D texture storage
        glTextureStorage2D(tex_, 1, GL_RGBA8, width_, height_);
        // update fbo color attachment
        glNamedFramebufferTexture(fbo_,  GL_COLOR_ATTACHMENT0, tex_, 0);
    }

    void bind() {
      glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    }

    void destroy() {
      glDeleteTextures(1, &tex_);
      glDeleteFramebuffers(1, &fbo_);
    }
};

struct RenderBufferFBO {
    GLuint  fbo_ = 0;
    GLuint  rb_ = 0;
    int32_t width_, height_;

    void resize(const int32_t width, const int32_t height) {
        if (fbo_ != 0 && width_ == width && height_ == height) {
          return;
        }

        width_ = width;
        height_ = height;

        if (!fbo_) {
          glGenFramebuffers(1, &fbo_);
        }

        GLuint rb2delete(rb_);
        glCreateRenderbuffers(1, &rb_);
        if (rb2delete) {
            glDeleteRenderbuffers(1, &rb2delete);
        }
        // allocate render buffer storage
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width_, height_);
        // update fbo color attachment
        glNamedFramebufferRenderbuffer(fbo_,  GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb_);
    }

    void bind() {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    }

    void destroy() {
        glDeleteRenderbuffers(1, &rb_);
        glDeleteFramebuffers(1, &fbo_);
    }
};

// GLSL

bool createGLSLShader(GLenum shader_type, GLuint& shader, const char* shader_src);

bool linkGLSLProgram(const GLuint vertex_shader, const GLuint fragment_shader, GLuint& program);
