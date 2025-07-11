/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "opengl_renderer.hpp"

#include <QQuickOpenGLUtils>
#include <QtQuick/QQuickWindow>

#include <cuda_gl_interop.h>

#define CUDA_TRY(stmt)                                                                        \
  ({                                                                                          \
    cudaError_t _holoscan_cuda_err = stmt;                                                    \
    if (cudaSuccess != _holoscan_cuda_err) {                                                  \
      HOLOSCAN_LOG_ERROR("CUDA Runtime call {} in line {} of file {} failed with '{}' ({}).", \
                         #stmt,                                                               \
                         __LINE__,                                                            \
                         __FILE__,                                                            \
                         cudaGetErrorString(_holoscan_cuda_err),                              \
                         static_cast<int>(_holoscan_cuda_err));                               \
    }                                                                                         \
    _holoscan_cuda_err;                                                                       \
  })

OpenGLRenderer::OpenGLRenderer(QtHoloscanSharedData* shared_data) : shared_data_(shared_data) {}

OpenGLRenderer::~OpenGLRenderer() {
  delete program_;
}

void OpenGLRenderer::init() {
  if (!program_) {
    QSGRendererInterface* rif = window_->rendererInterface();
    Q_ASSERT(rif->graphicsApi() == QSGRendererInterface::OpenGL);

    initializeOpenGLFunctions();

    vao_ = std::make_unique<QOpenGLVertexArrayObject>();
    if (!vao_->create()) { qFatal("Failed to create QOpenGLVertexArrayObject"); }
    vao_->bind();

    vertex_buffer_ = std::make_unique<QOpenGLBuffer>(QOpenGLBuffer::VertexBuffer);
    if (!vertex_buffer_->create()) { qFatal("Failed to create QOpenGLBuffer"); }

    const float vertices[] = {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f};
    vertex_buffer_->bind();
    vertex_buffer_->setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertex_buffer_->allocate(vertices, sizeof(vertices));

    program_ = new QOpenGLShaderProgram();
    if (!program_->create()) { qFatal("Failed to create QOpenGLShaderProgram"); }

    program_->addCacheableShaderFromSourceCode(
        QOpenGLShader::Vertex,
        "#version 330 core\n"
        "layout (location = 0) in vec2 vertices;\n"
        "out vec2 texCoord;\n"
        "void main() {\n"
        "  gl_Position = vec4((vertices * 2.0) - 1.0, 0.0, 1.0);\n"
        "  texCoord = vec2(vertices.x, 1.0 - vertices.y);\n"
        "}\n");

    program_->addCacheableShaderFromSourceCode(QOpenGLShader::Fragment,
                                               "#version 330 core\n"
                                               "out vec4 fragColor;\n"
                                               "in vec2 texCoord;\n"
                                               "uniform sampler2D inTexture;\n"
                                               "void main() {\n"
                                               "  fragColor = texture2D(inTexture, texCoord);\n"
                                               "}\n");
    program_->bindAttributeLocation("vertices", 0);
    program_->link();

    program_->enableAttributeArray("vertices");
    program_->setAttributeBuffer("vertices", GL_FLOAT, 0, 2, 2 * sizeof(float));

    program_->release();
    vao_->release();
    vertex_buffer_->release();

    CUDA_TRY(cudaStreamCreate(&cuda_stream_));
  }
}

/**
 * @brief Get the OpenGL texture format for a given GXF video format
 */
static GLenum getTextureFormat(nvidia::gxf::VideoFormat video_format) {
  switch (video_format) {
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA:
      return GL_RGBA8;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRA:
      return GL_BGRA;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_ABGR:
      return GL_ABGR_EXT;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY:
      return GL_LUMINANCE;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY16:
      return GL_LUMINANCE16;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32:
      return GL_LUMINANCE32UI_EXT;
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32F:
      return GL_LUMINANCE32F_ARB;
    default:
      throw std::runtime_error(fmt::format("Unsupported color format {}", int64_t(video_format)));
  }
}

/**
 * @brief Get the bytes per texel for a given texture format
 */
static uint32_t getBytesPerTexel(GLenum texture_format) {
  switch (texture_format) {
    case GL_RGBA8:
    case GL_BGRA:
    case GL_ABGR_EXT:
      return 4 * sizeof(uint8_t);
    case GL_LUMINANCE:
      return 1 * sizeof(uint8_t);
    case GL_LUMINANCE16:
      return 1 * sizeof(uint16_t);
    case GL_LUMINANCE32UI_EXT:
    case GL_LUMINANCE32F_ARB:
      return 1 * sizeof(uint32_t);
    default:
      throw std::runtime_error(fmt::format("Unhandled texture format {}", int64_t(texture_format)));
  }
}

void OpenGLRenderer::paint() {
  // Play nice with the RHI. Not strictly needed when the scenegraph uses
  // OpenGL directly.
  window_->beginExternalCommands();
  QQuickOpenGLUtils::resetOpenGLState();

  {
    std::lock_guard lock(shared_data_->mutex_);

    if (shared_data_->state_ == QtHoloscanSharedData::State::Ready) {
      // Update the texture using the given buffer.
      const uint32_t new_width = shared_data_->video_buffer_info_.width;
      const uint32_t new_height = shared_data_->video_buffer_info_.height;
      const GLenum new_texture_format =
          getTextureFormat(shared_data_->video_buffer_info_.color_format);
      if ((texture_format_ != new_texture_format) || (texture_width_ != new_width) ||
          (texture_height_ != new_height)) {
        texture_format_ = new_texture_format;
        texture_width_ = new_width;
        texture_height_ = new_height;
        // Create the texture.
        if (gl_texture_) {
          glDeleteTextures(1, &gl_texture_);
          gl_texture_ = 0;
        }
        glGenTextures(1, &gl_texture_);
        glBindTexture(GL_TEXTURE_2D, gl_texture_);
        glTexStorage2D(GL_TEXTURE_2D, 1, texture_format_, texture_width_, texture_height_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);

        if (cuda_resource_) {
          CUDA_TRY(cudaGraphicsUnregisterResource(cuda_resource_));
          cuda_resource_ = nullptr;
        }
        CUDA_TRY(cudaGraphicsGLRegisterImage(
            &cuda_resource_, gl_texture_, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
      }

      // Map the OpenGL texture to be used by CUDA, this automatically synchronizes with OpenGL
      // rendering
      CUDA_TRY(cudaGraphicsMapResources(1, &cuda_resource_, cuda_stream_));
      cudaArray* cuda_ptr = nullptr;
      CUDA_TRY(cudaGraphicsSubResourceGetMappedArray(&cuda_ptr, cuda_resource_, 0, 0));

      // Synchronize with the event provided by the operator
      CUDA_TRY(cudaStreamWaitEvent(cuda_stream_, shared_data_->cuda_event_));

      // Copy the new data to the mapped texture
      CUDA_TRY(cudaMemcpy2DToArrayAsync(cuda_ptr,
                                        0,
                                        0,
                                        shared_data_->pointer_,
                                        shared_data_->video_buffer_info_.color_planes[0].stride,
                                        texture_width_ * getBytesPerTexel(texture_format_),
                                        texture_height_,
                                        cudaMemcpyDeviceToDevice,
                                        cuda_stream_));

      // Record the event so that the operator can synchronize with the CUDA mem copy
      CUDA_TRY(cudaEventRecord(shared_data_->cuda_event_, cuda_stream_));

      // Unmap the OpenGL texture, this automatically synchronizes with OpenGL rendering
      CUDA_TRY(cudaGraphicsUnmapResources(1, &cuda_resource_, cuda_stream_));

      shared_data_->state_ = QtHoloscanSharedData::State::Processed;
      shared_data_->condition_variable_.notify_one();
    }
  }

  if (gl_texture_) {
    program_->bind();
    vao_->bind();

    // OpenGL is bottom-up, Qt top-down, have to calculate the correct y position
    const qreal ratio = window_->effectiveDevicePixelRatio();
    glViewport(geometry_.left() * ratio,
               (window_->size().height() - geometry_.top() - geometry_.height()) * ratio,
               geometry_.width() * ratio,
               geometry_.height() * ratio);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gl_texture_);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    vao_->release();
    program_->release();
  }

  window_->endExternalCommands();
}
