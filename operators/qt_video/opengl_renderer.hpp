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

#ifndef OPERATORS_QT_VIDEO_OPENGL_RENDERER
#define OPERATORS_QT_VIDEO_OPENGL_RENDERER

#include "shared_data.hpp"

#include <QOpenGLBuffer>
#include <QOpenGLContext>
#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>

class QQuickWindow;

typedef struct cudaGraphicsResource* cudaGraphicsResource_t;
typedef struct CUstream_st* cudaStream_t;

/**
 * @brief This class is responsible for rendering the image to the QtHoloscanVideo item
 *
 * It communicates through shared data with the QtHoloscanVideo item.
 */
class OpenGLRenderer : public QObject, protected QOpenGLExtraFunctions {
  Q_OBJECT

 public:
  /**
   * @brief Construct a new OpenGLRenderer object
   *
   * @param shared_data shared data
   */
  explicit OpenGLRenderer(QtHoloscanSharedData* shared_data);
  OpenGLRenderer() = delete;
  ~OpenGLRenderer();

  /**
   * @brief Set the geometry of the render area
   *
   * @param geometry render area
   */
  void setGeometry(const QRectF& geometry) { geometry_ = geometry; }

  /**
   * @brief Set the window object the renderer should use
   *
   * @param window
   */
  void setWindow(QQuickWindow* window) { window_ = window; }

 public slots:
  void init();
  void paint();

 private:
  QRectF geometry_;

  // OpenGL state
  QOpenGLShaderProgram* program_ = nullptr;
  QQuickWindow* window_ = nullptr;
  std::unique_ptr<QOpenGLVertexArrayObject> vao_;
  std::unique_ptr<QOpenGLBuffer> vertex_buffer_;
  GLuint sampler_ = 0;
  GLuint gl_texture_ = 0;
  uint32_t texture_width_ = 0;
  uint32_t texture_height_ = 0;
  GLenum texture_format_ = GL_NONE;

  QtHoloscanSharedData* const shared_data_;

  // CUDA state
  cudaStream_t cuda_stream_ = nullptr;
  cudaGraphicsResource_t cuda_resource_ = nullptr;
};

#endif /* OPERATORS_QT_VIDEO_OPENGL_RENDERER */
