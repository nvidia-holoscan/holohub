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

#ifndef OPERATORS_QT_VIDEO_QT_HOLOSCAN_VIDEO
#define OPERATORS_QT_VIDEO_QT_HOLOSCAN_VIDEO

#include "qt_video_op.hpp"
#include "shared_data.hpp"

#include <memory>

#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>

// forward declarations
class OpenGLRenderer;

/**
 * @brief This class is a QQuickItem which displays the video frames received by QtVideoOp.
 *
 */
class QtHoloscanVideo : public QQuickItem {
  Q_OBJECT
  QML_ELEMENT

 public:
  QtHoloscanVideo();

  /**
   * @brief Process a video buffer
   *
   * @param pointer pointer to CUDA memory
   * @param video_buffer_info video buffer information
   * @param cuda_event CUDA event to use for synchronization (wait before using memory, record after
   * using memory)
   */
  void processBuffer(void* pointer, const nvidia::gxf::VideoBufferInfo& video_buffer_info,
                     cudaEvent_t cuda_event);

 public slots:
  void sync();
  void cleanup();
  void forceRedraw();

 private slots:
  void handleWindowChanged(QQuickWindow* win);

 signals:
  void bufferChanged();

 private:
  void releaseResources() override;
  void geometryChange(const QRectF& newGeometry, const QRectF& oldGeometry) override;

  OpenGLRenderer* renderer_ = nullptr;
  QRectF geometry_;

  std::unique_ptr<QtHoloscanSharedData> shared_data_;
};

#endif /* OPERATORS_QT_VIDEO_QT_HOLOSCAN_VIDEO */
