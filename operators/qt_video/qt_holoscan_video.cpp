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

#include "qt_holoscan_video.hpp"

#include "opengl_renderer.hpp"

#include <QtQuick/qquickwindow.h>
#include <QtCore/QRunnable>

QtHoloscanVideo::QtHoloscanVideo() : renderer_(nullptr) {
  connect(this, &QQuickItem::windowChanged, this, &QtHoloscanVideo::handleWindowChanged);

  shared_data_ = std::make_unique<QtHoloscanSharedData>();

  QObject::connect(this, &QtHoloscanVideo::bufferChanged, this, &QtHoloscanVideo::forceRedraw);
}

void QtHoloscanVideo::processBuffer(void* pointer,
                                    const nvidia::gxf::VideoBufferInfo& video_buffer_info,
                                    cudaEvent_t cuda_event) {
  // signal that there is a new video buffer
  {
    std::lock_guard lock(shared_data_->mutex_);

    // set the implicit size of the item so it automatically resize in the UI is the user did
    // not set an explicit size
    if (shared_data_->video_buffer_info_.width != video_buffer_info.width) {
      setImplicitWidth(video_buffer_info.width);
    }
    if (shared_data_->video_buffer_info_.width != video_buffer_info.height) {
      setImplicitHeight(video_buffer_info.height);
    }

    // update the shared data and mark it ready for the renderer to use
    shared_data_->pointer_ = pointer;
    shared_data_->video_buffer_info_ = video_buffer_info;
    shared_data_->cuda_event_ = cuda_event;
    shared_data_->state_ = QtHoloscanSharedData::State::Ready;
  }
  shared_data_->condition_variable_.notify_one();

  // force redraw
  emit bufferChanged();

  // wait for the renderer
  {
    std::unique_lock lock(shared_data_->mutex_);
    shared_data_->condition_variable_.wait_until(
        lock, std::chrono::steady_clock::now() + std::chrono::seconds(5),
        [this] { return shared_data_->state_ == QtHoloscanSharedData::State::Processed; });
  }
}

void QtHoloscanVideo::forceRedraw() {
  // force redraw of window
  if (window()) window()->update();
}

void QtHoloscanVideo::handleWindowChanged(QQuickWindow* win) {
  if (win) {
    connect(win,
            &QQuickWindow::beforeSynchronizing,
            this,
            &QtHoloscanVideo::sync,
            Qt::DirectConnection);
    connect(win,
            &QQuickWindow::sceneGraphInvalidated,
            this,
            &QtHoloscanVideo::cleanup,
            Qt::DirectConnection);
  }
}

void QtHoloscanVideo::cleanup() {
  delete renderer_;
  renderer_ = nullptr;
}

class CleanupJob : public QRunnable {
 public:
  explicit CleanupJob(OpenGLRenderer* renderer) : renderer_(renderer) {}
  CleanupJob() = delete;
  void run() override { delete renderer_; }

 private:
  OpenGLRenderer* renderer_;
};

void QtHoloscanVideo::releaseResources() {
  window()->scheduleRenderJob(new CleanupJob(renderer_), QQuickWindow::BeforeSynchronizingStage);
  renderer_ = nullptr;
}

void QtHoloscanVideo::geometryChange(const QRectF& newGeometry, const QRectF& oldGeometry) {
  // record the geometry of the item for the renderer to use
  geometry_ = newGeometry;
  QQuickItem::geometryChange(newGeometry, oldGeometry);
}

void QtHoloscanVideo::sync() {
  if (!renderer_) {
    renderer_ = new OpenGLRenderer(shared_data_.get());
    connect(window(),
            &QQuickWindow::beforeRendering,
            renderer_,
            &OpenGLRenderer::init,
            Qt::DirectConnection);
    connect(window(),
            &QQuickWindow::beforeRenderPassRecording,
            renderer_,
            &OpenGLRenderer::paint,
            Qt::DirectConnection);
  }
  renderer_->setGeometry(geometry_);
  renderer_->setWindow(window());
}
