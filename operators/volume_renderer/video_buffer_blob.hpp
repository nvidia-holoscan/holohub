/* SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef VOLUME_RENDERER_VIDEO_BUFFER_BLOB
#define VOLUME_RENDERER_VIDEO_BUFFER_BLOB

#include <claraviz/util/Blob.h>

#include "gxf/multimedia/video.hpp"

/**
 * Blob holding a nvidia::gxf::VideoBuffer
 */
class VideoBufferBlob : public clara::viz::Blob {
 public:
  /**
   * Construct
   *
   * @param video_buffer [in] video buffer
   */
  explicit VideoBufferBlob(const nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>& video_buffer)
      : video_buffer_(video_buffer) {}
  VideoBufferBlob() = delete;

  std::unique_ptr<IBlob::AccessGuard> Access() override {
    std::unique_ptr<Blob::AccessGuard> guard(new AccessGuard(this));
    Blob::SyncAccess(guard.get());
    return guard;
  }

  std::unique_ptr<IBlob::AccessGuard> Access(CUstream stream) override {
    std::unique_ptr<Blob::AccessGuard> guard(new AccessGuard(this));
    Blob::SyncAccess(guard.get(), stream);
    return guard;
  }

  std::unique_ptr<IBlob::AccessGuardConst> AccessConst() override {
    std::unique_ptr<Blob::AccessGuardConst> guard(new AccessGuardConst(this));
    Blob::SyncAccessConst(guard.get());
    return guard;
  }

  std::unique_ptr<IBlob::AccessGuardConst> AccessConst(CUstream stream) override {
    std::unique_ptr<Blob::AccessGuardConst> guard(new AccessGuardConst(this));
    Blob::SyncAccessConst(guard.get(), stream);
    return guard;
  }

  size_t GetSize() const override { return video_buffer_->size(); }

 private:
  class AccessGuard : public Blob::AccessGuard {
   public:
    explicit AccessGuard(Blob* blob) : Blob::AccessGuard(blob) {}

    void* GetData() override {
      return reinterpret_cast<void*>(
          static_cast<VideoBufferBlob*>(blob_)->video_buffer_->pointer());
    }
  };

  class AccessGuardConst : public Blob::AccessGuardConst {
   public:
    explicit AccessGuardConst(Blob* blob) : Blob::AccessGuardConst(blob) {}

    const void* GetData() override {
      return reinterpret_cast<const void*>(
          static_cast<VideoBufferBlob*>(blob_)->video_buffer_->pointer());
    }
  };

  const nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> video_buffer_;
};

#endif /* VOLUME_RENDERER_VIDEO_BUFFER_BLOB */
