/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Wayland Technologies. All rights reserved.
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

#include <stdexcept>
#include <string>

#include "sdk_interface.hpp"

class RealSDKWrapper : public atracsys::ISDKInterface {
 public:
  ftkError setInt32(ftkLibrary lib, uint64_t sn, uint32_t optID, int32_t val) override {
    return ftkSetInt32(lib, sn, optID, val);
  }

  ftkError setData(ftkLibrary lib, uint64_t sn, uint32_t optID, ftkBuffer* buffer) override {
    return ftkSetData(lib, sn, optID, buffer);
  }

  ftkError getLastFrame(ftkLibrary lib, uint64_t sn, ftkFrameQuery* frame,
                        uint32_t timeout) override {
    return ftkGetLastFrame(lib, sn, frame, timeout);
  }

  ftkFrameQuery* createFrame() override { return ftkCreateFrame(); }

  void destroyFrame(ftkFrameQuery* frame) override { ftkDeleteFrame(frame); }

  ftkError setFrameOptions(bool pixels, uint32_t eventsSize, uint32_t leftRawDataSize,
                           uint32_t rightRawDataSize, uint32_t threeDFiducialsSize,
                           uint32_t markersSize, ftkFrameQuery* frame) override {
    return ftkSetFrameOptions(pixels, eventsSize, leftRawDataSize, rightRawDataSize,
                              threeDFiducialsSize, markersSize, frame);
  }

  ftkError setRigidBody(ftkLibrary lib, uint64_t sn, ftkRigidBody* rigidBody) override {
    return ftkSetRigidBody(lib, sn, rigidBody);
  }

  ftkError enumerateOptions(ftkLibrary lib, uint64_t sn, ftkOptionsEnumCallback cb,
                            void* user) override {
    return ftkEnumerateOptions(lib, sn, cb, user);
  }

  ftkError initExt(const char* config, ftkBuffer* buffer) override {
    ftkLibrary lib = ftkInitExt(config, buffer);
    if (!lib) {
      if (buffer && buffer->data && buffer->size > 0) {
        throw std::runtime_error(
            std::string("ftkInitExt failed: ") +
            std::string(reinterpret_cast<const char*>(buffer->data), buffer->size));
      }
      throw std::runtime_error("ftkInitExt failed (unknown error)");
    }
    return ftkError::FTK_OK;
  }

  void close(ftkLibrary* lib) override { ftkClose(lib); }

  ftkError getData(ftkLibrary lib, uint64_t sn, uint32_t optID, ftkBuffer* buffer) override {
    return ftkGetData(lib, sn, optID, buffer);
  }

  ftkError getInt32(ftkLibrary lib, uint64_t sn, uint32_t optID, int32_t* val) override {
    return ftkGetInt32(lib, sn, optID, val, ftkOptionGetter::FTK_VALUE);
  }

  ftkError getLastErrorString(ftkLibrary lib, uint32_t size, char* buffer) override {
    return ftkGetLastErrorString(lib, size, buffer);
  }

  int loadBody(ftkLibrary lib, const std::string& fileName, ftkRigidBody& geometry) override {
    return loadRigidBody(lib, fileName, geometry);
  }
};
