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

#include <ftkInterface.h>

#include <cstdint>
#include <string>

#include "geometry_helper.hpp"

namespace atracsys {

class ISDKInterface {
 public:
  virtual ~ISDKInterface() = default;

  virtual ftkError setInt32(ftkLibrary lib, uint64_t sn, uint32_t optID, int32_t val) = 0;
  virtual ftkError setData(ftkLibrary lib, uint64_t sn, uint32_t optID, ftkBuffer* buffer) = 0;
  virtual ftkError getLastFrame(ftkLibrary lib, uint64_t sn, ftkFrameQuery* frame,
                                uint32_t timeout) = 0;
  virtual ftkFrameQuery* createFrame() = 0;
  virtual void destroyFrame(ftkFrameQuery* frame) = 0;
  virtual ftkError setFrameOptions(bool pixels, uint32_t eventsSize, uint32_t leftRawDataSize,
                                   uint32_t rightRawDataSize, uint32_t threeDFiducialsSize,
                                   uint32_t markersSize, ftkFrameQuery* frame) = 0;
  virtual ftkError setRigidBody(ftkLibrary lib, uint64_t sn, ftkRigidBody* rigidBody) = 0;
  virtual ftkError enumerateOptions(ftkLibrary lib, uint64_t sn, ftkOptionsEnumCallback cb,
                                    void* user) = 0;
  virtual ftkError initExt(const char* config, ftkBuffer* buffer) = 0;
  virtual void close(ftkLibrary* lib) = 0;
  virtual ftkError getData(ftkLibrary lib, uint64_t sn, uint32_t optID, ftkBuffer* buffer) = 0;
  virtual ftkError getInt32(ftkLibrary lib, uint64_t sn, uint32_t optID, int32_t* val) = 0;
  virtual ftkError getLastErrorString(ftkLibrary lib, uint32_t size, char* buffer) = 0;
  virtual ftkError loadBody(ftkLibrary lib, const std::string& fileName,
                            ftkRigidBody& geometry) = 0;
};

}  // namespace atracsys
