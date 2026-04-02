/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

struct stereo_matching_engine;
struct GpuFrame3D;
enum class ImageType3D : uint32_t;
struct StereoParameters;

class IS3DKInterface {
 public:
  virtual ~IS3DKInterface() = default;

  virtual StereoParameters* createStereoParameters() = 0;
  virtual bool initializeDeviceHelper(uint64_t* device_sn, ftkLibrary lib,
                                      ImageType3D* image_type) = 0;
  virtual stereo_matching_engine* createDefaultEngine() = 0;
  virtual GpuFrame3D* createGpu3DFrame(ImageType3D image_type) = 0;
  virtual bool computeDispMap(const ftkFrameQuery* frame, const stereo_matching_engine* engine,
                              GpuFrame3D* frame3d, StereoParameters* stereo) = 0;
};
