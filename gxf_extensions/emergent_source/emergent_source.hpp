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
#ifndef NVIDIA_HOLOSCAN_GXF_EXTENSIONS_EMERGENT_SOURCE_HPP_
#define NVIDIA_HOLOSCAN_GXF_EXTENSIONS_EMERGENT_SOURCE_HPP_

#include <emergentframe.h>
#include <emergenterrors.h>
#include <gigevisiondeviceinfo.h>
#include <EmergentCamera.h>
#include <EmergentCameraAPIs.h>

#include <string>
#include <vector>

#include "gxf/std/codelet.hpp"
#include "gxf/std/transmitter.hpp"

#define FRAMES_BUFFERS 256

using namespace Emergent;

namespace nvidia {
namespace holoscan {

constexpr uint32_t kMaxCameras = 10;
constexpr uint32_t kNumBuffers = 10;

constexpr uint32_t kDefaultWidth = 4200;
constexpr uint32_t kDefaultHeight = 2160;
constexpr uint32_t kDefaultFramerate = 240;
constexpr bool kDefaultRDMA = false;
constexpr PIXEL_FORMAT kDefaultPixelFormat = GVSP_PIX_BAYGB8;
constexpr uint32_t kDefaultExposure = 3072;
constexpr uint32_t kDefaultGain = 4095;


/// @brief Video input codelet for use with Emergent cameras using ConnectX-6
///
/// Provides a codelet for supporting Emergent camera as a source.
/// It offers support for GPUDirect-RDMA on Quadro GPUs.
/// The output is a VideoBuffer object.

class EmergentSource : public gxf::Codelet {
 public:
  EmergentSource();

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

 private:
  EVT_ERROR CheckCameraCapabilities();
  EVT_ERROR OpenEVTCamera();
  void SetDefaultConfiguration();

  gxf::Parameter<gxf::Handle<gxf::Transmitter>> signal_;
  gxf::Parameter<uint32_t> width_;
  gxf::Parameter<uint32_t> height_;
  gxf::Parameter<uint32_t> framerate_;
  gxf::Parameter<bool> use_rdma_;
  gxf::Parameter<uint32_t> exposure_;
  gxf::Parameter<uint32_t> gain_;

  CEmergentCamera camera_;
  CEmergentFrame evt_frame_[FRAMES_BUFFERS];
  CEmergentFrame evt_frame_recv_;
};

}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_HOLOSCAN_GXF_EXTENSIONS_EMERGENT_SOURCE_HPP_
