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

#include <map>
#include <stdexcept>
#include <string>

#include <s3dk_gpu.hpp>

#include "helpers.hpp"
#include "s3dk_interface.hpp"

namespace atracsys::sdk::detail {

inline std::map<std::string, uint32_t> enumerate_sdk_options(ftkLibrary lib, uint64_t sn) {
  std::map<std::string, uint32_t> options;
  if (ftkEnumerateOptions(lib, sn, atracsys::sdk::optionEnumerator, &options) != ftkError::FTK_OK ||
      options.empty()) {
    throw std::runtime_error("Cannot retrieve Atracsys SDK options.");
  }
  return options;
}

inline std::string device_type_string(ftkLibrary lib, uint64_t sn,
                                      const std::map<std::string, uint32_t>& options) {
  auto it = options.find("sTk device type");
  if (it == options.end()) {
    throw std::runtime_error("Missing sTk device type option.");
  }

  ftkBuffer buffer{};
  if (ftkGetData(lib, sn, it->second, &buffer) != ftkError::FTK_OK) {
    throw std::runtime_error("Failed to fetch Atracsys device type.");
  }
  std::string result(reinterpret_cast<const char*>(buffer.data), buffer.size);
  if (!result.empty() && result.back() == '\0') {
    result.pop_back();
  }
  return result;
}

inline void set_device_option(ftkLibrary lib, uint64_t sn,
                              const std::map<std::string, uint32_t>& options, const char* name,
                              int32_t value) {
  auto it = options.find(name);
  if (it == options.end()) {
    return;
  }

  const ftkError status = ftkSetInt32(lib, sn, it->second, value);
  if (status != ftkError::FTK_OK) {
    throw std::runtime_error(
        std::string("Failed to set Atracsys option: ") + name +
        ", status: " + std::to_string(static_cast<int>(status)));
  }
}

inline void configure_device_defaults(ftkLibrary lib, uint64_t sn,
                                      const std::map<std::string, uint32_t>& options) {
  set_device_option(lib, sn, options, "Enable embedded processing", 1);
  set_device_option(lib, sn, options, "Enable images sending", 1);
  set_device_option(lib, sn, options, "Calibration export", 1);
  set_device_option(lib, sn, options, "Embedded Symmetrise coordinates", 0);
  set_device_option(lib, sn, options, "Enable 16 bits pictures", 0);
}

}  // namespace atracsys::sdk::detail

class RealS3DKWrapper : public atracsys::IS3DKInterface {
 public:
  StereoParameters* createStereoParameters() override { return create_stereo_parameters(); }
  void destroyStereoParameters(StereoParameters* ptr) override { delete ptr; }

  bool initializeDeviceHelper(uint64_t* device_sn, ftkLibrary lib,
                              ImageType3D* image_type) override {
    if (!device_sn || !lib || !image_type) {
      throw std::runtime_error("initializeDeviceHelper: invalid arguments");
    }
    auto options =
        atracsys::sdk::detail::enumerate_sdk_options(lib, *device_sn);
    const std::string device_type =
        atracsys::sdk::detail::device_type_string(lib, *device_sn, options);
    bool valid = false;
    *image_type = convert_string_image_type(device_type, &valid);
    if (!valid) {
      throw std::runtime_error("Unsupported Atracsys device type for S3DK: " + device_type);
    }
    atracsys::sdk::detail::configure_device_defaults(lib, *device_sn, options);
    return true;
  }

  stereo_matching_engine* createDefaultEngine() override { return create_default_engine(); }
  void destroyDefaultEngine(stereo_matching_engine* ptr) override { delete ptr; }

  GpuFrame3D* createGpu3DFrame(ImageType3D image_type) override {
    return create_gpu3DFrame(image_type);
  }
  void destroyGpu3DFrame(GpuFrame3D* ptr) override { delete ptr; }

  bool computeDispMap(const ftkFrameQuery* frame, const stereo_matching_engine* engine,
                      GpuFrame3D* frame3d, StereoParameters* stereo) override {
    const bool result = compute_disp_map(frame, engine, frame3d, stereo);
    if (result) {
      remove_invalid_disparity(frame3d, engine);
    }
    return result;
  }
};
