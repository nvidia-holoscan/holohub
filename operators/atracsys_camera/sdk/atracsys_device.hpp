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
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

#include "helpers.hpp"

class AtracsysDevice {
 public:
  static AtracsysDevice& instance() {
    static AtracsysDevice dev;
    return dev;
  }

  void init() {
    std::lock_guard<std::mutex> lk(m_);
    if (lib_) return;

    ftkLibrary new_lib = ftkInitExt(nullptr, &buffer_);
    if (!new_lib) {
      if (buffer_.data && buffer_.size > 0) {
        throw std::runtime_error(
            std::string(reinterpret_cast<const char*>(buffer_.data), buffer_.size));
      }
      throw std::runtime_error("ftkInitExt failed");
    }

    try {
      atracsys::sdk::DeviceData devInfo =
          atracsys::sdk::retrieveLastDevice(new_lib, false, false, true);
      const uint64_t serial = devInfo.SerialNumber;
      if (!serial) throw std::runtime_error("No Atracsys devices detected");

      std::map<std::string, uint32_t> options;
      if (ftkEnumerateOptions(
              new_lib, serial, atracsys::sdk::optionEnumerator, &options) !=
              ftkError::FTK_OK ||
          options.empty()) {
        throw std::runtime_error("ftkEnumerateOptions failed");
      }

      lib_ = new_lib;
      sn_ = serial;
      options_ = std::move(options);
    } catch (...) {
      ftkClose(&new_lib);
      throw;
    }
  }

  void shutdown() {
    std::lock_guard<std::mutex> lk(m_);
    if (lib_) ftkClose(&lib_);
    lib_ = nullptr;
    sn_ = 0;
    options_.clear();
  }

  ftkLibrary lib() const { return lib_; }
  uint64_t serial() const { return sn_; }
  ftkBuffer& buffer() { return buffer_; }
  const std::map<std::string, uint32_t>& options() const { return options_; }

 private:
  AtracsysDevice() = default;
  ~AtracsysDevice() = default;
  AtracsysDevice(const AtracsysDevice&) = delete;
  AtracsysDevice& operator=(const AtracsysDevice&) = delete;

  mutable std::mutex m_;
  ftkLibrary lib_{nullptr};
  uint64_t sn_{0};
  ftkBuffer buffer_{};
  std::map<std::string, uint32_t> options_;
};
