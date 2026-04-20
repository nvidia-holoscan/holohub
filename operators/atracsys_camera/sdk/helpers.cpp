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

#include "sdk/helpers.hpp"

#include <ftkPlatform.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

using namespace std;

void deviceEnumerator(uint64_t sn, void* user, ftkDeviceType type) {
  if (user != nullptr) {
    DeviceData* ptr = reinterpret_cast<DeviceData*>(user);
    ptr->SerialNumber = sn;
    ptr->Type = type;
  }
}

void fusionTrackEnumerator(uint64_t sn, void* user, ftkDeviceType devType) {
  if (user != nullptr && devType != ftkDeviceType::DEV_SIMULATOR) {
    DeviceData* ptr = reinterpret_cast<DeviceData*>(user);
    ptr->SerialNumber = sn;
    ptr->Type = devType;
  }
}

DeviceData retrieveLastDevice(ftkLibrary lib, bool allowSimulator, bool quiet,
                              bool /*dontWaitForKeyboard*/) {
  DeviceData device;
  device.SerialNumber = 0uLL;

  ftkError err;
  if (allowSimulator) {
    err = ftkEnumerateDevices(lib, deviceEnumerator, &device);
  } else {
    err = ftkEnumerateDevices(lib, fusionTrackEnumerator, &device);
  }

  if (err > ftkError::FTK_OK) {
    throw std::runtime_error("ftkEnumerateDevices returned an error");
  }

  if (device.SerialNumber == 0uLL) {
    throw std::runtime_error("No Atracsys device connected (or only simulator detected)");
  }

  if (!quiet) {
    string text;
    switch (device.Type) {
      case ftkDeviceType::DEV_SPRYTRACK_180:
        text = "sTk 180";
        break;
      case ftkDeviceType::DEV_SPRYTRACK_300:
        text = "sTk 300";
        break;
      case ftkDeviceType::DEV_FUSIONTRACK_500:
        text = "fTk 500";
        break;
      case ftkDeviceType::DEV_FUSIONTRACK_250:
        text = "fTk 250";
        break;
      case ftkDeviceType::DEV_SIMULATOR:
        text = "fTk simulator";
        break;
      default:
        text = "UNKNOWN";
        break;
    }
    cout << "Detected Atracsys device: " << text << " SN=0x" << setw(16)
         << setfill('0') << hex << device.SerialNumber << dec << '\n';
  }

  return device;
}

void optionEnumerator(uint64_t /*sn*/, void* user, ftkOptionsInfo* oi) {
  if (user == nullptr || oi == nullptr) return;
  auto* mapping = static_cast<map<string, uint32_t>*>(user);
  mapping->emplace(oi->name, oi->id);
}

namespace atracsys::sdk {

void throwSdkError(const char* message, bool /*dontWaitForKeyboard*/) {
  throw std::runtime_error(message);
}

void checkError(ftkLibrary lib, bool dontWaitForKeyboard, bool /*quit*/) {
  char buf[1024];
  const auto status = ftkGetLastErrorString(lib, sizeof(buf), buf);
  if (status != ftkError::FTK_OK) {
    throw std::runtime_error("ftkGetLastErrorString failed with error code: " + std::to_string(status));
  }
  if (std::strlen(buf) > 0) {
    throwSdkError(buf, dontWaitForKeyboard);
  }
}

} // namespace atracsys::sdk
