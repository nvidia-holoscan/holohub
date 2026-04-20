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

#include <string>

namespace atracsys::sdk {

struct DeviceData {
  uint64_t SerialNumber{0};
  ftkDeviceType Type{ftkDeviceType::DEV_UNKNOWN_DEVICE};
};

void deviceEnumerator(uint64_t sn, void* user, ftkDeviceType type);
void fusionTrackEnumerator(uint64_t sn, void* user, ftkDeviceType devType);
DeviceData retrieveLastDevice(ftkLibrary lib, bool allowSimulator = false, bool quiet = false,
                              bool dontWaitForKeyboard = true);
void optionEnumerator(uint64_t sn, void* user, ftkOptionsInfo* oi);

void throwSdkError(const char* message, bool dontWaitForKeyboard = false);
void checkError(ftkLibrary lib, bool dontWaitForKeyboard = false, bool quit = true);
}  // namespace atracsys::sdk
