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

#include "sdk/geometry_helper.hpp"

#include <ftkInterface.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>

static void getDataDirOptionId(uint64_t sn, void* user, ftkOptionsInfo* oi);

ftkError loadRigidBody(ftkLibrary lib, const std::string& fileName, ftkRigidBody& geometry) {
  std::string fullFileName;
  bool fromDataDir = false;
  if (!getFullFilePath(lib, fileName, fullFileName, &fromDataDir)) {
    return ftkError::FTK_WAR_FILE_NOT_FOUND;
  }
  ftkBuffer buffer{};
  if (!loadFileInBuffer(fullFileName, buffer)) {
    return ftkError::FTK_ERR_READ;
  }

  return ftkLoadRigidBodyFromFile(lib, &buffer, &geometry);
}

bool getFullFilePath(ftkLibrary lib, const std::string& fileName, std::string& fullFilePath,
                     bool* fromSystem) {
  std::ifstream local_input{fileName};
  if (local_input.is_open()) {
    fullFilePath = fileName;
    if (fromSystem != nullptr) *fromSystem = false;
    return true;
  }

  static uint32_t FTK_OPT_DATA_DIR = 0u;
  static std::string OPT_DIR;
  static std::once_flag flag;

  std::call_once(flag, [&]() {
    // When called with serial 0uLL (global scope), ftkEnumerateOptions returns
    // FTK_WAR_OPT_GLOBAL_ONLY (not FTK_OK) indicating options are global-only.
    // We intentionally check for that specific warning when relying on
    // getDataDirOptionId to populate FTK_OPT_DATA_DIR, so OPT_DIR resolves correctly.
    if (ftkEnumerateOptions(lib, 0uLL, getDataDirOptionId, &FTK_OPT_DATA_DIR) !=
        ftkError::FTK_WAR_OPT_GLOBAL_ONLY) {
      std::cerr << "Could not get the data directory option ID\n";
      FTK_OPT_DATA_DIR = 0u;
      return;
    }
    if (FTK_OPT_DATA_DIR != 0u) {
      ftkBuffer buffer{};
      if (ftkGetData(lib, 0uLL, FTK_OPT_DATA_DIR, &buffer) == ftkError::FTK_OK &&
          buffer.size >= 1u) {
        OPT_DIR.assign(reinterpret_cast<const char*>(buffer.data), buffer.size);
        if (!OPT_DIR.empty() && OPT_DIR.back() == '\0') {
          OPT_DIR.pop_back();
        }
      }
    }
  });

  if (FTK_OPT_DATA_DIR == 0u) {
    std::cerr << "Could not get the data directory option ID\n";
    return false;
  }
  if (OPT_DIR.empty()) {
    return false;
  }

  const std::string candidate = OPT_DIR + "/" + fileName;
  std::ifstream system_input{candidate};
  if (system_input.is_open()) {
    fullFilePath = candidate;
    if (fromSystem != nullptr) *fromSystem = true;
    return true;
  }

  return false;
}

bool loadFileInBuffer(const std::string& fullFilePath, ftkBuffer& buffer) {
  std::ifstream input(fullFilePath, std::ios::binary | std::ios::ate);
  if (!input.is_open()) {
    std::cerr << "Could not open file '" << fullFilePath << "'\n";
    return false;
  }

  buffer.reset();
  std::streampos pos = input.tellg();
  if (pos < 0 || pos > std::numeric_limits<uint32_t>::max()) {
    std::cerr << "File too large for buffer\n";
    return false;
  }

  const auto file_size = static_cast<size_t>(pos);
  if (file_size > sizeof(buffer.data)) {
    std::cerr << "Failed: file size exceeds ftkBuffer array capacity\n";
    return false;
  }
  buffer.size = 0u;

  input.seekg(0u, std::ios::beg);
  input.read(buffer.data, static_cast<std::streamsize>(pos));
  if (input.fail()) {
    std::cerr << "Could not read file '" << fullFilePath << "'\n";
    return false;
  }
  buffer.size = static_cast<uint32_t>(pos);
  return true;
}

static void getDataDirOptionId(uint64_t /*sn*/, void* user, ftkOptionsInfo* oi) {
  auto* ptr = reinterpret_cast<uint32_t*>(user);
  if (ptr != nullptr && strcmp(oi->name, "Data Directory") == 0) {
    *ptr = oi->id;
  }
}
