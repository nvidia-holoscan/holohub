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

#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

using namespace std;

static void getDataDirOptionId(uint64_t sn, void* user, ftkOptionsInfo* oi);

int loadRigidBody(ftkLibrary lib, const string& fileName, ftkRigidBody& geometry) {
  string fullFileName;
  bool fromDataDir = false;
  if (!getFullFilePath(lib, fileName, fullFileName, &fromDataDir)) {
    return 2;
  }
  ftkBuffer buffer{};
  if (!loadFileInBuffer(fullFileName, buffer)) {
    return 2;
  }

  ftkError err = ftkLoadRigidBodyFromFile(lib, &buffer, &geometry);
  if (err != ftkError::FTK_OK) {
    return 2;
  }

  return fromDataDir ? 1 : 0;
}

bool getFullFilePath(ftkLibrary lib, const string& fileName, string& fullFilePath,
                     bool* fromSystem) {
  static uint32_t FTK_OPT_DATA_DIR = 0u;
  static string OPT_DIR;

  if (FTK_OPT_DATA_DIR == 0u) {
    if (ftkEnumerateOptions(lib, 0uLL, getDataDirOptionId, &FTK_OPT_DATA_DIR) !=
        ftkError::FTK_WAR_OPT_GLOBAL_ONLY) {
      cerr << "Could not get the data directory option ID\n";
      return false;
    }
    if (FTK_OPT_DATA_DIR == 0u) {
      cerr << "Could not get the data directory option ID\n";
      return false;
    }
  }

  if (OPT_DIR.empty()) {
    ftkBuffer buffer{};
    if (ftkGetData(lib, 0uLL, FTK_OPT_DATA_DIR, &buffer) != ftkError::FTK_OK ||
        buffer.size < 1u) {
      return false;
    }
    OPT_DIR.assign(reinterpret_cast<const char*>(buffer.data), buffer.size);
    if (!OPT_DIR.empty() && OPT_DIR.back() == '\0') {
      OPT_DIR.pop_back();
    }
  }

  const string candidate = OPT_DIR + "/" + fileName;
  ifstream system_input{candidate};
  if (system_input.is_open()) {
    fullFilePath = candidate;
    if (fromSystem != nullptr) *fromSystem = true;
    return true;
  }

  return false;
}

bool loadFileInBuffer(const string& fullFilePath, ftkBuffer& buffer) {
  ifstream input(fullFilePath, ios::binary | ios::ate);
  if (!input.is_open()) {
    cerr << "Could not open file '" << fullFilePath << "'\n";
    return false;
  }

  buffer.reset();
  streampos pos = input.tellg();
  if (pos < 0 || pos > std::numeric_limits<uint32_t>::max()) {
    cerr << "File too large for buffer\n";
    return false;
  }

  if (static_cast<size_t>(pos) > buffer.capacity) {
    if (buffer.resize(static_cast<size_t>(pos)) != ftkError::FTK_OK) {
      cerr << "Failed to resize ftkBuffer for file\n";
      return false;
    }
  }

  input.seekg(0u, ios::beg);
  input.read(buffer.data, static_cast<streamsize>(pos));
  if (input.fail()) {
    cerr << "Could not read file '" << fullFilePath << "'\n";
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
