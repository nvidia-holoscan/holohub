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

#include <ftkTypes.h>

#include <string>

struct ftkLibraryImp;
typedef ftkLibraryImp* ftkLibrary;
struct ftkBuffer;
struct ftkRigidBody;

bool getFullFilePath(ftkLibrary lib, const std::string& fileName, std::string& fullFilePath,
                     bool* fromSystem = nullptr);
bool loadFileInBuffer(const std::string& fullFilePath, ftkBuffer& buffer);
int loadRigidBody(ftkLibrary lib, const std::string& fileName, ftkRigidBody& geometry);
