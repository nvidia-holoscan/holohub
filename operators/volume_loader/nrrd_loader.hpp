/* SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef VOLUME_LOADER_NRRD_LOADER
#define VOLUME_LOADER_NRRD_LOADER

#include <memory>
#include <string>
#include <vector>

namespace holoscan::ops {

class Volume;

bool is_nrrd(const std::string& file_name);

bool load_nrrd(const std::string& file_name, Volume& volume);
}  // namespace holoscan::ops

#endif /* VOLUME_LOADER_NRRD_LOADER */
