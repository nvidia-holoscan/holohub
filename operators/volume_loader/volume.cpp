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

#include "volume.hpp"

namespace holoscan::ops {

bool Volume::SetOrientation(const std::string& orientation) {
  if (orientation.length() != 3) { return false; }

  uint32_t rl_axis = 4;
  uint32_t is_axis = 4;
  uint32_t pa_axis = 4;

  bool rl_flip = false;
  bool is_flip = false;
  bool pa_flip = false;

  for (int axis = 0; axis < 3; ++axis) {
    if (orientation[axis] == 'R' || orientation[axis] == 'r') { rl_axis = axis; }

    if (orientation[axis] == 'L' || orientation[axis] == 'l') {
      rl_axis = axis;
      rl_flip = true;
    }

    if (orientation[axis] == 'I' || orientation[axis] == 'i') { is_axis = axis; }

    if (orientation[axis] == 'S' || orientation[axis] == 's') {
      is_axis = axis;
      is_flip = true;
    }

    if (orientation[axis] == 'P' || orientation[axis] == 'p') { pa_axis = axis; }

    if (orientation[axis] == 'A' || orientation[axis] == 'a') {
      pa_axis = axis;
      pa_flip = true;
    }
  }

  if ((rl_axis == 4) || (is_axis == 4) || (pa_axis == 4)) { return false; }

  permute_axis_ = {rl_axis, is_axis, pa_axis};
  flip_axes_ = {rl_flip, is_flip, pa_flip};
  return true;
}

}  // namespace holoscan::ops
