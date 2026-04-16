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

#include <array>

namespace holoscan::ops {

struct CameraCalibration {
  float fx{0.0F};
  float fy{0.0F};
  float cx{0.0F};
  float cy{0.0F};
  float skew{0.0F};
  std::array<float, 5> distortion{};
  int image_width{0};
  int image_height{0};

  [[nodiscard]] bool valid() const { return image_width > 0 && image_height > 0; }
};

}  // namespace holoscan::ops
