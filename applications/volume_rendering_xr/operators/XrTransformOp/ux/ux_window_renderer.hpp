/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef HOLOSCAN_OPERATORS_OPENXR_UX_UX_WINDOW_RENDERER_HPP
#define HOLOSCAN_OPERATORS_OPENXR_UX_UX_WINDOW_RENDERER_HPP

#include <array>
#include <vector>

#include "ux_widgets.hpp"

namespace holoscan::openxr {
class UxWindowRenderer {
 public:
  void render(UxWindow& box);

 private:
  void drawAxes(float length);
};
}  // namespace holoscan::openxr

#endif  // HOLOSCAN_OPERATORS_OPENXR_UX_UX_WINDOW_RENDERER_HPP
