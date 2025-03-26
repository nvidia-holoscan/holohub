/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "ux_window_renderer.hpp"
#include "holoscan/holoscan.hpp"

#include <holoviz/imgui/imgui.h>
#include <holoviz/holoviz.hpp>

#include <algorithm>
#include <iostream>

namespace holoscan::openxr {

void UxWindowRenderer::drawAxes(float length) {
  float data[3 * 2];
  data[0] = 0;
  data[1] = 0;
  data[2] = -1.5;

  viz::LineWidth(1.f);

  data[3] = length;
  data[4] = 0;
  data[5] = -1.5;
  viz::Color(1.f, 0.f, 0.f, 1.f);
  viz::Primitive(viz::PrimitiveTopology::LINE_LIST_3D, 1, sizeof(data) / sizeof(data[0]), data);

  data[3] = 0;
  data[4] = length;
  data[5] = -1.5;
  viz::Color(0.f, 1.f, 0.f, 1.f);
  viz::Primitive(viz::PrimitiveTopology::LINE_LIST_3D, 1, sizeof(data) / sizeof(data[0]), data);

  data[3] = 0;
  data[4] = 0;
  data[5] = -1.5 + length;
  viz::Color(0.f, 0.f, 1.f, 1.f);
  viz::Primitive(viz::PrimitiveTopology::LINE_LIST_3D, 1, sizeof(data) / sizeof(data[0]), data);
}

void UxWindowRenderer::render(UxWindow& window) {
  //
  //        |   -z
  //        y   /
  //        |  /
  //        | /
  //        +---x----+

  // drawAxes(2.0);
  //  showAll(box);
  Eigen::Affine3f T = window.transform;

  float hx = window.content[0];
  float hy = window.content[1];
  float hz = window.content[2];

  //        3--------2
  //        |        |
  //        |        |
  //        |        |
  //        |        |
  //        0--------1

  Eigen::Vector3f vertices[8];
  vertices[0] = T * Eigen::Vector3f(-hx, -hy, 0.0);
  vertices[1] = T * Eigen::Vector3f(hx, -hy, 0.0);
  vertices[2] = T * Eigen::Vector3f(hx, hy, 0.0);
  vertices[3] = T * Eigen::Vector3f(-hx, hy, 0.0);

  vertices[4] = T * Eigen::Vector3f(hx, hy + HEADER_HEIGHT, 0.0);
  vertices[5] = T * Eigen::Vector3f(-hx, hy + HEADER_HEIGHT, 0.0);

  // header
  if (window.handle.action == holoscan::openxr::DRAGGABLE ||
      window.handle.action == holoscan::openxr::DRAGGING) {
    viz::Color(0.145f, 0.478, 0.992, 1.f);
  } else {
    viz::Color(0.8f, 0.8f, 0.8f, 0.5f);
  }

  float data[6 * 3];
  Eigen::Map<Eigen::Vector<float, 6 * 3>>(data) << vertices[3], vertices[2], vertices[4],
      vertices[4], vertices[5], vertices[3];
  viz::Primitive(viz::PrimitiveTopology::TRIANGLE_LIST_3D, 2, sizeof(data) / sizeof(data[0]), data);

  // background depth buffer fill
  {
    float data[6 * 3];
    viz::Color(1.0f, 0.0f, 0.0f, 0.f);
    Eigen::Map<Eigen::Vector<float, 6 * 3>>(data) << vertices[0], vertices[1], vertices[2],
        vertices[2], vertices[3], vertices[0];
    viz::Primitive(
        viz::PrimitiveTopology::TRIANGLE_LIST_3D, 2, sizeof(data) / sizeof(data[0]), data);
  }
}

}  // namespace holoscan::openxr
