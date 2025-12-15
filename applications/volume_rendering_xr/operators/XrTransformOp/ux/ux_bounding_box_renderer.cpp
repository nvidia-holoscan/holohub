/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "ux_bounding_box_renderer.hpp"
#include "holoscan/holoscan.hpp"

#include <holoviz/imgui/imgui.h>
#include <holoviz/holoviz.hpp>

#include <algorithm>
#include <iostream>

namespace holoscan::openxr {

void UxBoundingBoxRenderer::drawCorner(UxCorner& state, Eigen::Vector3f point, Eigen::Vector3f dx,
                                       Eigen::Vector3f dy, Eigen::Vector3f dz,
                                       Eigen::Vector3f& eye_pos) {
  if (state.range > 0) {
    float data[6 * 3];

    // compute displacement to avoid z-fighting
    Eigen::Vector3f view_displacement = 0.005 * (eye_pos - point).normalized();

    dx = point + dx + view_displacement;
    dy = point + dy + view_displacement;
    dz = point + dz + view_displacement;

    Eigen::Map<Eigen::Vector<float, 6 * 3>>(data) << point, dx, point, dy, point, dz;

    viz::Primitive(viz::PrimitiveTopology::LINE_LIST_3D, 3, sizeof(data) / sizeof(data[0]), data);
  }
}

void UxBoundingBoxRenderer::drawEdge(UxEdge& state, Eigen::Vector3f p0, Eigen::Vector3f p1,
                                     Eigen::Vector3f& eye_pos) {
  if (state.range > 0) {
    Eigen::Vector3f edge = (p1 - p0);
    Eigen::Vector3f focal_point = p0 + (p1 - p0) * state.projection;

    // compute displacement to avoid z-fighting
    Eigen::Vector3f view_displacement = 0.005 * (eye_pos - focal_point).normalized();

    float segment_length = edge.norm() * UX_ACTIVATION_THRESHOLD;
    Eigen::Vector3f d = edge.normalized();
    const float L = 0.05;  // length of edge handle in meters
    Eigen::Vector3f s0 = focal_point - 0.5 * segment_length * d + view_displacement;
    Eigen::Vector3f s1 = focal_point + 0.5 * segment_length * d + view_displacement;

    float data[2 * 3];
    Eigen::Map<Eigen::Vector<float, 2 * 3>>(data) << s0, s1;

    viz::Primitive(viz::PrimitiveTopology::LINE_LIST_3D, 1, sizeof(data) / sizeof(data[0]), data);
  }
}

void UxBoundingBoxRenderer::drawFace(UxFace& state, Eigen::Vector3f p0, Eigen::Vector3f pu,
                                     Eigen::Vector3f pv, Eigen::Affine3f& transform) {
  if (state.range > 0) {
    Eigen::Vector3f u = pu - p0;
    Eigen::Vector3f v = pv - p0;
    Eigen::Vector3f focal_point = p0 + u * state.projection[0] + v * state.projection[1];

    Eigen::Vector3f du = u.normalized();
    Eigen::Vector3f dv = v.normalized();

    float L = std::min(u.norm(), v.norm()) * UX_ACTIVATION_THRESHOLD;
    Eigen::Vector3f u0 = focal_point + du * L;
    Eigen::Vector3f u1 = focal_point - du * L;
    Eigen::Vector3f v0 = focal_point + dv * L;
    Eigen::Vector3f v1 = focal_point - dv * L;

    float data[4 * 3];
    Eigen::Map<Eigen::Vector<float, 4 * 3>>(data) << u0, u1, v0, v1;

    viz::Primitive(viz::PrimitiveTopology::LINE_LIST_3D, 2, sizeof(data) / sizeof(data[0]), data);
  }
}

void UxBoundingBoxRenderer::drawAxes(float length) {
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

void UxBoundingBoxRenderer::drawOutline(Eigen::Vector3f vertices[8]) {
  float data5[5 * 3];

  // bottom edges
  Eigen::Map<Eigen::Vector<float, 5 * 3>>(data5) << vertices[0], vertices[1], vertices[2],
      vertices[3], vertices[0];
  viz::Primitive(viz::PrimitiveTopology::LINE_STRIP_3D, 4, sizeof(data5) / sizeof(data5[0]), data5);

  // top edges
  Eigen::Map<Eigen::Vector<float, 5 * 3>>(data5) << vertices[4], vertices[5], vertices[6],
      vertices[7], vertices[4];
  viz::Primitive(viz::PrimitiveTopology::LINE_STRIP_3D, 4, sizeof(data5) / sizeof(data5[0]), data5);

  // vertical edges
  float data8[8 * 3];
  Eigen::Map<Eigen::Vector<float, 8 * 3>>(data8) << vertices[0], vertices[4], vertices[1],
      vertices[5], vertices[2], vertices[6], vertices[3], vertices[7];
  viz::Primitive(viz::PrimitiveTopology::LINE_LIST_3D, 4, sizeof(data8) / sizeof(data8[0]), data8);
}

void UxBoundingBoxRenderer::render(UxBoundingBox& box, Eigen::Vector3f eye_pos) {
  //
  //        |   -z
  //        y   /
  //        |  /
  //        | /
  //        +---x----+

  // drawAxes(1.0);
  //  showAll(box);

  Eigen::Affine3f T = box.global_transform * box.local_transform;
  float hx = box.half_extent[0];
  float hy = box.half_extent[1];
  float hz = box.half_extent[2];

  //           7--------6
  //          /        /|
  //         /        / |
  //        4--------5  |
  //        |        |  |
  //        |        |  2
  //        |        | /
  //        |        |/
  //        0--------1

  Eigen::Vector3f vertices[8];
  vertices[0] = T * Eigen::Vector3f(-hx, -hy, hz);
  vertices[1] = T * Eigen::Vector3f(hx, -hy, hz);
  vertices[2] = T * Eigen::Vector3f(hx, -hy, -hz);
  vertices[3] = T * Eigen::Vector3f(-hx, -hy, -hz);
  vertices[4] = T * Eigen::Vector3f(-hx, hy, hz);
  vertices[5] = T * Eigen::Vector3f(hx, hy, hz);
  vertices[6] = T * Eigen::Vector3f(hx, hy, -hz);
  vertices[7] = T * Eigen::Vector3f(-hx, hy, -hz);

  // outline
  viz::LineWidth(6.f);
  viz::Color(0.8f, 0.8f, 0.8f, 0.5f);
  drawOutline(vertices);

  // controls
  viz::LineWidth(10.f);
  viz::Color(0.145f, 0.478, 0.992, 1.f);

  Eigen::Matrix3f R = T.rotation().matrix();
  Eigen::Vector3f dx = R.col(0) * hx * UX_ACTIVATION_THRESHOLD * 2;
  Eigen::Vector3f dy = R.col(1) * hy * UX_ACTIVATION_THRESHOLD * 2;
  Eigen::Vector3f dz = R.col(2) * hz * UX_ACTIVATION_THRESHOLD * 2;

  drawCorner(box.corners[0], vertices[0], dx, dy, -dz, eye_pos);
  drawCorner(box.corners[1], vertices[1], -dx, dy, -dz, eye_pos);
  drawCorner(box.corners[2], vertices[2], -dx, dy, dz, eye_pos);
  drawCorner(box.corners[3], vertices[3], dx, dy, dz, eye_pos);
  drawCorner(box.corners[4], vertices[4], dx, -dy, -dz, eye_pos);
  drawCorner(box.corners[5], vertices[5], -dx, -dy, -dz, eye_pos);
  drawCorner(box.corners[6], vertices[6], -dx, -dy, dz, eye_pos);
  drawCorner(box.corners[7], vertices[7], dx, -dy, dz, eye_pos);

  // edges

  //           +---10---+
  //          11       /|
  //         /        9 |
  //        +---8----+  6
  //        |        |  |
  //        4        5  +
  //        |        | 1
  //        |        |/
  //        +---0----+

  drawEdge(box.edges[0], vertices[0], vertices[1], eye_pos);
  drawEdge(box.edges[1], vertices[1], vertices[2], eye_pos);
  drawEdge(box.edges[2], vertices[2], vertices[3], eye_pos);
  drawEdge(box.edges[3], vertices[3], vertices[0], eye_pos);

  drawEdge(box.edges[4], vertices[0], vertices[4], eye_pos);
  drawEdge(box.edges[5], vertices[1], vertices[5], eye_pos);
  drawEdge(box.edges[6], vertices[2], vertices[6], eye_pos);
  drawEdge(box.edges[7], vertices[3], vertices[7], eye_pos);

  drawEdge(box.edges[8], vertices[4], vertices[5], eye_pos);
  drawEdge(box.edges[9], vertices[5], vertices[6], eye_pos);
  drawEdge(box.edges[10], vertices[6], vertices[7], eye_pos);
  drawEdge(box.edges[11], vertices[7], vertices[4], eye_pos);

  // faces

  //           +--------+
  //          /   5    /|
  //         /        / |
  //        +--------+  |
  //        |        | 2|
  //        |        |  +
  //        |   1    | /
  //        |        |/
  //        +--------+

  drawFace(box.faces[0], vertices[3], vertices[2], vertices[0], T);
  drawFace(box.faces[1], vertices[0], vertices[1], vertices[4], T);
  drawFace(box.faces[2], vertices[1], vertices[2], vertices[5], T);
  drawFace(box.faces[3], vertices[2], vertices[3], vertices[6], T);
  drawFace(box.faces[4], vertices[3], vertices[0], vertices[7], T);
  drawFace(box.faces[5], vertices[4], vertices[5], vertices[7], T);
}

void UxBoundingBoxRenderer::showAll(UxBoundingBox& box) {
  float hx = box.half_extent[0];
  float hy = box.half_extent[1];
  float hz = box.half_extent[2];

  for (int i = 0; i < 6; i++) {
    box.faces[i].range = 1;
  }
  box.faces[0].projection = Eigen::Vector2f(0.3f, 0.3);
  box.faces[1].projection = Eigen::Vector2f(0.3f, 0.3);
  box.faces[2].projection = Eigen::Vector2f(0.3f, 0.3);
  box.faces[3].projection = Eigen::Vector2f(0.3f, 0.3);
  box.faces[4].projection = Eigen::Vector2f(0.3f, 0.3);
  box.faces[5].projection = Eigen::Vector2f(0.3f, 0.3);

  for (int i = 0; i < 12; i++) {
    box.edges[i].range = 1;
    box.edges[i].projection = 0.3;
  }

  for (int i = 0; i < 8; i++) {
    box.corners[i].range = 1;
  }
}

}  // namespace holoscan::openxr
