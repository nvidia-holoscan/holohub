/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_OPENXR_UX_UX_WIDGETS_HPP
#define HOLOSCAN_OPERATORS_OPENXR_UX_UX_WIDGETS_HPP

#include <Eigen/Dense>
#include <array>

namespace holoscan::openxr {

#define BOX_MIN_EXTENT 0.1f  // minimum edge length in meters

enum UxAction { IDLE, DRAGGING };

struct UxEdge {
  UxAction action;

  // inverse normalized distance from cursor to projection point on edge [0,1]
  float range;

  // normalized distance of cursor projection along edge
  float projection;
};

struct UxCorner {
  UxAction action;

  // inverse normalized distance from cursor to action point at corner [0,1]
  float range;
};

struct UxFace {
  UxAction action;

  // inverse normalized distance from cursor to projected cursor [0,1]
  float range;

  // u,v coordinates of cursor projection onto plane
  Eigen::Vector2f projection;
};

#define UX_ACTIVATION_THRESHOLD 0.1f  // %of length

struct UxBoundingBox {
  //
  //        |   -z
  //        y   /
  //        |  /
  //        | /
  //        +---x----+

  // the transformatino of the bounding box within the
  // coordinate frame of the volumetric dataset. This
  // will be used to compute the clipping planes
  Eigen::Affine3f local_transform;

  // the transformation of the bounding box within the
  // global coordinate syste. This will be used to compute
  // the transformation matrix of the volumetric dataset
  Eigen::Affine3f global_transform;

  // scaling fator not included in transforms
  float scale;

  Eigen::Vector3f half_extent;

  //           +---10---+
  //          11       /|
  //         /        9 |
  //        +---8----+  6
  //        |        |  |
  //        4        5  +
  //        |        | 1
  //        |        |/
  //        +---0----+

  UxEdge edges[12];

  //           7--------6
  //          /        /|
  //         /        / |
  //        4--------5  |
  //        |        |  |
  //        |        |  2
  //        |        | /
  //        |        |/
  //        0--------1

  UxCorner corners[8];

  //           +--------+
  //          /   5    /|
  //         /        / |
  //        +--------+  |
  //        |        | 2|
  //        |        |  +
  //        |   1    | /
  //        |        |/
  //        +--------+

  UxFace faces[6];
};

struct UxCursor {
  UxAction state;

  // normalized distance from cursor to activation surface [0,1]
  float range;

  Eigen::Affine3f transform;
};
}  // namespace holoscan::openxr

#endif  // HOLOSCAN_OPERATORS_OPENXR_UX_UX_WIDGETS_HPP
