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

#include "ux_bounding_box_controller.hpp"
#include "holoscan/holoscan.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <iostream>

// #define DEBUG_PRINT 1

namespace holoscan::openxr {

Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ",", ",");

// TODO is this in Eigen ?
static float clamp(float v, float min, float max) {
  return (v < min) ? min : (v > max) ? max : v;
}

UxBoundingBoxController::Edge::Edge(int _p0_x, int _p0_y, int _p0_z, int _p1_x, int _p1_y,
                                    int _p1_z, UxEdge& _state, bool _enabled)
    : p0_x(_p0_x),
      p0_y(_p0_y),
      p0_z(_p0_z),
      p1_x(_p1_x),
      p1_y(_p1_y),
      p1_z(_p1_z),
      state(_state),
      enabled(_enabled) {}

UxBoundingBoxController::Face::Face(int _n_sign, int _n, int _u_sign, int _u, int _v_sign, int _v,
                                    UxFace& _state)
    : n_sign(_n_sign), n(_n), u_sign(_u_sign), u(_u), v_sign(_v_sign), v(_v), state(_state) {}

UxBoundingBoxController::Corner::Corner(int _sign_x, int _sign_y, int _sign_z, UxCorner& _state)
    : sign_x(_sign_x), sign_y(_sign_y), sign_z(_sign_z), state(_state) {}

UxBoundingBoxController::UxBoundingBoxController(UxBoundingBox& boundingBox)
    : widget_(0),
      active_action_(UNDEFINED),
      pending_action_(UNDEFINED),
      box_(boundingBox),
      global_transform_(boundingBox.global_transform),
      local_transform_(boundingBox.local_transform) {
  global_transform_inv_ = global_transform_.inverse();
  local_transform_inv_ = local_transform_.inverse();

  //
  //        |   -z
  //        y   /
  //        |  /
  //        | /
  //        +---x----+

  int X = 0;
  int Y = 1;
  int Z = 2;

  axes_[X] = Eigen::Vector3f(1, 0, 0);
  axes_[Y] = Eigen::Vector3f(0, 1, 0);
  axes_[Z] = Eigen::Vector3f(0, 0, 1);

  const int P = 1;   // positive
  const int N = -1;  // negative

  // add edges

  //           +---10---+
  //          11       /|
  //         /        9 |
  //        +---8----+  6
  //        |        |  |
  //        4        5  +
  //        |        | 1
  //        |        |/
  //        +---0----+

  edges_.push_back(Edge(N, N, P, P, N, P, box_.edges[0], true));
  edges_.push_back(Edge(P, N, P, P, N, N, box_.edges[1], true));
  edges_.push_back(Edge(P, N, N, N, N, N, box_.edges[2], true));
  edges_.push_back(Edge(N, N, N, N, N, P, box_.edges[3], true));
  edges_.push_back(Edge(N, N, P, N, P, P, box_.edges[4], true));
  edges_.push_back(Edge(P, N, P, P, P, P, box_.edges[5], true));
  edges_.push_back(Edge(P, N, N, P, P, N, box_.edges[6], true));
  edges_.push_back(Edge(N, N, N, N, P, N, box_.edges[7], true));
  edges_.push_back(Edge(N, P, P, P, P, P, box_.edges[8], true));
  edges_.push_back(Edge(P, P, P, P, P, N, box_.edges[9], true));
  edges_.push_back(Edge(P, P, N, N, P, N, box_.edges[10], true));
  edges_.push_back(Edge(N, P, N, N, P, P, box_.edges[11], true));

  // add faces

  //           +--------+
  //          /   5    /|
  //         /        / |
  //        +--------+  |
  //        |        | 2|
  //        |        |  +
  //        |   1    | /
  //        |        |/
  //        +--------+

  faces_.push_back(Face(N, Y, N, X, N, Z, box_.faces[0]));
  faces_.push_back(Face(P, Z, N, X, N, Y, box_.faces[1]));
  faces_.push_back(Face(P, X, P, Z, N, Y, box_.faces[2]));
  faces_.push_back(Face(N, Z, P, X, N, Y, box_.faces[3]));
  faces_.push_back(Face(N, X, N, Z, N, Y, box_.faces[4]));
  faces_.push_back(Face(P, Y, N, X, P, Z, box_.faces[5]));

  // add corners

  //           7--------6
  //          /        /|
  //         /        / |
  //        4--------5  |
  //        |        |  |
  //        |        |  2
  //        |        | /
  //        |        |/
  //        0--------1

  corners_.push_back(Corner(N, N, P, box_.corners[0]));
  corners_.push_back(Corner(P, N, P, box_.corners[1]));
  corners_.push_back(Corner(P, N, N, box_.corners[2]));
  corners_.push_back(Corner(N, N, N, box_.corners[3]));
  corners_.push_back(Corner(N, P, P, box_.corners[4]));
  corners_.push_back(Corner(P, P, P, box_.corners[5]));
  corners_.push_back(Corner(P, P, N, box_.corners[6]));
  corners_.push_back(Corner(N, P, N, box_.corners[7]));
}

void UxBoundingBoxController::reset() {
  global_transform_ = box_.global_transform;
  local_transform_ = box_.local_transform;
  global_transform_inv_ = global_transform_.inverse();
  local_transform_inv_ = local_transform_.inverse();

  pending_action_ = UNDEFINED;
  active_action_ = UNDEFINED;
  box_.state = holoscan::openxr::INACTIVE;
}

void UxBoundingBoxController::trackPadDown(Eigen::Vector2f trackpad) {
  start_trackpad_ = trackpad;
  start_extent_ = box_.half_extent;
  start_scale_ = box_.scale;
}

void UxBoundingBoxController::trackPadMove(Eigen::Vector2f trackpad) {
  static constexpr float kDeadzone = 0.15f;
  if (std::abs(trackpad.y() - start_trackpad_.y()) > kDeadzone) {
    float delta = trackpad.y() - start_trackpad_.y();
    float scale = 0.25f * (fabs(delta) - kDeadzone) * (delta < 0.0f ? -1.0f : 1.0f);

    box_.scale = std::max(0.50f, std::min(start_scale_ + scale, 5.0f));

    // grow extents
    float ds = box_.scale / start_scale_;
    box_.half_extent[0] = start_extent_[0] * ds;
    box_.half_extent[1] = start_extent_[1] * ds;
    box_.half_extent[2] = start_extent_[2] * ds;

    // move bbox center
    Eigen::Vector3f T = local_transform_.translation();
    box_.local_transform.translation() = T * ds;

    // make sure box center remains in place
    Eigen::Vector3f dT = T * (1 - ds);
    dT = global_transform_inv_.rotation().matrix() * dT;

    Eigen::Vector3f global_translation = global_transform_.translation();
    box_.global_transform.translation() = global_translation + dT;
  }
}

void UxBoundingBoxController::trackPadUp() {
  global_transform_ = box_.global_transform;
  local_transform_ = box_.local_transform;
  global_transform_inv_ = global_transform_.inverse();
  local_transform_inv_ = local_transform_.inverse();
}

#define RECORD_RANGE(widget, action) \
  if (range > pending_range) {       \
    pending_range = range;           \
    pending_action_ = action;        \
    widget_ = i;                     \
  }

void UxBoundingBoxController::cursorMove(Eigen::Affine3f pose) {
  // convert to local coordinate system
  Eigen::Affine3f local_pose = local_transform_inv_ * global_transform_inv_ * pose;
  Eigen::Vector3f current_cursor = local_pose.translation();

  switch (active_action_) {
    case UNDEFINED: {
      // reset activation range for all widgets
      for (int i = 0; i < edges_.size(); i++) { edges_[i].state.range = 0; }
      for (int i = 0; i < faces_.size(); i++) { faces_[i].state.range = 0; }
      for (int i = 0; i < corners_.size(); i++) { corners_[i].state.range = 0; }

      if (!test_box(current_cursor)) {
        // find closest widget
        float pending_range = 0.0;
        pending_action_ = UNDEFINED;
        widget_ = -1;

        // prescendence 1
        for (int i = 0; i < corners_.size(); i++) {
          float range = test_corner(current_cursor, corners_[i]);
          RECORD_RANGE(range, DRAG_CORNER)
        }

        if (pending_range == 0.0f) {
          // prescendence 2
          for (int i = 0; i < edges_.size(); i++) {
            float range = test_edge(current_cursor, edges_[i]);
            RECORD_RANGE(range, DRAG_EDGE)
          }

          if (pending_range == 0.0f) {
            // prescendence 3
            for (int i = 0; i < faces_.size(); i++) {
              float range = test_face(current_cursor, faces_[i]);
              RECORD_RANGE(range, DRAG_FACE)
            }
          }
        }

        switch (pending_action_) {
          case DRAG_FACE:
            faces_[widget_].state.range = pending_range;
            break;
          case DRAG_EDGE:
            edges_[widget_].state.range = pending_range;
            break;
          case DRAG_CORNER:
            corners_[widget_].state.range = pending_range;
            break;
        }

      } else {
        pending_action_ = DRAG_BOX;
      }

#ifdef DEBUG_PRINT
      printPending();
#endif
    } break;
    case DRAG_EDGE: {
      drag_edge(current_cursor, edges_[widget_]);
    } break;
    case DRAG_FACE: {
      drag_face(current_cursor, faces_[widget_]);
    } break;

    case DRAG_CORNER: {
      drag_corner(current_cursor, corners_[widget_]);
    } break;

    case DRAG_BOX: {
      drag_box(current_cursor);
    } break;
  }

  box_.state = pending_action_ == UNDEFINED && active_action_ == UNDEFINED
                   ? holoscan::openxr::INACTIVE
                   : holoscan::openxr::ACTIVE;
}

void UxBoundingBoxController::cursorClick(Eigen::Affine3f pose) {
  // convert to local coordinate system
  Eigen::Affine3f local_pose = local_transform_inv_ * global_transform_inv_ * pose;
  Eigen::Vector3f point = local_pose.translation();

  active_action_ = pending_action_;
  switch (active_action_) {
    case DRAG_EDGE: {
      Edge& edge = edges_[widget_];
      edge.state.action = DRAGGING;
    } break;

    case DRAG_FACE: {
      Face& face = faces_[widget_];
      face.state.action = DRAGGING;
    } break;

    case DRAG_CORNER: {
      Corner& corner = corners_[widget_];
      corner.state.action = DRAGGING;
    }
    case DRAG_BOX: {
    } break;
  }

  start_cursor_ = point;
  start_extent_ = box_.half_extent;
  start_scale_ = box_.scale;
}

void UxBoundingBoxController::cursorRelease() {
  switch (active_action_) {
    case DRAG_EDGE: {
      Edge& edge = edges_[widget_];
    } break;

    case DRAG_FACE: {
      Face& face = faces_[widget_];
    } break;

    case DRAG_CORNER: {
      Corner& corner = corners_[widget_];
    } break;
    case DRAG_BOX: {
    } break;
  }

  local_transform_ = box_.local_transform;
  local_transform_inv_ = local_transform_.inverse();
  global_transform_ = box_.global_transform;
  global_transform_inv_ = global_transform_.inverse();

  pending_action_ = UNDEFINED;
  active_action_ = UNDEFINED;
  box_.state = holoscan::openxr::INACTIVE;
}

float UxBoundingBoxController::test_edge(Eigen::Vector3f& cursor, Edge& edge) {
  // find distance from cursor to line
  Eigen::Vector3f p0(edge.p0_x * box_.half_extent[0],
                     edge.p0_y * box_.half_extent[1],
                     edge.p0_z * box_.half_extent[2]);
  Eigen::Vector3f p1(edge.p1_x * box_.half_extent[0],
                     edge.p1_y * box_.half_extent[1],
                     edge.p1_z * box_.half_extent[2]);

  Eigen::Vector3f v0 = cursor - p0;
  Eigen::Vector3f v1 = cursor - p1;
  Eigen::Vector3f segment = p1 - p0;
  float segment_length = segment.norm();
  float distance = v0.cross(v1).norm() / segment_length;

  float activation_distance = BOX_MIN_EXTENT;  //

  if (distance < activation_distance) {
    // test if projected point is on the edge
    Eigen::Vector3f n = segment / segment_length;
    float d = n.dot(v0);
    if (d > activation_distance / 2 && d < segment_length - activation_distance / 2) {
      edge.state.projection = d / segment_length;
      return 1.0f - clamp(distance / activation_distance, 0, 1);
    }
  }
  return 0;
}

void UxBoundingBoxController::drag_edge(Eigen::Vector3f& cursor, Edge& edge) {
  Eigen::Vector3f p0(edge.p0_x * box_.half_extent[0],
                     edge.p0_y * box_.half_extent[1],
                     edge.p0_z * box_.half_extent[2]);
  Eigen::Vector3f p1(edge.p1_x * box_.half_extent[0],
                     edge.p1_y * box_.half_extent[1],
                     edge.p1_z * box_.half_extent[2]);
  Eigen::Vector3f axis = (p1 - p0).normalized();

  Eigen::Vector3f s0 = start_cursor_.normalized();
  Eigen::Vector3f s1 = cursor.normalized();

  Eigen::Vector3f v0 = axis.cross(s0);
  Eigen::Vector3f v1 = axis.cross(s1);

  float angle = s1.dot(v0) * acos(v0.dot(v1));

  Eigen::Affine3f T = global_transform_ * local_transform_ * Eigen::AngleAxisf(angle, axis);
  box_.global_transform = T * local_transform_inv_;
}

float UxBoundingBoxController::test_face(Eigen::Vector3f& cursor, Face& face) {
  float du = box_.half_extent[face.u];
  float dv = box_.half_extent[face.v];
  // check if point is within face rectangle on plane
  if (cursor[face.u] > -du && cursor[face.u] < du && cursor[face.v] > -dv && cursor[face.v] < dv) {
    // check if point is on positive side of plane
    Eigen::Vector3f normal = face.n_sign * axes_[face.n];
    float distance = normal.dot(cursor) - fabs(box_.half_extent[face.n]);
    if (distance > 0) {
      float activation_distance = BOX_MIN_EXTENT;
      face.state.projection[0] = fabs(cursor[face.u] - face.u_sign * du) / (2 * du);
      face.state.projection[1] = fabs(cursor[face.v] - face.v_sign * dv) / (2 * dv);
      return 1.0f - clamp(distance / activation_distance, 0, 1);
    }
  }
  return 0;
}

void UxBoundingBoxController::drag_face(Eigen::Vector3f& cursor, Face& face) {
  /* DO NOT DELETE yet --
  Eigen::Vector3f normal = face.n_sign * axes_[face.n];
  float displacement = 0.5 * (normal.dot(cursor) - normal.dot(start_cursor_));
  if (start_extent_[face.n] + displacement > 0.1) {
    // shrink box by 1/2 displacement
    box_.half_extent[face.n] = start_extent_[face.n] + displacement;

    // move box by 1/2 displacement
    box_.local_transform = local_transform_;
    box_.local_transform.translate(displacement * normal);
  }
  */

  Eigen::Vector3f normal = face.n_sign * axes_[face.n];
  float delta = normal.dot(cursor) - normal.dot(start_cursor_);
  float ds = std::max(0.1f, (delta + start_extent_[face.n]) / start_extent_[face.n]);

  // grow extents
  box_.half_extent[0] = start_extent_[0] * ds;
  box_.half_extent[1] = start_extent_[1] * ds;
  box_.half_extent[2] = start_extent_[2] * ds;
  box_.scale = start_scale_ * ds;

  // move local bbox center
  Eigen::Vector3f T = local_transform_.translation();
  box_.local_transform.translation() = T * ds;

  // make sure
  // a) global box center remains in place
  // b) activated plane remains stationary
  Eigen::Vector3f dT = T * (ds - 1);
  dT = dT + delta * normal;

  dT = global_transform_.rotation() * dT;

  T = global_transform_.translation();
  box_.global_transform.translation() = T - dT;
}

float UxBoundingBoxController::test_corner(Eigen::Vector3f& cursor, Corner& corner) {
  // find distance from cursor to corner
  Eigen::Vector3f p0(corner.sign_x * box_.half_extent[0],
                     corner.sign_y * box_.half_extent[1],
                     corner.sign_z * box_.half_extent[2]);
  float distance = (cursor - p0).norm();
  float activation_distance = BOX_MIN_EXTENT;

  return 1.0f - clamp(distance / activation_distance, 0, 1);
}

void UxBoundingBoxController::drag_corner(Eigen::Vector3f& cursor, Corner& corner) {
  Eigen::Vector3f signs(corner.sign_x, corner.sign_y, corner.sign_z);
  Eigen::Vector3f delta = cursor - start_cursor_;

  // move box by 1/2 displacement
  box_.local_transform = local_transform_;
  box_.local_transform.translate(0.5 * delta);

  // shrink box by 1/2 displacement
  delta = signs.array() * delta.array();
  box_.half_extent = start_extent_ + 0.5 * delta;

  box_.half_extent[0] =
      std::max(2 * start_extent_[0] * UX_ACTIVATION_THRESHOLD, box_.half_extent[0]);
  box_.half_extent[1] =
      std::max(2 * start_extent_[1] * UX_ACTIVATION_THRESHOLD, box_.half_extent[1]);
  box_.half_extent[2] =
      std::max(2 * start_extent_[2] * UX_ACTIVATION_THRESHOLD, box_.half_extent[2]);
}

bool UxBoundingBoxController::test_box(Eigen::Vector3f& cursor) {
  bool inside = cursor(0) > -box_.half_extent[0] && cursor(0) < box_.half_extent[0] &&
                cursor(1) > -box_.half_extent[1] && cursor(1) < box_.half_extent[1] &&
                cursor(2) > -box_.half_extent[2] && cursor(2) < box_.half_extent[2];
  for (int i = 0; i < 8; i++) { corners_[i].state.range = inside ? 1.0 : 0.0; }
  return inside;
}

void UxBoundingBoxController::drag_box(Eigen::Vector3f& cursor) {
  box_.global_transform = global_transform_;
  box_.global_transform.translate(cursor - start_cursor_);
}

void UxBoundingBoxController::printPending() {
  switch (pending_action_) {
    case DRAG_EDGE: {
      Edge& edge = edges_[widget_];
      Eigen::Vector3f p0(edge.p0_x * box_.half_extent[0],
                         edge.p0_y * box_.half_extent[1],
                         edge.p0_z * box_.half_extent[2]);
      Eigen::Vector3f p1(edge.p1_x * box_.half_extent[0],
                         edge.p1_y * box_.half_extent[1],
                         edge.p1_z * box_.half_extent[2]);
      holoscan::log_error("EDGE-{}: {}", widget_, edge.state.range);
    } break;
    case DRAG_FACE: {
      Face& face = faces_[widget_];
      holoscan::log_error("FACE-{}: {}", widget_, face.state.range);
    } break;
    case DRAG_CORNER: {
      Corner& corner = corners_[widget_];
      holoscan::log_error("CORNER-{}: {}", widget_, corner.state.range);
    } break;
  }
}
}  // namespace holoscan::openxr
