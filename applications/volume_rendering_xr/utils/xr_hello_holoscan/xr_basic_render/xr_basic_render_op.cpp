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

#include "xr_basic_render_op.hpp"
#include <holoviz/imgui/imgui.h>
#include <random>
#define _USE_MATH_DEFINES
#include <cmath>

namespace holoscan::ops {

inline constexpr std::array<ImVec4, 3> AxisColors = {
    ImVec4{1.f, 0.f, 0.f, 1.f},
    ImVec4{0.f, 1.f, 0.f, 1.f},
    ImVec4{0.f, 0.f, 1.f, 1.f},
};

void BasicRenderOp::setup(OperatorSpec& spec) {
  spec.input<nvidia::gxf::Pose3D>("left_camera_pose").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::Pose3D>("right_camera_pose").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::CameraModel>("left_camera_model").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::CameraModel>("right_camera_model").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::Vector2f>("depth_range");
  spec.input<holoscan::gxf::Entity>("color_buffer_in");
  spec.input<holoscan::gxf::Entity>("depth_buffer_in");

  spec.input<nvidia::gxf::Pose3D>("head_pose").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::Pose3D>("aim_pose").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::Pose3D>("grip_pose").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::Pose3D>("eye_gaze_pose").condition(ConditionType::kNone);

  spec.input<bool>("trigger_click");
  spec.input<bool>("shoulder_click");
  spec.input<std::array<float, 2>>("trackpad");
  spec.input<bool>("trackpad_touch");

  spec.output<holoscan::gxf::Entity>("color_buffer_out");
  spec.output<holoscan::gxf::Entity>("depth_buffer_out");

  spec.param(display_width_, "display_width", "Display Width", "Display width in pixels", {});
  spec.param(display_height_, "display_height", "Display Height", "Display height in pixels", {});
}

void BasicRenderOp::start() {
  // Create a cuda stream for synchronizing CUDA events with the vulkan device queue.
  cudaStreamCreate(&cuda_stream_);
  //  Initialize HoloViz. Assume left and right displays are vertically stacked
  instance_ = viz::Create();
  viz::SetCurrent(instance_);
  viz::Init(display_width_.get(), display_height_.get() * 2, "Holoviz", viz::InitFlags::HEADLESS);
}

void BasicRenderOp::stop() {
  cudaStreamDestroy(cuda_stream_);
  viz::Shutdown(instance_);
  instance_ = nullptr;
}

void BasicRenderOp::compute(InputContext& input, OutputContext& output, ExecutionContext& context) {
  auto color_message_xr = input.receive<holoscan::gxf::Entity>("color_buffer_in").value();
  auto depth_message_xr = input.receive<holoscan::gxf::Entity>("depth_buffer_in").value();

  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> color_buffer_xr =
      static_cast<nvidia::gxf::Entity&>(color_message_xr).get<nvidia::gxf::VideoBuffer>().value();
  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> depth_buffer_xr =
      static_cast<nvidia::gxf::Entity&>(depth_message_xr).get<nvidia::gxf::VideoBuffer>().value();

  viz::SetCurrent(instance_);

  drawVizLayers(input);

  viz::ReadFramebuffer(viz::ImageFormat::R8G8B8A8_UNORM,
                       color_buffer_xr->video_frame_info().width,
                       color_buffer_xr->video_frame_info().height,
                       color_buffer_xr->size(),
                       reinterpret_cast<CUdeviceptr>(color_buffer_xr->pointer()));

  nvidia::gxf::Handle<nvidia::gxf::CudaEvent> color_cuda_event =
      static_cast<nvidia::gxf::Entity&>(color_message_xr).add<nvidia::gxf::CudaEvent>().value();
  color_cuda_event->init();
  if (cudaEventRecord(color_cuda_event->event().value(), cuda_stream_) != cudaSuccess) {
    throw std::runtime_error("cudaEventRecord failed");
  }

  output.emit(color_message_xr, "color_buffer_out");

  viz::ReadFramebuffer(viz::ImageFormat::D32_SFLOAT,
                       depth_buffer_xr->video_frame_info().width,
                       depth_buffer_xr->video_frame_info().height,
                       depth_buffer_xr->size(),
                       reinterpret_cast<CUdeviceptr>(depth_buffer_xr->pointer()));

  nvidia::gxf::Handle<nvidia::gxf::CudaEvent> depth_cuda_event =
      static_cast<nvidia::gxf::Entity&>(depth_message_xr).add<nvidia::gxf::CudaEvent>().value();
  depth_cuda_event->init();
  if (cudaEventRecord(depth_cuda_event->event().value(), cuda_stream_) != cudaSuccess) {
    throw std::runtime_error("cudaEventRecord failed");
  }

  output.emit(depth_message_xr, "depth_buffer_out");
}

void BasicRenderOp::drawVizLayers(InputContext& input) {
  auto depth_range_xr = input.receive<nvidia::gxf::Vector2f>("depth_range");
  auto left_model = input.receive<nvidia::gxf::CameraModel>("left_camera_model");
  auto right_model = input.receive<nvidia::gxf::CameraModel>("right_camera_model");

  const auto get_pose_or_id = [&](const char* input_name) {
    if (auto result = input.receive<nvidia::gxf::Pose3D>(input_name)) {
      return toEigen(result.value());
    }
    return Eigen::Affine3f::Identity();
  };
  // calculate let and right view matrices
  auto left_pose = get_pose_or_id("left_camera_pose");
  auto right_pose = get_pose_or_id("right_camera_pose");
  head_pose_ = get_pose_or_id("head_pose");
  aim_pose_ = get_pose_or_id("aim_pose");
  grip_pose_ = get_pose_or_id("grip_pose");
  eye_gaze_pose_ = get_pose_or_id("eye_gaze_pose");

  auto left_matrix = toModelView(left_pose, *left_model, depth_range_xr->x, depth_range_xr->y);
  auto right_matrix = toModelView(right_pose, *right_model, depth_range_xr->x, depth_range_xr->y);

  viz::Begin();

  viz::BeginGeometryLayer();
  {
    viz::LayerAddView(0.0, 0.0, 1.0, 0.5, left_matrix.data());
    viz::LayerAddView(0.0, 0.5, 1.0, 0.5, right_matrix.data());
    drawGeometryLayer(input);
  }
  viz::EndLayer();

  viz::BeginImGuiLayer();
  {
    Eigen::Affine3f ui_transform = Eigen::Affine3f::Identity();
    ui_transform.fromPositionOrientationScale(Eigen::Vector3f{0.f, -1.f, -2.5f},
                                              Eigen::Quaternionf::Identity(),
                                              Eigen::Vector3f{0.75f, -1.5f, 1.f});

    left_matrix = toModelView(
        ui_transform.inverse() * left_pose, *left_model, depth_range_xr->x, depth_range_xr->y);
    right_matrix = toModelView(
        ui_transform.inverse() * right_pose, *right_model, depth_range_xr->x, depth_range_xr->y);

    viz::LayerAddView(0.0, 0.0, 1.0, 0.5, left_matrix.data());
    viz::LayerAddView(0.0, 0.5, 1.0, 0.5, right_matrix.data());
    drawImGuiLayer(input);
  }
  viz::EndLayer();

  viz::End();
}

void BasicRenderOp::drawImGuiLayer(InputContext& input) {
  const auto tracked_pad_data =
      input.receive<std::array<float, 2>>("trackpad").value_or(std::array<float, 2>{0.f, 0.f});
  Eigen::Vector2f trackpad = Eigen::Vector2f::Map(tracked_pad_data.data());

  trigger_clicked_ = input.receive<bool>("trigger_click").value_or(false);
  bool shoulder_click = input.receive<bool>("shoulder_click").value_or(false);
  bool trackpad_touch = input.receive<bool>("trackpad_touch").value_or(false);

  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2(display_width_.get(), display_height_.get()));

  constexpr const ImGuiWindowFlags RootWinFlags =
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
      ImGuiDockNodeFlags_NoDockingInCentralNode;

  if (ImGui::Begin("HelloWorld", nullptr, RootWinFlags)) {
    constexpr const ImVec4 XrPuple{120 / 255.0f, 43 / 255.0f, 144 / 255.0f, 1.0f};

    ImGui::SetWindowFontScale(16.f);
    ImGui::TextColored(XrPuple, "Hello World!");

    ImGui::SetWindowFontScale(6.f);
    ImGui::NewLine();
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.f), "Input States");
    ImGui::Separator();

    ImGui::BeginDisabled();
    ImGui::Checkbox("Trigger Clicked", &trigger_clicked_);
    ImGui::Checkbox("Shoulder Clicked", &shoulder_click);
    ImGui::Checkbox("Trackpad Touched", &trackpad_touch);
    ImGui::EndDisabled();
    ImGui::Text("Trackpad: ");
    ImGui::SameLine();
    ImGuiVec2Text(trackpad);

    ImGui::NewLine();
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.f), "Input Poses");
    ImGui::Separator();

    ImGui::Text("Eye Gaze:");
    ImGui::Indent();
    ImGuiPoseText(eye_gaze_pose_);
    ImGui::Unindent();

    ImGui::Text("Aim Pose:");
    ImGui::Indent();
    ImGuiPoseText(aim_pose_);
    ImGui::Unindent();

    ImGui::Text("Grip Pose:");
    ImGui::Indent();
    ImGuiPoseText(grip_pose_);
    ImGui::Unindent();
  }
  ImGui::End();
}

void BasicRenderOp::drawGeometryLayer(InputContext& input) {
  // draw world origin
  drawAxes(Eigen::Affine3f::Identity(), 5.0f, 0.5f);
  // draw some arbitrary positioned & orientated cubes in the world.
  drawWorldCubes();
  drawController();
  drawEyeGaze();
}

void BasicRenderOp::drawWorldCubes() {
  static constexpr const float deg_to_rad = static_cast<float>(M_PI / 180.0);
  std::size_t axis_index = 0;
  for (auto cube_dir : {
           Eigen::Vector2f{-1.f, -1.0f},
           Eigen::Vector2f{-1.f, 0.f},
           Eigen::Vector2f{1.f, -1.0f},
           Eigen::Vector2f{1.f, 0.0f},
       }) {
    Eigen::Vector3f axis = Eigen::Vector3f::Zero();
    axis(axis_index) = 1.f;
    axis_index = (axis_index + 1) % 3;
    const Eigen::AngleAxisf rot(45.0f * deg_to_rad, axis);
    auto c1 = Eigen::Affine3f::Identity();
    c1.translate(Eigen::Vector3f(cube_dir.x() * 1.5f, 0, cube_dir.y() * 2.5f))
        .rotate(rot)
        .scale(0.4f);
    drawCube(c1);
  }
}

void BasicRenderOp::drawController() {
  Eigen::Affine3f grip_cube_tr = grip_pose_;
  grip_cube_tr.scale(Eigen::Vector3f(0.04f, 0.04f, 0.07f))
      .translate(Eigen::Vector3f(0.f, 0.f, 0.17f));
  drawCube(grip_cube_tr);

  drawAxes(aim_pose_, 1.5f, 0.05f);
  Eigen::Affine3f aim_pose_cube_tr = aim_pose_;
  const float click_scale = trigger_clicked_ ? 0.03f : 0.01f;
  aim_pose_cube_tr.scale(click_scale);
  drawCube(aim_pose_cube_tr);
}

void BasicRenderOp::drawEyeGaze() {
  Eigen::Affine3f eye_tr = head_pose_ * eye_gaze_pose_;
  eye_tr.translate(Eigen::Vector3f(0.f, 0.f, -1.5f));

  std::array<float, 3> eye_point;
  Eigen::Vector3f::Map(eye_point.data()) = eye_tr.translation();

  viz::Color(0.f, 0.4f, 0.f, 1.f);
  viz::PointSize(8.f);
  viz::Primitive(viz::PrimitiveTopology::POINT_LIST_3D, 1, eye_point.size(), eye_point.data());
}

void BasicRenderOp::drawCube(const Eigen::Affine3f& pose) {
  static const std::array<Eigen::Vector3f, 8> TriVerts = {
      Eigen::Vector3f{-0.5f, -0.5f, -0.5f},
      {0.5f, -0.5f, -0.5f},
      {0.5f, 0.5f, -0.5f},
      {-0.5f, 0.5f, -0.5f},
      {-0.5f, -0.5f, 0.5f},
      {0.5f, -0.5f, 0.5f},
      {0.5f, 0.5f, 0.5f},
      {-0.5f, 0.5f, 0.5f},
  };
  using tri_indices = std::array<std::size_t, 3>;
  static constexpr const std::array<tri_indices, 12> TriIndices = {
      tri_indices
      // Right face
      {1, 5, 6},
      {6, 2, 1},
      // Left face
      {4, 0, 3},
      {3, 7, 4},
      // Top face
      {3, 2, 6},
      {6, 7, 3},
      // Bottom face
      {4, 5, 1},
      {1, 0, 4},
      // Back face
      {5, 4, 7},
      {7, 6, 5},
      // Front face
      {0, 1, 2},
      {2, 3, 0},
  };
  static constexpr const std::array<ImVec4, 6> SideColors = {
      AxisColors[0],
      ImVec4(0.5f, 0.f, 0.f, 1.f),
      AxisColors[1],
      ImVec4(0.f, 0.5f, 0.f, 1.f),
      AxisColors[2],
      ImVec4(0.f, 0.0f, 0.5f, 1.f),
  };

  for (std::size_t tri_index = 0; tri_index < 12; tri_index += 2) {
    const auto& t1 = TriIndices[tri_index];
    const auto& t2 = TriIndices[tri_index + 1];

    const std::array<Eigen::Vector3f, 6> face = {
        TriVerts[t1[0]],
        TriVerts[t1[1]],
        TriVerts[t1[2]],
        TriVerts[t2[0]],
        TriVerts[t2[1]],
        TriVerts[t2[2]],
    };

    std::size_t vidx = 0;
    std::array<float, 18> tr_verts;
    for (const auto& vert : face) {
      Eigen::Vector3f::Map(tr_verts.data() + vidx) = pose * vert;
      vidx += 3;
    }

    const auto& sideColor = SideColors[tri_index / 2];
    viz::Color(sideColor.x, sideColor.y, sideColor.z, sideColor.w);
    viz::Primitive(viz::PrimitiveTopology::TRIANGLE_LIST_3D, 2, tr_verts.size(), tr_verts.data());
  }
}

void BasicRenderOp::drawAxes(const Eigen::Affine3f& pose, const float lineWidth,
                             const float scale /*= 1.0f*/) {
  const Eigen::Vector3f origin = pose.translation();
  const Eigen::Matrix3f axes = pose.linear();

  float line_points[2 * 3];
  Eigen::Vector3f::Map(line_points) = origin;

  viz::LineWidth(lineWidth);
  for (std::size_t axis_index = 0; axis_index < 3; ++axis_index) {
    const ImVec4& axisColor = AxisColors[axis_index];
    viz::Color(axisColor.x, axisColor.y, axisColor.z, axisColor.w);

    const Eigen::Vector3f axis = axes.col(axis_index);
    const Eigen::Vector3f p2 = scale * axis + origin;
    Eigen::Vector3f::Map(line_points + 3) = p2;
    viz::Primitive(viz::PrimitiveTopology::LINE_LIST_3D,
                   1,
                   sizeof(line_points) / sizeof(line_points[0]),
                   line_points);
  }
}

std::array<float, 16> BasicRenderOp::toModelView(Eigen::Affine3f world,
                                                 nvidia::gxf::CameraModel camera, float near_z,
                                                 float far_z) {
  Eigen::Matrix4f projection = Eigen::Matrix4f::Zero();
  float x_range = 0.5 * camera.dimensions.x / camera.focal_length.x;
  float y_range = 0.5 * camera.dimensions.y / camera.focal_length.y;
  projection(0, 0) = 1 / x_range;
  projection(0, 2) = 0;
  projection(1, 1) = -1 / y_range;
  projection(1, 2) = 0;
  projection(2, 2) = -(far_z) / (far_z - near_z);
  projection(2, 3) = -(far_z * near_z) / (far_z - near_z);
  projection(3, 2) = -1;

  std::array<float, 16> matrix;
  Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(matrix.data()) =
      projection * world.matrix().inverse();

  return matrix;
}

Eigen::Affine3f BasicRenderOp::toEigen(nvidia::gxf::Pose3D pose) {
  Eigen::Affine3f matrix;
  matrix.setIdentity();
  matrix.translation() = Eigen::Map<Eigen::Vector3f>(pose.translation.data());
  matrix.matrix().block(0, 0, 3, 3) =
      Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(pose.rotation.data());
  return matrix;
}

void BasicRenderOp::ImGuiVec2Text(const Eigen::Vector2f& p) {
  ImGui::TextColored(AxisColors[0], "x: %.2f ", p.x());
  ImGui::SameLine();
  ImGui::TextColored(AxisColors[1], "y: %.2f", p.y());
}

void BasicRenderOp::ImGuiVec3Text(const Eigen::Vector3f& p) {
  ImGuiVec2Text(p.head<2>());
  ImGui::SameLine();
  ImGui::TextColored(AxisColors[2], " z: %.2f", p.z());
}

void BasicRenderOp::ImGuiQuatText(const Eigen::Quaternionf& q) {
  ImGuiVec3Text(q.vec());
  ImGui::SameLine();
  ImGui::TextColored(ImVec4{1.f, 1.0f, 0.f, 1.f}, " w: %.2f", q.w());
}

void BasicRenderOp::ImGuiPoseText(const Eigen::Affine3f& q) {
  constexpr const ImVec4 label_color(0.5f, 0.5f, 0.5f, 1.f);
  ImGui::TextColored(label_color, "Position: ");
  ImGui::SameLine();
  ImGuiVec3Text(q.translation());

  ImGui::TextColored(label_color, "Orientation: ");
  ImGui::SameLine();
  ImGuiQuatText(Eigen::Quaternionf{q.rotation()});
}

}  // namespace holoscan::ops
