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

#include "xr_transform_render_op.hpp"

#include <nlohmann/json.hpp>

#include <holoviz/imgui/imgui.h>

namespace holoscan::openxr {

struct TransferFunction {
  float opacity;
  std::string name;
};

struct XrTransformRenderOp::Params {
  nlohmann::json settings;
  std::vector<TransferFunction> transfer_functions;
};

void XrTransformRenderOp::setup(OperatorSpec& spec) {
  spec.input<nvidia::gxf::Pose3D>("left_camera_pose").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::Pose3D>("right_camera_pose").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::CameraModel>("left_camera_model").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::CameraModel>("right_camera_model").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::Vector2f>("depth_range");
  spec.input<holoscan::gxf::Entity>("color_buffer_in");
  spec.input<holoscan::gxf::Entity>("depth_buffer_in");
  spec.input<UxBoundingBox>("ux_box");
  spec.input<UxCursor>("ux_cursor");
  spec.input<UxWindow>("ux_window");

  spec.output<holoscan::gxf::Entity>("color_buffer_out");
  spec.output<holoscan::gxf::Entity>("depth_buffer_out");
  spec.output<nlohmann::json>("render_settings").condition(ConditionType::kNone);

  spec.param(display_width_, "display_width", "Display Width", "Display width in pixels", {});
  spec.param(display_height_, "display_height", "Display Height", "Display height in pixels", {});
  spec.param(config_file_,
             "config_file",
             "Configuration file",
             "Configuration file",
             std::string("./configs/ctnv_bb_er.json"));

  render_params_.reset(new Params);
}

void XrTransformRenderOp::start() {
  // Create a cuda stream for synchronizing CUDA events with the vulkan device queue.
  if (cudaStreamCreate(&cuda_stream_) != cudaSuccess) {
    throw std::runtime_error("cudaStreamCreate failed");
  }

  //  Initialize HoloViz. Assume left and right displays are vertically stacked
  instance_ = viz::Create();
  viz::SetCurrent(instance_);
  viz::Init(display_width_.get(), display_height_.get() * 2, "Holoviz", viz::InitFlags::HEADLESS);
  viz::SetCudaStream(cuda_stream_);

  // setup ImGui
  float scale = 4.f;
  ImGuiStyle& style = ImGui::GetStyle();
  style.ScaleAllSizes(scale);
  ImGuiIO& io = ImGui::GetIO();
  io.FontGlobalScale *= scale;

  // load transfer functions
  std::ifstream input_file_stream(config_file_.get());
  render_params_->settings = nlohmann::json::parse(input_file_stream);
  nlohmann::json transfer_functions = render_params_->settings["TransferFunction"];

  for (auto component : transfer_functions.at("components")) {
    TransferFunction entry;
    entry.opacity = component["opacity"].get<float>();
    if (component.contains("name")) {
      entry.name = component["name"].get<std::string>();
    } else {
      entry.name = "unnamed";
    }

    render_params_->transfer_functions.push_back(entry);
  }
}

void XrTransformRenderOp::stop() {
  cudaStreamDestroy(cuda_stream_);
  if (instance_) { viz::Shutdown(instance_); }
}

void XrTransformRenderOp::compute(InputContext& input, OutputContext& output,
                                  ExecutionContext& context) {
  auto color_message = input.receive<holoscan::gxf::Entity>("color_buffer_in").value();
  auto depth_message = input.receive<holoscan::gxf::Entity>("depth_buffer_in").value();
  auto depth_range = input.receive<nvidia::gxf::Vector2f>("depth_range");

  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> color_buffer =
      static_cast<nvidia::gxf::Entity&>(color_message).get<nvidia::gxf::VideoBuffer>().value();
  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> depth_buffer =
      static_cast<nvidia::gxf::Entity&>(depth_message).get<nvidia::gxf::VideoBuffer>().value();

  auto color_cuda_events = color_message.findAll<nvidia::gxf::CudaEvent>().value();
  for (auto cuda_event : color_cuda_events) {
    if (cuda_event.has_value() && cuda_event.value()->event().has_value()) {
      if (cudaStreamWaitEvent(cuda_stream_, cuda_event.value()->event().value()) != cudaSuccess) {
        throw std::runtime_error("cudaStreamWaitEvent failed");
      }
    }
  }

  auto depth_cuda_events = depth_message.findAll<nvidia::gxf::CudaEvent>().value();
  for (auto cuda_event : depth_cuda_events) {
    if (cuda_event.has_value() && cuda_event.value()->event().has_value()) {
      if (cudaStreamWaitEvent(cuda_stream_, cuda_event.value()->event().value()) != cudaSuccess) {
        throw std::runtime_error("cudaStreamWaitEvent failed");
      }
    }
  }

  viz::SetCurrent(instance_);

  viz::Begin();

  // render volume dataset
  viz::BeginImageLayer();
  viz::ImageCudaDevice(color_buffer->video_frame_info().width,
                       color_buffer->video_frame_info().height,
                       viz::ImageFormat::R8G8B8A8_UNORM,
                       reinterpret_cast<CUdeviceptr>(color_buffer->pointer()));
  viz::ImageCudaDevice(depth_buffer->video_frame_info().width,
                       depth_buffer->video_frame_info().height,
                       viz::ImageFormat::D32_SFLOAT,
                       reinterpret_cast<CUdeviceptr>(depth_buffer->pointer()));
  viz::EndLayer();

  // calculate let and right view matrices
  auto left_pose = toEigen(*input.receive<nvidia::gxf::Pose3D>("left_camera_pose"));
  auto left_model = input.receive<nvidia::gxf::CameraModel>("left_camera_model");
  auto right_pose = toEigen(*input.receive<nvidia::gxf::Pose3D>("right_camera_pose"));
  auto right_model = input.receive<nvidia::gxf::CameraModel>("right_camera_model");

  std::array<float, 16> left_matrix;
  left_matrix = toModelView(left_pose, *left_model, depth_range->x, depth_range->y);
  std::array<float, 16> right_matrix;
  right_matrix = toModelView(right_pose, *right_model, depth_range->x, depth_range->y);

  // render ux widgets
  auto ux_box = input.receive<UxBoundingBox>("ux_box");
  auto ux_cursor = input.receive<UxCursor>("ux_cursor");
  auto ux_window = input.receive<UxWindow>("ux_window");

  viz::BeginGeometryLayer();
  viz::LayerAddView(0.0, 0.0, 1.0, 0.5, left_matrix.data());
  viz::LayerAddView(0.0, 0.5, 1.0, 0.5, right_matrix.data());
  ui_box_renderer_.render(ux_box.value(),
                          0.5 * (right_pose.translation() + left_pose.translation()));
  ui_window_renderer_.render(ux_window.value());
  viz::EndLayer();

  // render imGui

  // translate controller events to imGui
  ImGuiIO& io = ImGui::GetIO();
  io.DisplayFramebufferScale.x = 1;
  io.DisplayFramebufferScale.y = 1;
  io.DeltaTime = 1.0f / 60.0f;
  if (ux_window->face.range > 0) {
    io.MouseDrawCursor = true;
    io.MousePos = ImVec2(ux_window->cursor(0) * display_width_.get(),
                         (1.0 - ux_window->cursor(1)) * display_height_.get());
    io.MouseDown[0] = ux_window->button;
  } else {
    io.MouseDrawCursor = false;
    io.MousePos = ImVec2(0, 0);
    io.MouseDown[0] = false;
  }

  Eigen::Affine3f ui_transform;
  ui_transform.setIdentity();
  ui_transform.scale(Eigen::Vector3f{ux_window->content(0), -ux_window->content(1) * 2, 1.f});
  ui_transform.translate(Eigen::Vector3f{0.f, 0.5f, 0.001f});
  ui_transform = ux_window->transform * ui_transform;

  left_matrix =
      toModelView(ui_transform.inverse() * left_pose, *left_model, depth_range->x, depth_range->y);
  right_matrix = toModelView(
      ui_transform.inverse() * right_pose, *right_model, depth_range->x, depth_range->y);

  // render window widgets
  viz::BeginImGuiLayer();
  viz::LayerAddView(0.0, 0.0, 1.0, 0.5, left_matrix.data());
  viz::LayerAddView(0.0, 0.5, 1.0, 0.5, right_matrix.data());

  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2(display_width_.get(), display_height_.get()));

  // calculate text width
  float textWidth = 0.0f;
  for (int i = 0; i < render_params_->transfer_functions.size(); i++) {
    TransferFunction& entry = render_params_->transfer_functions[i];
    textWidth = std::max(textWidth, ImGui::CalcTextSize(entry.name.c_str()).x);
  }
  ImGuiStyle& style = ImGui::GetStyle();
  textWidth += 2 * style.WindowPadding.x;

  bool modified = false;
  ImGui::Begin("",
               0,
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_NoCollapse | ImGuiDockNodeFlags_NoDockingInCentralNode);
  for (int i = 0; i < render_params_->transfer_functions.size(); i++) {
    TransferFunction& entry = render_params_->transfer_functions[i];
    ImGui::Text("%s", entry.name.c_str());
    ImGui::SameLine(textWidth);
    std::string name = "##" + std::to_string(i);
    modified |= ImGui::SliderFloat(name.c_str(), &entry.opacity, 0.0f, 1.0f);
  }

  ImGui::End();

  viz::EndLayer();
  viz::End();

  // emit render settings
  if (modified) {
    nlohmann::json& transfer_functions = render_params_->settings["TransferFunction"];
    for (int i = 0; i < transfer_functions["components"].size(); i++) {
      nlohmann::json& component = transfer_functions["components"].at(i);
      TransferFunction entry = render_params_->transfer_functions[i];
      component["opacity"] = entry.opacity;
    }
    output.emit(render_params_->settings, "render_settings");
  }

  viz::ReadFramebuffer(viz::ImageFormat::R8G8B8A8_UNORM,
                       color_buffer->video_frame_info().width,
                       color_buffer->video_frame_info().height,
                       color_buffer->size(),
                       reinterpret_cast<CUdeviceptr>(color_buffer->pointer()));

  nvidia::gxf::Handle<nvidia::gxf::CudaEvent> color_cuda_event =
      static_cast<nvidia::gxf::Entity&>(color_message).add<nvidia::gxf::CudaEvent>().value();
  color_cuda_event->init();
  if (cudaEventRecord(color_cuda_event->event().value(), cuda_stream_) != cudaSuccess) {
    throw std::runtime_error("cudaEventRecord failed");
  }

  output.emit(color_message, "color_buffer_out");

  viz::ReadFramebuffer(viz::ImageFormat::D32_SFLOAT,
                       depth_buffer->video_frame_info().width,
                       depth_buffer->video_frame_info().height,
                       depth_buffer->size(),
                       reinterpret_cast<CUdeviceptr>(depth_buffer->pointer()));

  nvidia::gxf::Handle<nvidia::gxf::CudaEvent> depth_cuda_event =
      static_cast<nvidia::gxf::Entity&>(depth_message).add<nvidia::gxf::CudaEvent>().value();
  depth_cuda_event->init();
  if (cudaEventRecord(depth_cuda_event->event().value(), cuda_stream_) != cudaSuccess) {
    throw std::runtime_error("cudaEventRecord failed");
  }

  output.emit(depth_message, "depth_buffer_out");
}

std::array<float, 16> XrTransformRenderOp::toModelView(Eigen::Affine3f world,
                                                       nvidia::gxf::CameraModel camera,
                                                       float near_z, float far_z) {
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

Eigen::Affine3f XrTransformRenderOp::toEigen(nvidia::gxf::Pose3D pose) {
  Eigen::Affine3f matrix;
  matrix.setIdentity();
  matrix.translation() = Eigen::Map<Eigen::Vector3f>(pose.translation.data());
  matrix.matrix().block(0, 0, 3, 3) =
      Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(pose.rotation.data());
  return matrix;
}

}  // namespace holoscan::openxr
