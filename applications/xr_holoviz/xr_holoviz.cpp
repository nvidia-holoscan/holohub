/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>

#include <memory>
#include <string>

#include "holoscan/holoscan.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"
#include "xr_begin_frame_op.hpp"
#include "xr_composition_layers.hpp"
#include "xr_end_frame_op.hpp"
#include "xr_session.hpp"
#include "holoviz/holoviz.hpp"
#include "openxr/openxr.hpp"

#include "holoscan/operators/holoviz/holoviz.hpp"
#include "xr_view_helper.hpp"
#include "xr_buffer_composition.hpp"
#include "xr_manager.hpp"

// Constants for torus generation
constexpr float TORUS_MAJOR_RADIUS = 0.3f;  // Distance from center to tube center
constexpr float TORUS_MINOR_RADIUS = 0.1f;  // Radius of the tube
constexpr int TORUS_MAJOR_SEGMENTS = 32;    // Number of segments around the major radius
constexpr int TORUS_MINOR_SEGMENTS = 16;    // Number of segments around the minor radius

// Generate torus vertices - now using triangles (3 vertices per triangle)
std::array<std::array<float, 3>, TORUS_MAJOR_SEGMENTS * TORUS_MINOR_SEGMENTS * 6> torus_array;

// Helper function to generate torus vertices
void generate_torus_vertices() {
  int vertex_index = 0;
  for (int i = 0; i < TORUS_MAJOR_SEGMENTS; ++i) {
    float major_angle = (2.0f * M_PI * i) / TORUS_MAJOR_SEGMENTS;
    float next_major_angle = (2.0f * M_PI * (i + 1)) / TORUS_MAJOR_SEGMENTS;
    float cos_major = cosf(major_angle);
    float sin_major = sinf(major_angle);
    float cos_next_major = cosf(next_major_angle);
    float sin_next_major = sinf(next_major_angle);

    for (int j = 0; j < TORUS_MINOR_SEGMENTS; ++j) {
      float minor_angle = (2.0f * M_PI * j) / TORUS_MINOR_SEGMENTS;
      float next_minor_angle = (2.0f * M_PI * (j + 1)) / TORUS_MINOR_SEGMENTS;

      // Calculate all four vertices for this quad
      auto get_vertex = [&](float major_angle, float minor_angle) {
        float cos_minor = cosf(minor_angle);
        float sin_minor = sinf(minor_angle);
        float cos_major = cosf(major_angle);
        float sin_major = sinf(major_angle);

        float x = (TORUS_MAJOR_RADIUS + TORUS_MINOR_RADIUS * cos_minor) * cos_major;
        float y = (TORUS_MAJOR_RADIUS + TORUS_MINOR_RADIUS * cos_minor) * sin_major;
        float z = TORUS_MINOR_RADIUS * sin_minor;

        // Add subtle variation for visual interest
        float variation = 0.05f * sinf(major_angle * 2.0f) * cosf(minor_angle * 3.0f);
        return std::array<float, 3>{x + variation, y + variation, z + variation};
      };

      // Get the four vertices of the quad
      auto v1 = get_vertex(major_angle, minor_angle);
      auto v2 = get_vertex(major_angle, next_minor_angle);
      auto v3 = get_vertex(next_major_angle, next_minor_angle);
      auto v4 = get_vertex(next_major_angle, minor_angle);

      // Create two triangles for this quad
      // First triangle
      torus_array[vertex_index++] = v1;
      torus_array[vertex_index++] = v2;
      torus_array[vertex_index++] = v3;

      // Second triangle
      torus_array[vertex_index++] = v1;
      torus_array[vertex_index++] = v3;
      torus_array[vertex_index++] = v4;
    }
  }
}

namespace holoscan::ops {
class XrGeometrySourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(XrGeometrySourceOp)

  XrGeometrySourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<xr::FrameState>("xr_frame_state");
    spec.param(xr_manager_, "xr_manager");
    spec.param(allocator_, "allocator");

    spec.output<gxf::Entity>("outputs");
    spec.output<std::vector<HolovizOp::InputSpec>>("output_specs");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto xr_session = xr_manager_->get_xr_session();
    auto frame_state = op_input.receive<xr::FrameState>("xr_frame_state");

    auto entity = gxf::Entity::New(&context);
    auto specs = std::vector<HolovizOp::InputSpec>();

    /* ======
     Add geometry to the entity by using add_data.
     ======
    */

    // Initialize the torus vertices
    generate_torus_vertices();
    add_data<TORUS_MAJOR_SEGMENTS * TORUS_MINOR_SEGMENTS * 6, 3>(
        entity,  // 6 vertices per quad (2 triangles × 3 vertices)
        "torus",
        torus_array,
        context);

    add_data<2, 2>(entity, "dynamic_text", {{0.F, 0.F}}, context);

    /* ======
     Create input specs for holovizs with stereo views.
     ======
    */

    // Get the located views with current frame state, cached in xr_manager
    // This is shared by all objects
    auto located_views = xr_manager_->update_located_views(*frame_state);

    // Create model matrix for each objects
    // Rotate the cube with display time
    glm::mat4 model_matrix =
        glm::rotate(glm::mat4{1},
                    static_cast<float>(frame_state->predictedDisplayTime.get()) / 1'000'000'000,
                    glm::vec3(0, -1, 0));

    HolovizOp::InputSpec torus_spec = XrViewsHelper::create_spec_with_views(
        "torus", HolovizOp::InputType::TRIANGLES_3D, located_views, xr_session, model_matrix);

    specs.push_back(torus_spec);

    HolovizOp::InputSpec text_spec = XrViewsHelper::create_spec_with_views(
        "dynamic_text", HolovizOp::InputType::TEXT, located_views, xr_session, model_matrix);
    text_spec.text_.push_back("Hello, World!");
    specs.push_back(text_spec);

    // emit outputs
    op_output.emit(entity, "outputs");
    op_output.emit(specs, "output_specs");
  }

 private:
  // Helper function to add a tensor with data to an entity.
  template <std::size_t N, std::size_t C>
  void add_data(gxf::Entity& entity, const char* name,
                const std::array<std::array<float, C>, N>& data, ExecutionContext& context) {
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                         allocator_->gxf_cid());
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>(name).value();
    tensor->reshape<float>(
        nvidia::gxf::Shape({N, C}), nvidia::gxf::MemoryStorageType::kHost, allocator.value());
    std::memcpy(tensor->pointer(), data.data(), N * C * sizeof(float));
  }

  Parameter<std::shared_ptr<holoscan::UnboundedAllocator>> allocator_;
  Parameter<std::shared_ptr<holoscan::XrManager>> xr_manager_;
};

}  // namespace holoscan::ops

class HolovizGeometryApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto xr_session = make_resource<holoscan::XrSession>(
        "xr_session", holoscan::Arg("application_name") = std::string("XR Render Cube"));

    auto xr_manager =
        make_resource<holoscan::XrManager>("xr_manager", Arg("xr_session") = xr_session);

    auto xr_begin_frame = make_operator<holoscan::ops::XrBeginFrameOp>(
        "xr_begin_frame", holoscan::Arg("xr_session") = xr_session);
    auto xr_end_frame = make_operator<holoscan::ops::XrEndFrameOp>(
        "xr_end_frame", holoscan::Arg("xr_session") = xr_session);

    auto source = make_operator<ops::XrGeometrySourceOp>(
        "source",
        Arg("xr_manager") = xr_manager,
        Arg("allocator") = make_resource<holoscan::UnboundedAllocator>("allocator"));

    // TODO: width and height are hardcoded for now, can't get the width and height from headset at
    // this point
    auto holoviz_args = from_config("holoviz");
    auto visualizer = make_operator<ops::HolovizOp>(
        "holoviz",
        // Arg("enable_render_buffer_input", true),
        Arg("enable_render_buffer_output", true),
        Arg("headless", true),
        Arg("allocator") = make_resource<holoscan::UnboundedAllocator>("allocator"),
        holoviz_args);

    auto buffer_composition = make_operator<ops::XrBufferCompositionOp>(
        "buffer_composition", Arg("xr_manager") = xr_manager);

    // source -> holoviz
    add_flow(source, visualizer, {{"outputs", "receivers"}});
    add_flow(source, visualizer, {{"output_specs", "input_specs"}});
    // add_flow(source, visualizer, {{"render_buffer_output", "render_buffer_input"}});
    // The core OpenXR render loop: begin frame -> end frame.
    add_flow(xr_begin_frame, xr_end_frame, {{"xr_frame_state", "xr_frame_state"}});
    add_flow(xr_begin_frame, source, {{"xr_frame_state", "xr_frame_state"}});

    add_flow(
        visualizer, buffer_composition, {{"render_buffer_output", "color_render_buffer_output"}});
    add_flow(buffer_composition, xr_end_frame, {{"xr_composition_layer", "xr_composition_layers"}});
  }
};

int main(int argc, char** argv) {
  // Get the configuration
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path /= std::filesystem::path("config.yaml");
  if (argc >= 2) {
    config_path = argv[1];
  }

  auto app = holoscan::make_application<HolovizGeometryApp>();
  app->config(config_path);
  app->run();

  return 0;
}
