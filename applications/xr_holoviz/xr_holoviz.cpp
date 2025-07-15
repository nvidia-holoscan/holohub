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
#include "xr_composition_layer_manager.hpp"

// Constants for torus generation
constexpr float TORUS_MAJOR_RADIUS = 0.3f;  // Distance from center to tube center
constexpr float TORUS_MINOR_RADIUS = 0.1f;  // Radius of the tube
constexpr int TORUS_MAJOR_SEGMENTS = 32;    // Number of segments around the major radius
constexpr int TORUS_MINOR_SEGMENTS = 16;    // Number of segments around the minor radius

// Generate torus vertices - using triangles (3 vertices per triangle)
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

        return std::array<float, 3>{x, y, z};
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

// Cube
std::array<std::array<float, 3>, 24> cube_array = {
    {{-0.2f, -0.2f, -0.2f}, {-0.2f, 0.2f, -0.2f}, {-0.2f, 0.2f, -0.2f}, {0.2f, 0.2f, -0.2f},
     {0.2f, 0.2f, -0.2f},   {0.2f, -0.2f, -0.2f}, {0.2f, -0.2f, -0.2f}, {-0.2f, -0.2f, -0.2f},
     {-0.2f, -0.2f, 0.2f},  {-0.2f, 0.2f, 0.2f},  {-0.2f, 0.2f, 0.2f},  {0.2f, 0.2f, 0.2f},
     {0.2f, 0.2f, 0.2f},    {0.2f, -0.2f, 0.2f},  {0.2f, -0.2f, 0.2f},  {-0.2f, -0.2f, 0.2f},
     {-0.2f, -0.2f, -0.2f}, {-0.2f, -0.2f, 0.2f}, {-0.2f, 0.2f, -0.2f}, {-0.2f, 0.2f, 0.2f},
     {0.2f, 0.2f, -0.2f},   {0.2f, 0.2f, 0.2f},   {0.2f, -0.2f, -0.2f}, {0.2f, -0.2f, 0.2f}}};

namespace holoscan::ops {
class XrGeometrySourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(XrGeometrySourceOp)

  XrGeometrySourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<xr::FrameState>("xr_frame_state");
    spec.param(xr_composition_layer_manager_, "xr_composition_layer_manager");
    spec.param(allocator_, "allocator");

    spec.output<gxf::Entity>("outputs");
    spec.output<std::vector<HolovizOp::InputSpec>>("output_specs");
    spec.output<gxf::Entity>("render_buffer_output");
    spec.output<gxf::Entity>("depth_buffer_output");
    spec.output<std::shared_ptr<xr::CompositionLayerBaseHeader>>("xr_composition_layer");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
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

    add_data<24, 3>(entity, "cube", cube_array, context);

    /* ======
     Create an XR composition layer by current frame state.
     ======
    */
    auto xr_composition_layer =
        xr_composition_layer_manager_->create_composition_layer(*frame_state);

    /* ======
     Create input specs for HolovizOp with stereo views.
     ======
    */

    // Create model matrix for each objects
    glm::mat4 model_matrix_torus =
        glm::translate(glm::mat4{1}, glm::vec3(0, 0, -1.0f)) *  // Put it to (0, 0, -1)
        glm::rotate(glm::mat4{1},
                    static_cast<float>(frame_state->predictedDisplayTime.get()) /
                        1'000'000'000,  // Rotate the object with display time
                    glm::vec3(0, -1, 0));
    glm::mat4 model_matrix_cube =
        glm::translate(glm::mat4{1}, glm::vec3(0, 0.5f, -1.0f)) *  // Put it to (0, 0.5, -1)
        glm::rotate(glm::mat4{1},
                    static_cast<float>(frame_state->predictedDisplayTime.get()) / 1'000'000'000,
                    glm::vec3(0, -1, 0));

    HolovizOp::InputSpec torus_spec = XrViewsHelper::create_spec_with_views(
        "torus", HolovizOp::InputType::TRIANGLES_3D, xr_composition_layer, model_matrix_torus);
    specs.push_back(torus_spec);

    HolovizOp::InputSpec cube_spec = XrViewsHelper::create_spec_with_views(
        "cube", HolovizOp::InputType::LINES_3D, xr_composition_layer, model_matrix_cube);
    specs.push_back(cube_spec);

    // emit outputs
    op_output.emit(entity, "outputs");
    op_output.emit(specs, "output_specs");
    op_output.emit(std::static_pointer_cast<xr::CompositionLayerBaseHeader>(xr_composition_layer),
                   "xr_composition_layer");
    create_render_buffer(context, op_output);
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
  // Create a render buffer with swapchain
  void create_render_buffer(ExecutionContext& context, OutputContext& op_output) {

    holoscan::Tensor color_tensor = xr_composition_layer_manager_->acquire_color_swapchain_image();
    holoscan::Tensor depth_tensor = xr_composition_layer_manager_->acquire_depth_swapchain_image();

    // Prepare color render buffer to read in HolovizOp
    auto render_gxf_output = nvidia::gxf::Entity::New(context.context());
    auto video_buffer = render_gxf_output.value().add<nvidia::gxf::VideoBuffer>("render_buffer_output");
    nvidia::gxf::VideoBufferInfo video_buffer_info;
    video_buffer_info.width = color_tensor.shape()[1];
    video_buffer_info.height = color_tensor.shape()[0];
    video_buffer_info.color_format = nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA;
    video_buffer_info.surface_layout = nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR;

    nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA> video_format_size;
    video_buffer_info.color_planes = video_format_size.getDefaultColorPlanes(
        video_buffer_info.width, video_buffer_info.height, false);

    video_buffer.value()->wrapMemory(video_buffer_info,
                                     color_tensor.nbytes(),
                                     nvidia::gxf::MemoryStorageType::kDevice,
                                     color_tensor.data(),
                                     [](void*) mutable { return nvidia::gxf::Success; });
    auto render_output = gxf::Entity(std::move(render_gxf_output.value()));
    op_output.emit(render_output, "render_buffer_output");

    // Prepare depth buffer to read in HolovizOp
    auto depth_gxf_output = nvidia::gxf::Entity::New(context.context());
    auto depth_video_buffer = depth_gxf_output.value().add<nvidia::gxf::VideoBuffer>("depth_buffer_output");
    nvidia::gxf::VideoBufferInfo depth_buffer_info;
    depth_buffer_info.width = depth_tensor.shape()[1];
    depth_buffer_info.height = depth_tensor.shape()[0];
    depth_buffer_info.color_format = nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F;
    depth_buffer_info.surface_layout = nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR;

    nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F> video_format_size_depth;
    depth_buffer_info.color_planes = video_format_size_depth.getDefaultColorPlanes(
        depth_buffer_info.width, depth_buffer_info.height, false);

    depth_video_buffer.value()->wrapMemory(depth_buffer_info,
                                     depth_tensor.nbytes(),
                                     nvidia::gxf::MemoryStorageType::kDevice,
                                     depth_tensor.data(),
                                     [](void*) mutable { return nvidia::gxf::Success; });
    auto depth_output = gxf::Entity(std::move(depth_gxf_output.value()));
    op_output.emit(depth_output, "depth_buffer_output");
  }

  Parameter<std::shared_ptr<holoscan::UnboundedAllocator>> allocator_;
  Parameter<std::shared_ptr<holoscan::XrCompositionLayerManager>> xr_composition_layer_manager_;
};

// Submit the composition layer, release the swapchains and synchronize before end frame.
class XrCompositionLayerSubmitOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(XrCompositionLayerSubmitOp)

  XrCompositionLayerSubmitOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<nvidia::gxf::VideoBuffer>("render_buffer_input");
    spec.input<nvidia::gxf::VideoBuffer>("depth_buffer_input");
    spec.input<std::shared_ptr<xr::CompositionLayerBaseHeader>>("xr_composition_layer");
    spec.output<std::shared_ptr<xr::CompositionLayerBaseHeader>>("xr_composition_layer");
    spec.param(xr_composition_layer_manager_, "xr_composition_layer_manager");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // Get the render buffer output from HolovizOp, but not used. This is just for synchronization.
    auto render_buffer = op_input.receive<gxf::Entity>("render_buffer_input").value();
    auto depth_buffer = op_input.receive<gxf::Entity>("depth_buffer_input").value();
    auto composition_layer =
        op_input.receive<std::shared_ptr<xr::CompositionLayerBaseHeader>>("xr_composition_layer")
            .value();

    // Release the swapchains and synchronize
    cudaStream_t cuda_stream = cudaStreamDefault;
    xr_composition_layer_manager_->release_swapchain_images(cuda_stream);

    // Emit the composition layer for the end frame operator
    op_output.emit(composition_layer, "xr_composition_layer");
  }

 private:
  Parameter<std::shared_ptr<holoscan::XrCompositionLayerManager>> xr_composition_layer_manager_;
};

}  // namespace holoscan::ops

class HolovizGeometryApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto allocator = make_resource<holoscan::UnboundedAllocator>("pool");

    auto xr_session = make_resource<holoscan::XrSession>(
        "xr_session", holoscan::Arg("application_name") = std::string("XR Render Cube"));

    auto xr_composition_layer_manager = make_resource<holoscan::XrCompositionLayerManager>(
        "xr_composition_layer_manager", Arg("xr_session") = xr_session);

    auto xr_begin_frame = make_operator<holoscan::ops::XrBeginFrameOp>(
        "xr_begin_frame", holoscan::Arg("xr_session") = xr_session);
    auto xr_end_frame = make_operator<holoscan::ops::XrEndFrameOp>(
        "xr_end_frame", holoscan::Arg("xr_session") = xr_session);

    auto source = make_operator<ops::XrGeometrySourceOp>(
        "source",
        Arg("xr_composition_layer_manager") = xr_composition_layer_manager,
        Arg("allocator") = allocator);

    // TODO: width and height are hardcoded for now, can't get the width and height from headset at
    // this point
    auto holoviz_args = from_config("holoviz");
    auto visualizer = make_operator<ops::HolovizOp>("holoviz",
                                                    Arg("enable_render_buffer_input", true),
                                                    Arg("enable_render_buffer_output", true),
                                                    Arg("enable_depth_buffer_input", true),
                                                    Arg("enable_depth_buffer_output", true),
                                                    Arg("headless", true),
                                                    Arg("allocator") = allocator,
                                                    holoviz_args);

    auto xr_submit = make_operator<ops::XrCompositionLayerSubmitOp>(
        "xr_submit", Arg("xr_composition_layer_manager") = xr_composition_layer_manager);

    // The core OpenXR render loop: begin frame -> end frame.
    add_flow(xr_begin_frame, xr_end_frame, {{"xr_frame_state", "xr_frame_state"}});
    add_flow(xr_begin_frame, source, {{"xr_frame_state", "xr_frame_state"}});

    // source -> holoviz
    add_flow(source, visualizer, {{"outputs", "receivers"}});
    add_flow(source, visualizer, {{"output_specs", "input_specs"}});
    add_flow(source, visualizer, {{"render_buffer_output", "render_buffer_input"}});
    add_flow(source, visualizer, {{"depth_buffer_output", "depth_buffer_input"}});

    add_flow(visualizer, xr_submit, {{"render_buffer_output", "render_buffer_input"}});
    add_flow(visualizer, xr_submit, {{"depth_buffer_output", "depth_buffer_input"}});
    add_flow(source, xr_submit, {{"xr_composition_layer", "xr_composition_layer"}});
    add_flow(xr_submit, xr_end_frame, {{"xr_composition_layer", "xr_composition_layers"}});
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
