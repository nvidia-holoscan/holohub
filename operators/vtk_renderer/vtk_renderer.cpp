/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "vtk_renderer.hpp"
#include "../version_helper_macros.hpp"

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"

#include <cuda.h>

#include <vtkCamera.h>
#include <vtkImageActor.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkTextActor.h>

#define CUDA_TRY(stmt)                                                                   \
  ({                                                                                     \
    cudaError_t _holoscan_cuda_err = stmt;                                               \
    if (cudaSuccess != _holoscan_cuda_err) {                                             \
      GXF_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).", \
                    #stmt,                                                               \
                    __LINE__,                                                            \
                    __FILE__,                                                            \
                    cudaGetErrorString(_holoscan_cuda_err),                              \
                    _holoscan_cuda_err);                                                 \
    }                                                                                    \
    _holoscan_cuda_err;                                                                  \
  })

/**
 * @brief Render text at a specific location on the renderer.
 *
 * @param text The text to render.
 * @param x The x coordinate of the text.
 * @param y The y coordinate of the text.
 * @param renderer The renderer to render the text on.
 */
static void render_text_at_location(std::string text, float x, float y, vtkRenderer* renderer) {
  vtkNew<vtkTextActor> textActor;
  textActor->SetInput(text.c_str());
  textActor->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
  textActor->GetPositionCoordinate()->SetValue(x, y);
  renderer->AddActor(textActor);
}

/**
 * @brief Reposition the camera to fit the image in the renderer.
 *
 * @param renderer The renderer to reposition the camera in.
 * @param imageData The image data to fit in the renderer.
 */
static void reposition_camera(vtkRenderer* renderer, vtkImageData* imageData) {
  // Set up the background camera to fill the renderer with the image.
  double origin[3];
  double spacing[3];
  int extent[6];
  imageData->GetOrigin(origin);
  imageData->GetSpacing(spacing);
  imageData->GetExtent(extent);

  vtkCamera* camera = renderer->GetActiveCamera();
  camera->ParallelProjectionOn();

  // Center the image in the renderer.
  double xc = origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0];
  double yc = origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1];
  double yd = (extent[3] - extent[2] + 1) * spacing[1];
  double d = camera->GetDistance();
  camera->SetParallelScale(0.5 * yd);
  camera->SetFocalPoint(xc, yc, 0.0);
  camera->SetPosition(xc, yc, d);
}

namespace holoscan::ops {

// VTK members
struct VtkRendererOp::Internals {
  vtkNew<vtkImageActor> image_actor;
  vtkNew<vtkRenderer> background_renderer;
  vtkNew<vtkRenderer> foreground_renderer;
  vtkNew<vtkRenderWindow> renderer_window;
};

void VtkRendererOp::setup(OperatorSpec& spec) {
  this->internals = std::make_shared<VtkRendererOp::Internals>();

  spec.input<gxf::Entity>("annotations");
  spec.input<gxf::Entity>("videostream");
  spec.param(
      this->window_name, "window_name", "Window Name", "Window Name", std::string("VTK (Kitware)"));
  spec.param(this->width, "width", "Window width", "Window width in px", 640u);
  spec.param(this->height, "height", "Window height", "Window height in px", 480u);
  spec.param(this->labels, "labels", "tools labels", "tools labels");
}

void VtkRendererOp::start() {
  this->internals->renderer_window->SetWindowName(this->window_name.get().c_str());
  // Two layers, one for the background, one for the foreground
  this->internals->renderer_window->SetNumberOfLayers(2);
  this->internals->renderer_window->SetSize(this->width.get(), this->height.get());

  // Foreground at the top-most layer.
  this->internals->foreground_renderer->SetLayer(1);
  this->internals->renderer_window->AddRenderer(this->internals->foreground_renderer);

  this->internals->background_renderer->SetLayer(0);
  this->internals->background_renderer->InteractiveOff();
  this->internals->renderer_window->AddRenderer(this->internals->background_renderer);
  this->internals->background_renderer->AddActor(this->internals->image_actor);
}

void VtkRendererOp::compute(InputContext& op_input, OutputContext&, ExecutionContext& context) {
  auto annotations = op_input.receive<gxf::Entity>("annotations").value();

  // Render the annotations on the foreground layer.
  auto scaled_coords_tensor = annotations.get<Tensor>("scaled_coords");
  if (scaled_coords_tensor) {
    std::vector<float> scaled_coords(scaled_coords_tensor->size());

    // Copy the data from the tensor to the host.
    CUDA_TRY(cudaMemcpy(scaled_coords.data(),
                        scaled_coords_tensor->data(),
                        scaled_coords_tensor->nbytes(),
                        cudaMemcpyDeviceToHost));

    this->internals->foreground_renderer->RemoveAllViewProps();

    // scale_coords comes in the format [X0 Y0 S0 X1 Y1 S1 ... Xn Yn Sn]
    // each numbered tuple represent a label (scissors, clipper...)
    for (int i = 0; i < scaled_coords.size(); i += 3) {
      if (scaled_coords[i] > 0) {
        std::string label = labels.get()[i / 3];
        float x = scaled_coords[i];
        float y = scaled_coords[i + 1];
        render_text_at_location(label, x, y, this->internals->foreground_renderer);
      }
    }
  }

  // Render the videostream on the background layer.
  auto videostream = op_input.receive<gxf::Entity>("videostream").value();
  const auto videostream_tensor = videostream.get<Tensor>("");
  if (videostream_tensor) {
  #if GXF_HAS_DLPACK_SUPPORT
    nvidia::gxf::Tensor in_tensor_gxf{videostream_tensor->dl_ctx()};
  #else
    holoscan::gxf::GXFTensor in_tensor_gxf{videostream_tensor->dl_ctx()};
  #endif
    auto& shape = in_tensor_gxf.shape();
    const int y = shape.dimension(0);
    const int x = shape.dimension(1);
    const int z = shape.dimension(2);
    const unsigned int expected_length = x * y * z;  // Each pixel is an uchar (1B)
    const unsigned int received_length = videostream_tensor->nbytes();

    // Precondition checks
    if (expected_length != received_length) {
      throw std::runtime_error(
          fmt::format("videostream input : Invalid input size : expected {}B, received {}B",
                      expected_length,
                      received_length));
    }

    if (x != this->width || y != this->height || z != 3) {
      throw std::runtime_error(
          fmt::format("videostream input : Invalid image dims : expects {}x{}x{} dims",
                      this->width,
                      this->height,
                      3));
    }

    vtkNew<vtkImageData> imageData;
    imageData->SetDimensions(x, y, 1);
    imageData->AllocateScalars(VTK_UNSIGNED_CHAR, z);

    // Copy the data from the tensor to the image data.
    CUDA_TRY(cudaMemcpy(static_cast<char*>(imageData->GetScalarPointer()),
                        videostream_tensor->data(),
                        videostream_tensor->nbytes(),
                        cudaMemcpyDeviceToHost));

    this->internals->image_actor->SetInputData(imageData);

    // Render first to find the extent of the image.
    this->internals->renderer_window->Render();

    // Adjust camera so that it can render see the whole extent of the image.
    reposition_camera(this->internals->background_renderer, imageData);

    // Final render with the camera at the proper position.
    this->internals->renderer_window->Render();
  }
}

}  // namespace holoscan::ops
