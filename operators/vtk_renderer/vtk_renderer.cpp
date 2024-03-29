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

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

#include "vtkCamera.h"
#include "vtkImageData.h"
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

void VtkRendererOp::setup(OperatorSpec& spec) {
  auto& in_annotations = spec.input<gxf::Entity>("annotations");
  auto& in_videostream = spec.input<gxf::Entity>("videostream");

  spec.param(this->in_annotations,
             "in_annotations",
             "Input Annotations",
             "Input Annotations",
             &in_annotations);
  spec.param(this->in_videostream,
             "in_videostream",
             "Input Videostream",
             "Input Annotations",
             &in_videostream);

  spec.param(width, "width", "Window width", "Window width in px", 640u);
  spec.param(height, "height", "Window height", "Window height in px", 480u);
  spec.param(labels, "labels", "tools labels", "tools labels");
}

void VtkRendererOp::start() {
  this->rendererWindow->SetWindowName("VTK (Kitware)");
  // Two layers, one for the background, one for the foreground
  this->rendererWindow->SetNumberOfLayers(2);
  this->rendererWindow->SetSize(width.get(), height.get());

  // Foreground at the top-most layer.
  this->foregroundRenderer->SetLayer(1);
  this->rendererWindow->AddRenderer(foregroundRenderer);

  this->backgroundRenderer->SetLayer(0);
  this->backgroundRenderer->InteractiveOff();
  this->rendererWindow->AddRenderer(backgroundRenderer);
  this->backgroundRenderer->AddActor(imageActor);
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

    this->foregroundRenderer->RemoveAllViewProps();

    // scale_coords comes in the format [X0 Y0 X1 Y1 ... Xn Yn]
    // each numbered tuple represent a label (scissors, clipper...)
    for (int i = 0; i < scaled_coords.size(); i += 2) {
      if (scaled_coords[i] > 0) {
        std::string label = labels.get()[i / 2];
        float x = scaled_coords[i];
        float y = scaled_coords[i + 1];
        render_text_at_location(label, x, y, this->foregroundRenderer);
      }
    }
  }

  // Render the videostream on the background layer.
  auto videostream = op_input.receive<gxf::Entity>("videostream").value();
  const auto videostream_tensor = videostream.get<Tensor>("");
  if (videostream_tensor) {
    holoscan::gxf::GXFTensor in_tensor_gxf{videostream_tensor->dl_ctx()};
    auto& shape = in_tensor_gxf.shape();
    const int x = shape.dimension(0);
    const int y = shape.dimension(1);
    const int z = shape.dimension(2);

    vtkNew<vtkImageData> imageData;
    imageData->SetDimensions(y, x, 1);
    imageData->AllocateScalars(VTK_UNSIGNED_CHAR, z);

    // Copy the data from the tensor to the image data.
    CUDA_TRY(cudaMemcpy(static_cast<char*>(imageData->GetScalarPointer()),
                        videostream_tensor->data(),
                        videostream_tensor->nbytes(),
                        cudaMemcpyDeviceToHost));

    imageActor->SetInputData(imageData);

    // Render first to find the extent of the image.
    this->rendererWindow->Render();

    // Adjust camera so that it can render see the whole extent of the image.
    reposition_camera(this->backgroundRenderer, imageData);

    // Final render with the camera at the proper position.
    this->rendererWindow->Render();
  }
}

}  // namespace holoscan::ops
