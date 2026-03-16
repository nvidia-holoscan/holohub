/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pva_radar_graphics.hpp"

#include <ErrorCheckMacros.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Tensor.h>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace holoscan::ops {

PVARadarGraphicsOp::PVARadarGraphicsOp() = default;
PVARadarGraphicsOp::~PVARadarGraphicsOp() = default;

void PVARadarGraphicsOp::setup(OperatorSpec& spec) {
  spec.param(
      allocator_, "allocator", "Allocator", "Allocator used to allocate the output image tensor.");
  spec.param(
      doa_scale_,
      "doa_scale",
      "DOA scale",
      "Scale factor mapping DOA real-space coordinates to the unit cube for Holoviz (default "
      "85.0, derived empirically from PVA sample data).",
      {85.0f});
  spec.input<NVCVTensorHandle>("nci");
  spec.input<NVCVTensorHandle>("peak_count");
  spec.input<NVCVTensorHandle>("doa");
  spec.output<nvidia::gxf::Entity>("output_image");
  spec.output<holoscan::TensorMap>("output_xyz");
}

std::shared_ptr<Tensor> PVARadarGraphicsOp::allocTensorSpace(nvidia::gxf::Shape shape,
                                                             nvidia::gxf::PrimitiveType type) {
  void* buffer = nullptr;
  int64_t element_size =
      type == nvidia::gxf::PrimitiveType::kFloat32 ? sizeof(float) : sizeof(uint8_t);
  // Use pinned host memory so we can write from host in compute() (DOA data) and so
  // Holoviz can access it; wrapMemory reports kDevice for compatibility.
  if (cudaMallocHost(&buffer, shape.size() * element_size) != cudaSuccess || !buffer) {
    throw std::runtime_error("cudaMallocHost failed in allocTensorSpace");
  }
  memset(buffer, 0, shape.size() * element_size);

  auto gxf_tensor = std::make_shared<nvidia::gxf::Tensor>();
  gxf_tensor->wrapMemory(shape,
                         type,
                         element_size,
                         nvidia::gxf::ComputeTrivialStrides(shape, element_size),
                         nvidia::gxf::MemoryStorageType::kDevice,
                         buffer,
                         [orig_pointer = buffer](void*) mutable {
                           cudaFreeHost(orig_pointer);
                           return nvidia::gxf::Success;
                         });
  auto maybe_dl_ctx = gxf_tensor->toDLManagedTensorContext();
  if (!maybe_dl_ctx) {
    throw std::runtime_error("failed to get DLManagedTensorContext from nvidia::gxf::Tensor");
  }
  return std::make_shared<holoscan::Tensor>(maybe_dl_ctx.value());
}

void PVARadarGraphicsOp::initialize() {
  Operator::initialize();

  // Origin lines to help with visualization of the point cloud.
  origin_x_tensor_ =
      allocTensorSpace(nvidia::gxf::Shape{2, 3}, nvidia::gxf::PrimitiveType::kFloat32);
  float* ox = reinterpret_cast<float*>(origin_x_tensor_->data());
  ox[0] = 0.0f;
  ox[1] = 0.0f;
  ox[2] = 0.0f;
  ox[3] = 0.15f;
  ox[4] = 0.0f;
  ox[5] = 0.0f;

  origin_y_tensor_ =
      allocTensorSpace(nvidia::gxf::Shape{2, 3}, nvidia::gxf::PrimitiveType::kFloat32);
  float* oy = reinterpret_cast<float*>(origin_y_tensor_->data());
  oy[0] = 0.0f;
  oy[1] = 0.0f;
  oy[2] = 0.0f;
  oy[3] = 0.0f;
  oy[4] = 0.15f;
  oy[5] = 0.0f;

  origin_z_tensor_ =
      allocTensorSpace(nvidia::gxf::Shape{2, 3}, nvidia::gxf::PrimitiveType::kFloat32);
  float* oz = reinterpret_cast<float*>(origin_z_tensor_->data());
  oz[0] = 0.0f;
  oz[1] = 0.0f;
  oz[2] = 0.0f;
  oz[3] = 0.0f;
  oz[4] = 0.0f;
  oz[5] = 0.15f;
}

void PVARadarGraphicsOp::compute(InputContext& op_input, OutputContext& op_output,
                                 ExecutionContext& context) {
  // receive the required inputs from upstream operator(s)
  NVCVTensorHandle nci_handle = op_input.receive<NVCVTensorHandle>("nci").value();
  NVCVTensorHandle peak_count_handle = op_input.receive<NVCVTensorHandle>("peak_count").value();
  NVCVTensorHandle doa_handle = op_input.receive<NVCVTensorHandle>("doa").value();

  // Utilize the input CUDA stream for async copy of the image buffer
  cudaStream_t stream = op_input.receive_cuda_stream("nci");

  // --- convert NCI -> RGBA image ---
  NVCVTensorData nciData;
  NVCV_CHECK_ERROR(nvcvTensorExportData(nci_handle, &nciData));
  uint8_t* nciHostPtr = nullptr;
  CUPVA_CHECK_ERROR(CupvaMemGetHostPointer((void**)&nciHostPtr, nciData.buffer.strided.basePtr));

  // Construct gxf Tensor for output image using the allocator pool provided at setup time
  auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      fragment()->executor().context(), allocator_->gxf_cid());
  if (!allocator) {
    HOLOSCAN_LOG_ERROR("Failed to get GXF allocator");
    return;
  }
  nvidia::gxf::Shape imageShape{
      static_cast<int32_t>(nciData.shape[0]), static_cast<int32_t>(nciData.shape[1]), 4};
  std::array<size_t, 8> imageStrides{
      nciData.buffer.strided.strides[0], nciData.buffer.strided.strides[1], 1, 0, 0, 0, 0, 0};
  auto image_tensor_map = nvidia::gxf::CreateTensorMap(context.context(),
                                                       allocator.value(),
                                                       {{"image",
                                                         nvidia::gxf::MemoryStorageType::kHost,
                                                         imageShape,
                                                         nvidia::gxf::PrimitiveType::kUnsigned8,
                                                         0,
                                                         imageStrides}},
                                                       false);
  if (!image_tensor_map) {
    HOLOSCAN_LOG_ERROR("Failed to create image tensor map");
    return;
  }
  const auto image_tensor = image_tensor_map.value().get<nvidia::gxf::Tensor>();
  if (!image_tensor) {
    HOLOSCAN_LOG_ERROR("Failed to get image tensor from map");
    return;
  }
  uint8_t* image_tensor_data = reinterpret_cast<uint8_t*>(image_tensor.value()->pointer());
  if (!image_tensor_data) {
    HOLOSCAN_LOG_ERROR("Image tensor pointer is null");
    return;
  }

  // Very simple colormap routine: just interpret every int32_t element as an RGBA pixel, and force
  // A channel to 255 (fully opaque).
  for (int64_t i = 0; i < nciData.shape[0]; i++) {
    for (int64_t j = 0; j < nciData.shape[1]; j++) {
      int64_t idx = i * nciData.buffer.strided.strides[0] + j * nciData.buffer.strided.strides[1];
      image_tensor_data[idx] = nciHostPtr[idx];
      image_tensor_data[idx + 1] = nciHostPtr[idx + 1];
      image_tensor_data[idx + 2] = nciHostPtr[idx + 2];
      image_tensor_data[idx + 3] = 255;
    }
  }

  // --- convert Peak count + DOA -> xyz cloud ---
  NVCVTensorData targetCountData;
  NVCV_CHECK_ERROR(nvcvTensorExportData(peak_count_handle, &targetCountData));
  int32_t* targetCountHostPtr = nullptr;
  CUPVA_CHECK_ERROR(
      CupvaMemGetHostPointer((void**)&targetCountHostPtr, targetCountData.buffer.strided.basePtr));
  int32_t targetCount = targetCountHostPtr[0];

  nvidia::gxf::Shape xyzShape{static_cast<int32_t>(targetCount), 3};
  auto xyzStrides = nvidia::gxf::ComputeTrivialStrides(xyzShape, sizeof(float));
  auto xyz_tensor_map = nvidia::gxf::CreateTensorMap(context.context(),
                                                     allocator.value(),
                                                     {{"xyz",
                                                       nvidia::gxf::MemoryStorageType::kHost,
                                                       xyzShape,
                                                       nvidia::gxf::PrimitiveType::kFloat32,
                                                       0,
                                                       xyzStrides}},
                                                     false);
  if (!xyz_tensor_map) {
    HOLOSCAN_LOG_ERROR("Failed to create xyz tensor map");
    return;
  }
  const auto xyz_gxf_tensor = xyz_tensor_map.value().get<nvidia::gxf::Tensor>();
  if (!xyz_gxf_tensor) {
    HOLOSCAN_LOG_ERROR("Failed to get xyz tensor from map");
    return;
  }
  float* xyz_tensor_data = reinterpret_cast<float*>(xyz_gxf_tensor.value()->pointer());
  if (!xyz_tensor_data) {
    HOLOSCAN_LOG_ERROR("XYZ tensor data pointer is null");
    return;
  }

  // Convert DOA data to xyz coordinates
  NVCVTensorData targetListData;
  NVCV_CHECK_ERROR(nvcvTensorExportData(doa_handle, &targetListData));
  float* targetListHostPtr = nullptr;
  // The DOA tensor layout from PVA is a struct of 7 arrays, documented in the DOA header.
  // We access just the X,Y, and Z arrays here (the 5th, 6th, and 7th arrays).
  constexpr int32_t kXRowIndex = 4;
  constexpr int32_t kYRowIndex = 5;
  constexpr int32_t kZRowIndex = 6;
  CUPVA_CHECK_ERROR(
      CupvaMemGetHostPointer((void**)&targetListHostPtr, targetListData.buffer.strided.basePtr));
  float* targetListHostPtrX =
      reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(targetListHostPtr) +
                               kXRowIndex * targetListData.buffer.strided.strides[0]);
  float* targetListHostPtrY =
      reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(targetListHostPtr) +
                               kYRowIndex * targetListData.buffer.strided.strides[0]);
  float* targetListHostPtrZ =
      reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(targetListHostPtr) +
                               kZRowIndex * targetListData.buffer.strided.strides[0]);

  // The scale factor maps real space to the unit cube for holoviz consumption.
  // It can be set via the doa_scale parameter (e.g. from YAML); default 85.0 from PVA sample data.
  float scale = std::max(doa_scale_.get(), 1.0f);
  for (int32_t i = 0; i < targetCount; i++) {
    xyz_tensor_data[i * 3] = targetListHostPtrX[i] / scale;
    xyz_tensor_data[i * 3 + 1] = targetListHostPtrY[i] / scale;
    /// NOTE: inversion of the Z axis is to transform between right-handed and left-handed
    /// coordinate system for display purposes.
    xyz_tensor_data[i * 3 + 2] = -1.0f * targetListHostPtrZ[i] / scale;
  }

  // Post the output messages to downstream operators.
  auto image_message = nvidia::gxf::Entity{std::move(image_tensor_map.value())};
  op_output.emit(image_message, "output_image");

  // Convert to Holoscan tensor
  auto maybe_dl_ctx = xyz_gxf_tensor.value()->toDLManagedTensorContext();
  if (!maybe_dl_ctx) {
    HOLOSCAN_LOG_ERROR("Failed to convert GXF tensor to Holoscan tensor");
    return;
  }
  auto xyz_tensor = std::make_shared<holoscan::Tensor>(maybe_dl_ctx.value());

  holoscan::TensorMap xyz_message;
  xyz_message.insert({"origin_x", origin_x_tensor_});
  xyz_message.insert({"origin_y", origin_y_tensor_});
  xyz_message.insert({"origin_z", origin_z_tensor_});
  xyz_message.insert({"xyz", xyz_tensor});
  op_output.emit(xyz_message, "output_xyz");
}

}  // namespace holoscan::ops
