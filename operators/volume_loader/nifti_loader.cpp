/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nifti_loader.hpp"

#include <filesystem>

#include <nifti2_io.h>

#include "volume.hpp"

namespace holoscan::ops {

bool is_nifty(const std::string& file_name) {
  std::filesystem::path path(file_name);

  if ((path.extension() == ".nii") ||
      ((path.extension() == ".gz") && (path.replace_extension().extension() == ".nii"))) {
    return true;
  }

  return false;
}

bool load_nifty(const std::string& file_name, Volume& volume) {
  std::filesystem::path path(file_name);

  std::unique_ptr<nifti_image> image;
  image.reset(nifti_image_read(file_name.c_str(), true));
  if (!image) { return false; }

  if ((image->ndim != 3) && (image->ndim != 4)) {
    holoscan::log_error("NIFTI unhandled number of dimensions {}, expected 3 or 4", image->ndim);
    return false;
  }

  nvidia::gxf::PrimitiveType primitive_type;
  uint64_t primitive_size;
  switch (image->datatype) {
    case NIFTI_TYPE_INT8:
      primitive_type = nvidia::gxf::PrimitiveType::kInt8;
      break;
    case NIFTI_TYPE_UINT8:
      primitive_type = nvidia::gxf::PrimitiveType::kUnsigned8;
      break;
    case NIFTI_TYPE_INT16:
      primitive_type = nvidia::gxf::PrimitiveType::kInt16;
      break;
    case NIFTI_TYPE_UINT16:
      primitive_type = nvidia::gxf::PrimitiveType::kUnsigned16;
      break;
    case NIFTI_TYPE_INT32:
      primitive_type = nvidia::gxf::PrimitiveType::kInt32;
      break;
    case NIFTI_TYPE_UINT32:
      primitive_type = nvidia::gxf::PrimitiveType::kUnsigned32;
      break;
    case NIFTI_TYPE_FLOAT32:
      primitive_type = nvidia::gxf::PrimitiveType::kFloat32;
      break;
    default:
      holoscan::log_error("NIFTI unhandled datatype {}", image->datatype);
      return false;
  }

  int i = 0, j = 0, k = 0;
  if (image->qform_code > 0) {
    nifti_dmat44_to_orientation(image->qto_xyz, &i, &j, &k);
  } else if (image->sform_code > 0) {
    nifti_dmat44_to_orientation(image->sto_xyz, &i, &j, &k);
  }

  if ((i > 0) && (j > 0) && (k > 0)) {
    auto toString = [](int i) -> char {
      switch (i) {
        case NIFTI_L2R:
          return 'L';
        case NIFTI_R2L:
          return 'R';
        case NIFTI_P2A:
          return 'P';
        case NIFTI_A2P:
          return 'A';
        case NIFTI_I2S:
          return 'I';
        case NIFTI_S2I:
          return 'S';
      }
      throw std::runtime_error("NIFTI unhandled orientation");
    };
    std::string orientation_string;
    orientation_string += toString(i);
    orientation_string += toString(j);
    orientation_string += toString(k);
    volume.SetOrientation(orientation_string);
  }

  // convert units to internally used meter
  float to_millimeter;
  switch (image->xyz_units) {
    case NIFTI_UNITS_METER:
      to_millimeter = 1000.f;
      break;
    default:
    case NIFTI_UNITS_MM:
      to_millimeter = 1.f;
      break;
    case NIFTI_UNITS_MICRON:
      to_millimeter = 0.001f;
      break;
  }
  volume.spacing_ = {float(image->dx) * to_millimeter,
                     float(image->dy) * to_millimeter,
                     float(image->dz) * to_millimeter};

  switch (image->time_units) {
    case NIFTI_UNITS_USEC:
      volume.frame_duration_ = std::chrono::duration<float, std::chrono::microseconds::period>(
          static_cast<float>(image->dt));
      break;
    case NIFTI_UNITS_MSEC:
      volume.frame_duration_ = std::chrono::duration<float, std::chrono::milliseconds::period>(
          static_cast<float>(image->dt));
      break;
    default:
    case NIFTI_UNITS_SEC:
      volume.frame_duration_ =
          std::chrono::duration<float, std::chrono::seconds::period>(static_cast<float>(image->dt));
      break;
  }

  // allocate the tensor
  std::vector<int32_t> dims;
  if (image->nt > 1) { dims.push_back(image->nt); }
  dims.push_back(image->nz);
  dims.push_back(image->ny);
  dims.push_back(image->nx);

  if (!volume.tensor_->reshapeCustom(nvidia::gxf::Shape(dims),
                                     primitive_type,
                                     nvidia::gxf::PrimitiveTypeSize(primitive_type),
                                     nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                     volume.storage_type_,
                                     volume.allocator_)) {
    holoscan::log_error("NIFTI failed to reshape tensor");
    return false;
  }

  // copy the data
  switch (volume.storage_type_) {
    case nvidia::gxf::MemoryStorageType::kDevice:
      if (cudaMemcpy(volume.tensor_->pointer(),
                     reinterpret_cast<const void*>(image->data),
                     image->nt * image->nz * image->ny * image->nx * image->nbyper,
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        holoscan::log_error("NIFTI failed to copy to GPU memory");
        return false;
      }
      break;
    case nvidia::gxf::MemoryStorageType::kHost:
    case nvidia::gxf::MemoryStorageType::kSystem:
      memcpy(volume.tensor_->pointer(),
             image->data,
             image->nt * image->nz * image->ny * image->nx * image->nbyper);
      break;
    default:
      holoscan::log_error("NIFTI unhandled storage type {}", int(volume.storage_type_));
      return false;
  }

  return true;
}

}  // namespace holoscan::ops
