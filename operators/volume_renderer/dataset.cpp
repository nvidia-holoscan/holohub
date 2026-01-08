/* SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dataset.hpp"

#include <filesystem>
#include <fstream>

#include <zlib.h>

#include <holoscan/logger/logger.hpp>

// need to redefine RuntimeError undefined in the header
#define RuntimeError() \
  clara::viz::Exception<std::runtime_error>(__FILE__, __LINE__, "") << "Runtime error "
#include <claraviz/util/CudaMemoryBlob.h>

/// This class holds the data and information for 3D volume
class DataArray {
 public:
  /// Handle
  using Handle = std::shared_ptr<DataArray>;

  /// element data type
  clara::viz::DataElementType type_;
  /// element count of the volume in each direction
  clara::viz::Vector3ui dims_;
  /// spacing between elements in millimeter
  clara::viz::Vector3f spacing_{1.f, 1.f, 1.f};
  /// axis permutation
  std::vector<uint32_t> permute_axis_{0, 1, 2};
  /// axis flip
  std::vector<bool> flip_axes_{false, false, false};
  /// value range of the elements, used for animated volumes
  std::vector<clara::viz::Vector2f> element_range_;

  /// Memory blob
  std::shared_ptr<clara::viz::CudaMemoryBlob> blob_;
};

void Dataset::SetVolume(Types type, const std::array<float, 3>& spacing,
                        const std::array<uint32_t, 3>& permute_axis,
                        const std::array<bool, 3>& flip_axes,
                        const std::vector<clara::viz::Vector2f>& element_range,
                        const nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor,
                        cudaStream_t cuda_stream) {
  DataArray data_array;

  switch (tensor->element_type()) {
    case nvidia::gxf::PrimitiveType::kInt8:
      data_array.type_ = clara::viz::DataElementType::INT8;
      break;
    case nvidia::gxf::PrimitiveType::kUnsigned8:
      data_array.type_ = clara::viz::DataElementType::UINT8;
      break;
    case nvidia::gxf::PrimitiveType::kInt16:
      data_array.type_ = clara::viz::DataElementType::INT16;
      break;
    case nvidia::gxf::PrimitiveType::kUnsigned16:
      data_array.type_ = clara::viz::DataElementType::UINT16;
      break;
    case nvidia::gxf::PrimitiveType::kInt32:
      data_array.type_ = clara::viz::DataElementType::INT32;
      break;
    case nvidia::gxf::PrimitiveType::kUnsigned32:
      data_array.type_ = clara::viz::DataElementType::UINT32;
      break;
    case nvidia::gxf::PrimitiveType::kFloat32:
      data_array.type_ = clara::viz::DataElementType::FLOAT;
      break;
    default:
      holoscan::log_error("Unhandled element type'{}'.", int(tensor->element_type()));
      return;
  }

  nvidia::gxf::Shape shape = tensor->shape();
  data_array.dims_ = clara::viz::Vector3ui(shape.dimension(shape.rank() - 1),
                                           shape.dimension(shape.rank() - 2),
                                           shape.dimension(shape.rank() - 3));
  data_array.spacing_ = clara::viz::Vector3f(spacing[0], spacing[1], spacing[2]);
  data_array.permute_axis_ = {permute_axis[0], permute_axis[1], permute_axis[2]};
  data_array.flip_axes_ = {flip_axes[0], flip_axes[1], flip_axes[2]};
  data_array.element_range_ = element_range;

  int32_t frames = 1;
  if (shape.rank() == 4) { frames = shape.dimension(0); }

  // copy the data
  const size_t volume_size =
      tensor->bytes_per_element() * data_array.dims_(0) * data_array.dims_(1) * data_array.dims_(2);
  uintptr_t volume_data = reinterpret_cast<uintptr_t>(tensor->pointer());
  for (uint32_t frame = 0; frame < frames; ++frame) {
    DataArray::Handle cur_data_array(new DataArray);
    *cur_data_array = data_array;

    cur_data_array->blob_ = std::make_shared<clara::viz::CudaMemoryBlob>(
        std::make_unique<clara::viz::CudaMemory>(volume_size));

    {
      std::unique_ptr<clara::viz::IBlob::AccessGuard> access_gpu =
          cur_data_array->blob_->Access(cuda_stream);

      cudaMemcpy3DParms copy_params = {0};

      copy_params.srcPtr.ptr = reinterpret_cast<void*>(volume_data);
      copy_params.srcPtr.pitch = tensor->stride(shape.rank() - 2);
      copy_params.srcPtr.xsize = data_array.dims_(0) * tensor->bytes_per_element();
      copy_params.srcPtr.ysize = data_array.dims_(1);

      copy_params.dstPtr.ptr = access_gpu->GetData();
      copy_params.dstPtr.pitch = data_array.dims_(0) * tensor->bytes_per_element();
      copy_params.dstPtr.xsize = copy_params.srcPtr.xsize;
      copy_params.dstPtr.ysize = copy_params.srcPtr.ysize;

      copy_params.extent.width = data_array.dims_(0) * tensor->bytes_per_element();
      copy_params.extent.height = data_array.dims_(1);
      copy_params.extent.depth = data_array.dims_(2);

      switch (tensor->storage_type()) {
        case nvidia::gxf::MemoryStorageType::kDevice:
          copy_params.kind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
          break;
        case nvidia::gxf::MemoryStorageType::kHost:
        case nvidia::gxf::MemoryStorageType::kSystem:
          copy_params.kind = cudaMemcpyKind::cudaMemcpyHostToDevice;
          break;
        default:
          holoscan::log_error("NIFTI unhandled storage type {}", int(tensor->storage_type()));
          return;
      }

      if (cudaMemcpy3DAsync(&copy_params, cuda_stream) != cudaSuccess) {
        holoscan::log_error("Failed to copy to GPU memory");
        return;
      }
    }

    switch (type) {
      case Types::Density:
        density_.push_back(cur_data_array);
        break;
      case Types::Segmentation:
        segmentation_.push_back(cur_data_array);
        break;
      default:
        throw std::runtime_error("Unhandled type");
    }

    volume_data += tensor->stride(0);
  }
}

void Dataset::ResetVolume(Types type) {
  switch (type) {
    case Types::Density:
      density_.clear();
      break;
    case Types::Segmentation:
      segmentation_.clear();
      break;
    default:
      throw std::runtime_error("Unhandled type");
  }
}

void Dataset::Configure(clara::viz::DataConfigInterface& data_config_interface) {
  clara::viz::DataConfigInterface::AccessGuard access(data_config_interface);

  // enable the streaming if there if more than one frame
  access->streaming = (density_.size() > 1);

  if (!density_.empty()) {
    access->arrays.emplace_back();
    const auto array = --(access->arrays.end());
    array->id = "density";
    array->dimension_order.Set("DXYZ");
    array->element_type.Set(density_[0]->type_);
    array->permute_axis.Set({0,
                             density_[0]->permute_axis_[0] + 1,
                             density_[0]->permute_axis_[1] + 1,
                             density_[0]->permute_axis_[2] + 1});
    array->flip_axes.Set({false,
                          density_[0]->flip_axes_[0],
                          density_[0]->flip_axes_[1],
                          density_[0]->flip_axes_[2]});

    array->levels.emplace_back();
    const auto level = array->levels.begin();
    level->size.Set({1, density_[0]->dims_(0), density_[0]->dims_(1), density_[0]->dims_(2)});
    level->element_size.Set(
        {1.0f, density_[0]->spacing_(0), density_[0]->spacing_(1), density_[0]->spacing_(2)});
    level->element_range.Set(density_[0]->element_range_);
  }
  if (!segmentation_.empty()) {
    access->arrays.emplace_back();
    const auto array = --(access->arrays.end());
    array->id = "segmentation";
    array->dimension_order.Set("MXYZ");
    array->element_type.Set(segmentation_[0]->type_);
    array->permute_axis.Set({0,
                             segmentation_[0]->permute_axis_[0] + 1,
                             segmentation_[0]->permute_axis_[1] + 1,
                             segmentation_[0]->permute_axis_[2] + 1});
    array->flip_axes.Set({false,
                          segmentation_[0]->flip_axes_[0],
                          segmentation_[0]->flip_axes_[1],
                          segmentation_[0]->flip_axes_[2]});

    array->levels.emplace_back();
    const auto level = array->levels.begin();
    level->size.Set(
        {1, segmentation_[0]->dims_(0), segmentation_[0]->dims_(1), segmentation_[0]->dims_(2)});
    level->element_size.Set({1.0f,
                             segmentation_[0]->spacing_(0),
                             segmentation_[0]->spacing_(1),
                             segmentation_[0]->spacing_(2)});
    level->element_range.Set(segmentation_[0]->element_range_);
  }
}

void Dataset::Set(clara::viz::DataInterface& data_interface, uint32_t frame_index) {
  {
    clara::viz::DataInterface::AccessGuard access(data_interface);
    if (!density_.empty()) {
      if (frame_index > density_.size()) {
        throw std::runtime_error("Invalid density frame index");
      }
      const std::shared_ptr<DataArray> data_array = density_[frame_index];
      access->array_id.Set("density");
      access->level.Set(0);
      access->offset.Set({0, 0, 0, 0});
      access->size.Set({1, data_array->dims_(0), data_array->dims_(1), data_array->dims_(2)});
      access->blob = data_array->blob_;
    }
  }
  {
    clara::viz::DataInterface::AccessGuard access(data_interface);
    if (!segmentation_.empty()) {
      if (frame_index > segmentation_.size()) {
        throw std::runtime_error("Invalid segmentation frame index");
      }
      const std::shared_ptr<DataArray> data_array = segmentation_[frame_index];
      access->array_id.Set("segmentation");
      access->level.Set(0);
      access->offset.Set({0, 0, 0, 0});
      access->size.Set({1, data_array->dims_(0), data_array->dims_(1), data_array->dims_(2)});
      access->blob = data_array->blob_;
    }
  }
}

uint32_t Dataset::GetNumberFrames() const {
  return density_.size();
}

void Dataset::SetFrameDuration(const std::chrono::duration<float>& frame_duration) {
  frame_duration_ = frame_duration;
}

const std::chrono::duration<float>& Dataset::GetFrameDuration() const {
  return frame_duration_;
}
