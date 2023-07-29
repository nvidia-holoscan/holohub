/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "volume_loader.hpp"

#include "mhd_loader.hpp"
#include "nifti_loader.hpp"
#include "nrrd_loader.hpp"
#include "volume.hpp"

namespace holoscan::ops {

void VolumeLoaderOp::initialize() {
  // call base class
  Operator::initialize();
}

void VolumeLoaderOp::setup(OperatorSpec& spec) {
  spec.param(file_name_, "file_name", "FileName", "Volume data file name", {});
  spec.param(allocator_, "allocator", "Allocator", "Allocator used to allocate the volume data");

  spec.output<holoscan::gxf::Entity>("volume");
  spec.output<std::array<float, 3>>("spacing").condition(ConditionType::kNone);
  spec.output<std::array<uint32_t, 3>>("permute_axis").condition(ConditionType::kNone);
  spec.output<std::array<bool, 3>>("flip_axes").condition(ConditionType::kNone);
  spec.output<std::array<float, 3>>("extent").condition(ConditionType::kNone);
}

void VolumeLoaderOp::compute(InputContext& input, OutputContext& output,
                             ExecutionContext& context) {
  if (!allocator_.get()) { throw std::runtime_error("No allocator set."); }

  auto entity = gxf::Entity::New(&context);

  Volume volume;

  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  volume.allocator_ = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
                          context.context(), allocator_.get()->gxf_cid())
                          .value();

  volume.tensor_ =
      static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>("volume").value();

  if (is_nifty(file_name_.get())) {
    load_nifty(file_name_.get(), volume);
  } else if (is_mhd(file_name_.get())) {
    load_mhd(file_name_.get(), volume);
  } else if (is_nrrd(file_name_.get())) {
    load_nrrd(file_name_.get(), volume);
  } else {
    holoscan::log_error("File is not a supported volume format {}", file_name_.get());
  }

  output.emit(entity, "volume");
  output.emit(volume.spacing_, "spacing");
  output.emit(volume.permute_axis_, "permute_axis");
  output.emit(volume.flip_axes_, "flip_axes");

  std::array<float, 3> extent;
  nvidia::gxf::Shape shape = volume.tensor_->shape();
  for (int i = 0; i < extent.size(); ++i) {
    extent[i] = shape.dimension(shape.rank() - 1 - volume.permute_axis_[i]) *
                volume.spacing_[volume.permute_axis_[i]];
  }
  output.emit(extent, "extent");
}

}  // namespace holoscan::ops
