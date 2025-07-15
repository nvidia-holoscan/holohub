/* SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef VOLUME_LOADER_VOLUME
#define VOLUME_LOADER_VOLUME

#include <holoscan/holoscan.hpp>

#include <chrono>
#include <memory>
#include <vector>

namespace holoscan::ops {

/// This class holds the data and information for 3D volume
class Volume {
 public:
  /// Handle
  using Handle = std::shared_ptr<Volume>;

  /**
   * Set the orientation of the volume using anatomical orientation and defines the transformation
   * of the anatomical orientation to the image coordinate system. Each axis U, V, W of the image
   * coordinate system is described using one character of the anatomical orientation system:
   * A - Anterior (front)
   * P - Posterior (back)
   * S - Superior (head)
   * I - Inferior (feet)
   * L - Left
   * R - Right
   *
   * @param orientation [in] a string describing the anatomical orientation
   */
  bool SetOrientation(const std::string& orientation);

  /// spacing between elements in millimeter
  std::array<float, 3> spacing_{1.f, 1.f, 1.f};
  /// axis permutation
  std::array<uint32_t, 3> permute_axis_{0, 1, 2};
  /// axis flip
  std::array<bool, 3> flip_axes_{false, false, false};
  /// frame duration
  std::chrono::duration<float> frame_duration_;
  /// space origin
  std::array<double, 3> space_origin_{0.0, 0.0, 0.0};

  nvidia::gxf::MemoryStorageType storage_type_ = nvidia::gxf::MemoryStorageType::kDevice;
  nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator_;
  nvidia::gxf::Handle<nvidia::gxf::Tensor> tensor_;
};

}  // namespace holoscan::ops

#endif /* VOLUME_LOADER_VOLUME */
