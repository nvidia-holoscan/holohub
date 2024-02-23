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

#ifndef VOLUME_RENDERER_DATASET
#define VOLUME_RENDERER_DATASET

#include <claraviz/interface/DataInterface.h>
// ClaraViz is defining RuntimeError which collides with Holoscan
#undef RuntimeError

#include <gxf/std/tensor.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

class DataArray;

class Dataset {
 public:
  using Handle = std::shared_ptr<Dataset>;

  enum class Types { Density, Segmentation };

  void SetVolume(Types type, const std::array<float, 3>& spacing,
                 const std::array<uint32_t, 3>& permute_axis, const std::array<bool, 3>& flip_axes,
                 const nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor);

  void ResetVolume(Types type);

  void Configure(clara::viz::DataConfigInterface& data_config_interface);
  void Set(clara::viz::DataInterface& data_interface, uint32_t frame_index = 0);

  uint32_t GetNumberFrames() const;

  void SetFrameDuration(const std::chrono::duration<float>& frame_duration);
  const std::chrono::duration<float>& GetFrameDuration() const;

 private:
  std::vector<std::shared_ptr<DataArray>> density_;
  std::vector<std::shared_ptr<DataArray>> segmentation_;
  std::chrono::duration<float> frame_duration_ = std::chrono::duration<float>(1);
};

#endif /* VOLUME_RENDERER_DATASET */
