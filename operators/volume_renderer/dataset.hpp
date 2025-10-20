/* SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef OPERATORS_VOLUME_RENDERER_DATASET
#define OPERATORS_VOLUME_RENDERER_DATASET

#include <claraviz/interface/DataInterface.h>
// ClaraViz is defining RuntimeError which collides with Holoscan
#undef RuntimeError

#include <gxf/std/tensor.hpp>
#if !__has_include("gxf/std/dlpack_utils.hpp")
  // Holoscan 1.0 used GXF without DLPack so gxf_tensor.hpp was needed to add it
  #include <holoscan/core/gxf/gxf_tensor.hpp>
#endif

#include <chrono>
#include <memory>
#include <string>
#include <vector>

class DataArray;

/**
 * Defines a volume dataset, includes a density volume and optional a segmentation volume. Also
 * supports volume sequences.
 */
class Dataset {
 public:
  using Handle = std::shared_ptr<Dataset>;  /// a dataset handle

  enum class Types { Density, Segmentation };  /// supported volume types

  /**
   * Set a volume of the dataset.
   *
   * @param type volume type
   * @param spacing physical size of a element
   * @param permute_axis permutes the given data axes, e.g. to swap x and y of a volume specify (1,
   * 0, 2)
   * @param flip_axes Flips the given axes
   * @param element_range optional range of the values contained in the volume, if the vector is empty
   * then the range is calculated form the data. For example a full range of a uint8 data type is
   * defined by {0.f, 255.f}.
   * @param tensor volume data
   */
  void SetVolume(Types type, const std::array<float, 3>& spacing,
                 const std::array<uint32_t, 3>& permute_axis, const std::array<bool, 3>& flip_axes,
                 const std::vector<clara::viz::Vector2f>& element_range,
                 const nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor);

  /**
   * Reset a volume of the dataset.
   *
   * @param type volume type
   */
  void ResetVolume(Types type);

  /**
   * Sent the dataset configuration to the ClaraViz data configuration interface.
   *
   * @param data_config_interface ClaraViz data config interface
   */
  void Configure(clara::viz::DataConfigInterface& data_config_interface);
  /**
   * Set the volume data used by ClaraViz for a given frame.
   *
   * @param data_interface ClaraViz data interface
   * @param frame_index Frame index (optional)
   */
  void Set(clara::viz::DataInterface& data_interface, uint32_t frame_index = 0);

  /**
   * @returns Get the number of frames stored in this dataset
   */
  uint32_t GetNumberFrames() const;

  /**
   * @brief Set the frame duration
   */
  void SetFrameDuration(const std::chrono::duration<float>& frame_duration);
  /**
   * @returns the frame duration
   */
  const std::chrono::duration<float>& GetFrameDuration() const;

 private:
  std::vector<std::shared_ptr<DataArray>> density_;
  std::vector<std::shared_ptr<DataArray>> segmentation_;
  std::chrono::duration<float> frame_duration_ = std::chrono::duration<float>(1);
};

#endif /* OPERATORS_VOLUME_RENDERER_DATASET */
