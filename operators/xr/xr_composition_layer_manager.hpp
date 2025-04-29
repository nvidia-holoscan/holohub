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

#ifndef XR_COMPOSITION_LAYER_MANAGER_HPP
#define XR_COMPOSITION_LAYER_MANAGER_HPP

#include <memory>
#include <vector>

#include "holoscan/holoscan.hpp"
#include "xr_session.hpp"
#include "xr_swapchain_cuda.hpp"
#include "xr_composition_layers.hpp"
#include "openxr/openxr.hpp"

namespace holoscan {

/*
 * XrCompositionLayerManager manages the composition layers for XR,
 * prepare everything needed to create an XR composition layer,
 * including swapchains and composition layer storage.
 */

class XrCompositionLayerManager : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS(XrCompositionLayerManager)

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  ~XrCompositionLayerManager();

  holoscan::Tensor acquire_color_swapchain_image();
  holoscan::Tensor acquire_depth_swapchain_image();
  void release_swapchain_images(cudaStream_t cuda_stream);

  // Create an XR composition layer using current frame state
  std::shared_ptr<XrCompositionLayerProjectionStorage> create_composition_layer(
      const xr::FrameState& xr_frame_state);

 private:
  Parameter<std::shared_ptr<holoscan::XrSession>> xr_session_;
  std::unique_ptr<XrSwapchainCuda> color_swapchain_;
  std::unique_ptr<XrSwapchainCuda> depth_swapchain_;
  bool is_initialized_;
};

}  // namespace holoscan

#endif /* XR_COMPOSITION_LAYER_MANAGER_HPP */
