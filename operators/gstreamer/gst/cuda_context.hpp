/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOLOSCAN__GSTREAMER__GST__CUDA_CONTEXT_HPP
#define HOLOSCAN__GSTREAMER__GST__CUDA_CONTEXT_HPP

#include <gst/gst.h>

#include <stdexcept>
#include <string>

#include "object.hpp"

#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
#include <gst/cuda/gstcuda.h>
#endif

namespace holoscan {
namespace gst {

#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
/**
 * @brief Wrapper class for ::GstCudaContext
 */
class CudaContext : public ObjectBase<CudaContext, ::GstCudaContext> {
 public:
  explicit CudaContext(::GstCudaContext* context = nullptr) : ObjectBase(context) {}

  // Implicit conversion from base ObjectBase type
  explicit CudaContext(const ObjectBase& other) : ObjectBase(other) {}
  explicit CudaContext(ObjectBase&& other) : ObjectBase(std::move(other)) {}

  /**
   * @brief Create a new CUDA context for the specified device
   * @param device_id CUDA device ID
   * @returns A new CudaContext object wrapping a newly created GstCudaContext
   * @throws std::runtime_error if context creation fails
   * @note Wraps gst_cuda_context_new() for type-safe context creation
   */
  static CudaContext create(gint device_id) {
    CudaContext context(gst_cuda_context_new(device_id));
    if (!context.get()) {
      throw std::runtime_error("Failed to create CUDA context for device " +
                               std::to_string(device_id));
    }
    return context.ref_sink();
  }

  // Provide GType for type-safe casting
  static GType get_type_func() { return GST_TYPE_CUDA_CONTEXT; }
};
#endif  // HOLOSCAN_GSTREAMER_CUDA_SUPPORT

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__CUDA_CONTEXT_HPP */
