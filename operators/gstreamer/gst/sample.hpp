/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN__GSTREAMER__GST__SAMPLE_HPP
#define HOLOSCAN__GSTREAMER__GST__SAMPLE_HPP

#include <gst/gst.h>

#include "buffer.hpp"
#include "mini_object.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief RAII wrapper for GstSample with automatic cleanup
 *
 * This class manages the lifetime of a GstSample object and automatically
 * calls gst_sample_unref when destroyed.
 */
class Sample : public MiniObjectBase<Sample, ::GstSample> {
 public:
  /**
   * @brief Constructor from raw pointer (takes ownership)
   * @param sample GstSample pointer to wrap (nullptr is allowed)
   */
  explicit Sample(::GstSample* sample = nullptr) : MiniObjectBase(sample) {}

  /**
   * @brief Get the buffer associated with this sample
   * @return Buffer object with incremented refcount (empty if sample has no buffer)
   * @note The returned buffer is independent of the sample's lifetime
   */
  Buffer get_buffer() const;

  // Define ref/unref functions required by MiniObjectBase
  static constexpr auto ref_func = gst_sample_ref;
  static constexpr auto unref_func = gst_sample_unref;
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__SAMPLE_HPP */
