/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN__GSTREAMER__GST__PAD_HPP
#define HOLOSCAN__GSTREAMER__GST__PAD_HPP

#include <gst/gst.h>

#include "caps.hpp"
#include "object.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief Templated base class for GStreamer pad types (GstPad and its derivatives)
 * @tparam Derived The derived class type (for CRTP pattern)
 * @tparam NativeType The GStreamer pad type (e.g. ::GstPad, specialized pads)
 *
 * This template provides common pad functionality for all GStreamer pads.
 * Derive concrete classes like Pad from this base class.
 */
template <typename Derived, typename NativeType>
class PadBase : public ObjectBase<Derived, NativeType> {
 public:
  explicit PadBase(NativeType* pad = nullptr) : ObjectBase<Derived, NativeType>(pad) {}

  // Implicit conversion from base ObjectBase type
  explicit PadBase(const ObjectBase<Derived, NativeType>& other)
      : ObjectBase<Derived, NativeType>(other) {}
  explicit PadBase(ObjectBase<Derived, NativeType>&& other)
      : ObjectBase<Derived, NativeType>(std::move(other)) {}

  /// @brief Get the current caps of this pad
  /// @returns A Caps wrapper for the current caps, or empty Caps if no caps are set
  /// @note Returns owned reference from gst_pad_get_current_caps (no sinking needed)
  Caps get_current_caps() const { return Caps(gst_pad_get_current_caps(GST_PAD(this->get()))); }
};

/**
 * @brief Wrapper class for ::GstPad
 */
class Pad : public PadBase<Pad, ::GstPad> {
 public:
  explicit Pad(::GstPad* pad = nullptr) : PadBase(pad) {}

  // Implicit conversion from base types
  explicit Pad(const PadBase& other) : PadBase(other) {}
  explicit Pad(PadBase&& other) : PadBase(std::move(other)) {}

  // Provide GType for type-safe casting
  static GType get_type_func() { return GST_TYPE_PAD; }
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__PAD_HPP */
