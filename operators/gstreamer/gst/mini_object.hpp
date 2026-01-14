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

#ifndef HOLOSCAN__GSTREAMER__GST__MINI_OBJECT_HPP
#define HOLOSCAN__GSTREAMER__GST__MINI_OBJECT_HPP

#include <gst/gst.h>

#include <memory>

namespace holoscan {
namespace gst {

/**
 * @brief Internal RAII base class for GStreamer GstMiniObject types with automatic cleanup
 * @tparam DerivedT The derived class type (for CRTP pattern)
 * @tparam NativeTypeT The GStreamer GstMiniObject type (e.g. ::GstBuffer, ::GstMemory, ::GstCaps,
 * etc.)
 *
 * This template provides automatic reference counting and cleanup for GstMiniObject-derived types.
 * Each GstMiniObject type has its own reference counting functions that can be specialized:
 * - ref_func: Function to increment reference count (e.g., gst_buffer_ref, gst_memory_ref)
 * - unref_func: Function to decrement reference count (e.g., gst_buffer_unref, gst_memory_unref)
 *
 * Derived classes must define these static constexpr members for their specific type.
 *
 * @note This class is for GstMiniObject hierarchy only (Buffer, Memory, Caps, Message, Sample,
 * etc.). For GstObject types (Element, Bus, Pad, Pipeline), use ObjectBase instead. GstMiniObject
 * types do NOT support floating references - there is no sink() method. GstMiniObject types do NOT
 * have GObject properties - there is no set_properties() method. This class is designed for
 * internal use only and has no virtual functions.
 */
template <typename DerivedT, typename NativeTypeT>
class MiniObjectBase {
 public:
  // Local type aliases for cleaner code inside the class
  using Derived = DerivedT;
  using NativeType = NativeTypeT;

  // Constructor from raw pointer (takes ownership)
  explicit MiniObjectBase(NativeType* object = nullptr)
      : ptr_(object, [](NativeType* obj) {
          if (obj)
            Derived::unref_func(obj);
        }) {}

  ~MiniObjectBase() = default;

  // Enable copy semantics
  MiniObjectBase(const MiniObjectBase& other) = default;
  MiniObjectBase& operator=(const MiniObjectBase& other) = default;

  // Enable move semantics
  MiniObjectBase(MiniObjectBase&& other) = default;
  MiniObjectBase& operator=(MiniObjectBase&& other) = default;

  // Get the raw pointer
  NativeType* get() const { return ptr_.get(); }

  // Access GStreamer object members directly
  NativeType* operator->() const { return ptr_.get(); }

  // Bool conversion
  explicit operator bool() const { return static_cast<bool>(ptr_); }

  // Increment GStreamer reference count and return reference for chaining
  Derived& ref() {
    if (ptr_)
      Derived::ref_func(ptr_.get());
    return static_cast<Derived&>(*this);
  }

  // Const overload for const objects
  const Derived& ref() const {
    if (ptr_)
      Derived::ref_func(ptr_.get());
    return static_cast<const Derived&>(*this);
  }

  // Note: No sink() method - GstMiniObject types don't support floating references
  // Note: No set_properties() method - GstMiniObject types don't have GObject properties

  // Reset the guard
  void reset() { ptr_.reset(); }

  // Derived classes must define these for their specific GstMiniObject type
  // Example: static constexpr auto ref_func = gst_buffer_ref;
  //          static constexpr auto unref_func = gst_buffer_unref;

 private:
  std::shared_ptr<NativeType> ptr_;
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__MINI_OBJECT_HPP */
