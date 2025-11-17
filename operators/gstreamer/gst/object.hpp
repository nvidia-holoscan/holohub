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

#ifndef HOLOSCAN__GSTREAMER__GST__OBJECT_HPP
#define HOLOSCAN__GSTREAMER__GST__OBJECT_HPP

#include <gst/gst.h>

#include <functional>
#include <memory>

#include "config.hpp"

#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
#include <gst/cuda/gstcudacontext.h>
#endif  // HOLOSCAN_GSTREAMER_CUDA_SUPPORT

namespace holoscan {
namespace gst {

/**
 * @brief RAII wrapper for GStreamer objects with automatic cleanup
 * @tparam T The GStreamer object type (e.g. ::GstElement, ::GstCaps, ::GstBuffer, etc.)
 * @tparam RefFunc Function to increment reference count
 * @tparam UnrefFunc Function to decrement reference count
 * @tparam RefSinkFunc Function to sink floating references (nullptr for types without floating
 * refs)
 *
 * @note This class is designed for code reuse via inheritance, NOT for polymorphic use.
 *       It has no virtual functions and should never be used as a polymorphic base class.
 *       Always use concrete types (Buffer, Memory, Caps, etc.) directly.
 */
template <typename T, T* (*RefFunc)(T*), void (*UnrefFunc)(T*), T* (*RefSinkFunc)(T*) = nullptr>
class Object {
 public:
  // Constructor from raw pointer (takes ownership)
  explicit Object(T* object = nullptr)
      : ptr_(object, [](T* obj) {
          if (obj) UnrefFunc(obj);
        }) {
    if (object && RefSinkFunc) {
      RefSinkFunc(object);
    }
  }

  ~Object() = default;

  // Enable copy semantics
  Object(const Object& other) = default;
  Object& operator=(const Object& other) = default;

  // Enable move semantics
  Object(Object&& other) = default;
  Object& operator=(Object&& other) = default;

  // Get the raw pointer
  T* get() const { return ptr_.get(); }

  // Access GStreamer object members directly
  T* operator->() const { return ptr_.get(); }

  // Bool conversion
  explicit operator bool() const { return static_cast<bool>(ptr_); }

  // Increment GStreamer reference count and return the raw pointer
  T* ref() const {
    if (ptr_) return RefFunc(ptr_.get());
    return nullptr;
  }

  // Reset the guard
  void reset() { ptr_.reset(); }

 private:
  std::shared_ptr<T> ptr_;
};

// ============================================================================
// Helper Functions for GstObject Types
// ============================================================================

// Helper functions to adapt GStreamer's gpointer-based APIs to typed pointers
template <typename T>
inline T* gst_object_ref_typed(T* obj) {
  return static_cast<T*>(gst_object_ref(obj));
}

template <typename T>
inline void gst_object_unref_typed(T* obj) {
  gst_object_unref(obj);
}

template <typename T>
inline T* gst_object_ref_sink_typed(T* obj) {
  return static_cast<T*>(gst_object_ref_sink(obj));
}

// ============================================================================
// Helper Functions for GstMiniObject Types
// ============================================================================

// Helper functions for GstMiniObject-based types (Buffer, Memory, Caps, etc.)
// Note: Using reinterpret_cast because GStreamer types are opaque C structs
template <typename T>
inline T* gst_mini_object_ref_typed(T* obj) {
  return reinterpret_cast<T*>(gst_mini_object_ref(GST_MINI_OBJECT_CAST(obj)));
}

template <typename T>
inline void gst_mini_object_unref_typed(T* obj) {
  gst_mini_object_unref(GST_MINI_OBJECT_CAST(obj));
}

// ============================================================================
// Convenience Aliases
// ============================================================================

/**
 * @brief Convenience alias for ::GstElement
 */
using Element =
    Object<::GstElement, gst_object_ref_typed<::GstElement>, gst_object_unref_typed<::GstElement>,
           gst_object_ref_sink_typed<::GstElement>>;

/**
 * @brief Convenience alias for ::GstElementFactory
 */
using ElementFactory = Object<::GstElementFactory, gst_object_ref_typed<::GstElementFactory>,
                              gst_object_unref_typed<::GstElementFactory>,
                              gst_object_ref_sink_typed<::GstElementFactory>>;

/**
 * @brief Convenience alias for ::GstBus
 */
using Bus = Object<::GstBus, gst_object_ref_typed<::GstBus>, gst_object_unref_typed<::GstBus>,
                   gst_object_ref_sink_typed<::GstBus>>;

/**
 * @brief Convenience alias for ::GstAllocator
 */
using Allocator =
    Object<::GstAllocator, gst_object_ref_typed<::GstAllocator>,
           gst_object_unref_typed<::GstAllocator>, gst_object_ref_sink_typed<::GstAllocator>>;

/**
 * @brief Convenience alias for ::GstPad
 */
using Pad = Object<::GstPad, gst_object_ref_typed<::GstPad>, gst_object_unref_typed<::GstPad>,
                   gst_object_ref_sink_typed<::GstPad>>;

#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
/**
 * @brief Convenience alias for ::GstCudaContext
 */
using CudaContext =
    Object<::GstCudaContext, gst_object_ref_typed<::GstCudaContext>,
           gst_object_unref_typed<::GstCudaContext>, gst_object_ref_sink_typed<::GstCudaContext>>;
#endif  // HOLOSCAN_GSTREAMER_CUDA_SUPPORT

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__OBJECT_HPP */
