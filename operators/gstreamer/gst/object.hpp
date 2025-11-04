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

#ifndef HOLOSCAN__GSTREAMER__GST__OBJECT_HPP
#define HOLOSCAN__GSTREAMER__GST__OBJECT_HPP

#include <functional>
#include <memory>
#include <gst/gst.h>
#include "config.hpp"

#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
#include <gst/cuda/gstcudacontext.h>
#endif  // HOLOSCAN_GSTREAMER_CUDA_SUPPORT

namespace holoscan {
namespace gst {

/**
 * @brief RAII wrapper for GStreamer objects with automatic cleanup
 * @tparam T The GStreamer object type (e.g. ::GstElement, ::GstCaps, ::GstBuffer, etc.)
 * 
 * Supports custom deleters for types with specialized ref/unref functions.
 * Default deleter uses gst_object_unref for GstObject-derived types.
 */
template<typename T>
class Object {
 public:
  using Deleter = std::function<void(T*)>; 
  // Constructor from raw pointer (takes ownership)
  explicit Object(T* object = nullptr, Deleter deleter = Deleter()) :
    ptr_(object, deleter ? deleter : [](T* obj) {
      if (obj)
        gst_object_unref(obj);
    }) {}

  virtual ~Object() = default;

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
  // Useful when you need to pass ownership to GStreamer APIs
  virtual T* ref() const {
    if (ptr_)
      return static_cast<T*>(gst_object_ref(ptr_.get()));
    return nullptr;
  }
  
  // Transfer ownership out of the guard
  // Increments the ref count, resets the guard, and returns the pointer
  T* release() {
    if (!ptr_)
      return nullptr;
    auto result = ref();
    reset();
    return result;
  }
  
  // Reset the guard
  void reset() { ptr_.reset(); }

 private:
  std::shared_ptr<T> ptr_;
};

// ============================================================================
// Convenience Aliases
// ============================================================================

/**
 * @brief Convenience alias for ::GstElement
 */
using Element = Object<::GstElement>;

/**
 * @brief Convenience alias for ::GstElementFactory
 */
using ElementFactory = Object<::GstElementFactory>;

/**
 * @brief Convenience alias for ::GstBus
 */
using Bus = Object<::GstBus>;

/**
 * @brief Convenience alias for ::GstAllocator
 */
using Allocator = Object<::GstAllocator>;

/**
 * @brief Convenience alias for ::GstPad
 */
 using Pad = Object<::GstPad>;


#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
/**
 * @brief Convenience alias for ::GstCudaContext
 */
using CudaContext = Object<::GstCudaContext>;
#endif  // HOLOSCAN_GSTREAMER_CUDA_SUPPORT

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__OBJECT_HPP */

