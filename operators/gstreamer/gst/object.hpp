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

#ifndef GSTREAMER_GST_OBJECT_HPP
#define GSTREAMER_GST_OBJECT_HPP

#include <memory>
#include <gst/gst.h>
#include <gst/cuda/gstcudacontext.h>

namespace holoscan {
namespace gst {

/**
 * @brief RAII wrapper for GStreamer objects with automatic cleanup
 * @tparam T The GStreamer object type (::GstElement, ::GstBus, etc.)
 */
template<typename T>
class Object {
 private:
  std::shared_ptr<T> ptr_;

 public:
  // Constructors
  Object() = default;
  
  // Constructor from raw pointer (takes ownership)
  explicit Object(T* object) : ptr_(object, [](T* obj) {
    if (obj) {
      gst_object_unref(obj);
    }
  }) {}
  
  // Copy constructor and assignment
  Object(const Object&) = default;
  Object& operator=(const Object&) = default;
  
  // Move constructor and assignment
  Object(Object&&) noexcept = default;
  Object& operator=(Object&&) noexcept = default;
  
  // Get the raw pointer
  T* get() const { return ptr_.get(); }
  
  // Dereference operators
  T& operator*() const { return *ptr_; }
  T* operator->() const { return ptr_.get(); }
  
  // Bool conversion
  explicit operator bool() const { return ptr_ != nullptr; }
  
  // Increment GStreamer reference count and return the raw pointer
  // Useful when you need to pass ownership to GStreamer APIs
  T* ref() const {
    if (ptr_) {
      gst_object_ref(ptr_.get());
      return ptr_.get();
    }
    return nullptr;
  }
  
  // Transfer ownership out of the guard
  // Increments the ref count, resets the guard, and returns the pointer
  T* release() {
    auto result = ref();
    ptr_.reset();
    return result;
  }
  
  // Reset the guard
  void reset() { ptr_.reset(); }
  
  // Get the underlying shared_ptr
  const std::shared_ptr<T>& shared_ptr() const { return ptr_; }
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
 * @brief Convenience alias for ::GstCudaContext
 */
using CudaContext = Object<::GstCudaContext>;

/**
 * @brief Convenience alias for ::GstAllocator
 */
using Allocator = Object<::GstAllocator>;

}  // namespace gst
}  // namespace holoscan

#endif /* GSTREAMER_GST_OBJECT_HPP */

