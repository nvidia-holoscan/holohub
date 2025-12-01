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

#ifndef HOLOSCAN__GSTREAMER__GST__ALLOCATOR_HPP
#define HOLOSCAN__GSTREAMER__GST__ALLOCATOR_HPP

#include <gst/gst.h>

#include "object.hpp"

#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
#include <gst/cuda/gstcudamemory.h>
#include <stdexcept>
#include <string>
#include "cuda_context.hpp"
#include "memory.hpp"
#include "video_info.hpp"
#endif

namespace holoscan {
namespace gst {

/**
 * @brief Templated base class for GStreamer allocator types (GstAllocator and its derivatives)
 * @tparam Derived The derived class type (for CRTP pattern)
 * @tparam NativeType The GStreamer allocator type (e.g. ::GstAllocator, specialized allocators)
 *
 * This template provides common allocator functionality for all GStreamer allocators.
 * Derive concrete classes like Allocator from this base class.
 */
template <typename Derived, typename NativeType>
class AllocatorBase : public ObjectBase<Derived, NativeType> {
 public:
  explicit AllocatorBase(NativeType* allocator = nullptr)
      : ObjectBase<Derived, NativeType>(allocator) {}

  // Implicit conversion from base ObjectBase type
  explicit AllocatorBase(const ObjectBase<Derived, NativeType>& other)
      : ObjectBase<Derived, NativeType>(other) {}
  explicit AllocatorBase(ObjectBase<Derived, NativeType>&& other)
      : ObjectBase<Derived, NativeType>(std::move(other)) {}

  // TODO: Add more common allocator functionality here (alloc, free, etc.)
};

/**
 * @brief Wrapper class for ::GstAllocator
 */
class Allocator : public AllocatorBase<Allocator, ::GstAllocator> {
 public:
  explicit Allocator(::GstAllocator* allocator = nullptr) : AllocatorBase(allocator) {}

  // Implicit conversion from base AllocatorBase type
  explicit Allocator(const AllocatorBase& other) : AllocatorBase(other) {}
  explicit Allocator(AllocatorBase&& other) : AllocatorBase(std::move(other)) {}

  // Provide GType for type-safe casting
  static GType get_type_func() { return GST_TYPE_ALLOCATOR; }
};

#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
/**
 * @brief Templated base class for GStreamer CUDA allocator types
 * @tparam Derived The derived class type (for CRTP pattern)
 * @tparam NativeType The GStreamer CUDA allocator type (e.g. ::GstCudaAllocator,
 * ::GstCudaPoolAllocator)
 *
 * This template provides common CUDA allocator functionality for all GStreamer CUDA allocators.
 */
template <typename Derived, typename NativeType>
class CudaAllocatorBase : public AllocatorBase<Derived, NativeType> {
 public:
  explicit CudaAllocatorBase(NativeType* allocator = nullptr)
      : AllocatorBase<Derived, NativeType>(allocator) {}

  // Implicit conversion from base AllocatorBase type
  explicit CudaAllocatorBase(const ObjectBase<Derived, NativeType>& other)
      : AllocatorBase<Derived, NativeType>(other) {}
  explicit CudaAllocatorBase(ObjectBase<Derived, NativeType>&& other)
      : AllocatorBase<Derived, NativeType>(std::move(other)) {}

  /**
   * @brief Wrap existing CUDA device memory in GstCudaMemory
   * @param context CUDA context for the memory
   * @param video_info Video format information for the memory
   * @param device_ptr CUDA device pointer to wrap
   * @param user_data User data to associate with the memory (for cleanup)
   * @param destroy_notify Callback to free user_data when memory is released
   * @returns A new Memory object wrapping the CUDA device memory
   * @note Wraps gst_cuda_allocator_alloc_wrapped() for type-safe usage
   */
  Memory alloc_wrapped(const CudaContext& context, const VideoInfo& video_info,
                       CUdeviceptr device_ptr, void* user_data,
                       GDestroyNotify destroy_notify) const {
    return Memory(gst_cuda_allocator_alloc_wrapped(GST_CUDA_ALLOCATOR(this->get()),
                                                   context.get(),
                                                   nullptr,  // CUDA stream (nullptr = default)
                                                   video_info.get(),  // video info
                                                   device_ptr,        // device pointer
                                                   user_data,         // user_data
                                                   destroy_notify));  // destroy_notify callback
  }
};

/**
 * @brief Templated base class for GStreamer CUDA pool allocator types
 * @tparam Derived The derived class type (for CRTP pattern)
 * @tparam NativeType The GStreamer CUDA pool allocator type (e.g. ::GstCudaPoolAllocator)
 *
 * This template provides common CUDA pool allocator functionality.
 */
template <typename Derived, typename NativeType>
class CudaPoolAllocatorBase : public CudaAllocatorBase<Derived, NativeType> {
 public:
  explicit CudaPoolAllocatorBase(NativeType* allocator = nullptr)
      : CudaAllocatorBase<Derived, NativeType>(allocator) {}

  // Implicit conversion from base types
  explicit CudaPoolAllocatorBase(const ObjectBase<Derived, NativeType>& other)
      : CudaAllocatorBase<Derived, NativeType>(other) {}
  explicit CudaPoolAllocatorBase(ObjectBase<Derived, NativeType>&& other)
      : CudaAllocatorBase<Derived, NativeType>(std::move(other)) {}

  // TODO: Add pool-specific functionality here (reset, get_config, etc.)
};

/**
 * @brief Wrapper class for ::GstCudaPoolAllocator
 */
class CudaPoolAllocator : public CudaPoolAllocatorBase<CudaPoolAllocator, ::GstCudaPoolAllocator> {
 public:
  explicit CudaPoolAllocator(::GstCudaPoolAllocator* allocator = nullptr)
      : CudaPoolAllocatorBase(allocator) {}

  // Implicit conversion from base types
  explicit CudaPoolAllocator(const CudaPoolAllocatorBase& other) : CudaPoolAllocatorBase(other) {}
  explicit CudaPoolAllocator(CudaPoolAllocatorBase&& other)
      : CudaPoolAllocatorBase(std::move(other)) {}

  // Provide GType for type-safe casting
  static GType get_type_func() { return GST_TYPE_CUDA_POOL_ALLOCATOR; }

  /**
   * @brief Create a CUDA pool allocator for efficient video memory management
   * @param context CUDA context for the allocator
   * @param video_info Video format information for the pool
   * @returns A new CudaPoolAllocator wrapping a GstCudaPoolAllocator
   * @throws std::runtime_error if allocator creation fails
   * @note Wraps gst_cuda_pool_allocator_new() for type-safe pool allocator creation
   * @note Pool allocators are better for video streaming as they reuse memory buffers
   * @note Available since GStreamer 1.24
   */
  static CudaPoolAllocator create(const CudaContext& context, const VideoInfo& video_info) {
    ::GstCudaPoolAllocator* pool_alloc =
        gst_cuda_pool_allocator_new(context.get(), nullptr, video_info.get());

    if (!pool_alloc) {
      throw std::runtime_error("Failed to create CUDA pool allocator");
    }

    return CudaPoolAllocator(pool_alloc).ref_sink();
  }
};
#endif  // HOLOSCAN_GSTREAMER_CUDA_SUPPORT

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__ALLOCATOR_HPP */
