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

#include "guards.hpp"

namespace holoscan {
namespace gst {

// ============================================================================
// RAII Guard Implementations
// ============================================================================

template<typename T>
GstObjectGuard<T> make_gst_object_guard(T* object) {
  return std::shared_ptr<T>(object, [](T* obj) {
    if (obj) {
      gst_object_unref(obj);
    }
  });
}

// Explicit template instantiations for common types
template GstObjectGuard<GstElement> make_gst_object_guard<GstElement>(GstElement* object);
template GstObjectGuard<GstElementFactory> make_gst_object_guard<GstElementFactory>(GstElementFactory* object);
template GstObjectGuard<GstBus> make_gst_object_guard<GstBus>(GstBus* object);
template GstObjectGuard<GstCudaContext> make_gst_object_guard<GstCudaContext>(GstCudaContext* object);
template GstObjectGuard<GstAllocator> make_gst_object_guard<GstAllocator>(GstAllocator* object);

GstMessageGuard make_gst_message_guard(GstMessage* message) {
  return std::shared_ptr<GstMessage>(message, [](GstMessage* msg) {
    if (msg) {
      gst_message_unref(msg);
    }
  });
}

GstErrorGuard make_gst_error_guard(GError* error) {
  return std::shared_ptr<GError>(error, [](GError* err) {
    if (err) {
      g_error_free(err);
    }
  });
}

}  // namespace gst
}  // namespace holoscan

