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

#ifndef HOLOSCAN__GSTREAMER__GST__WRAPPER_BASE_HPP
#define HOLOSCAN__GSTREAMER__GST__WRAPPER_BASE_HPP

namespace holoscan {
namespace gst {

/**
 * @brief Marker base class for all GStreamer wrapper types
 * 
 * This empty base class serves as a type marker to identify our GStreamer C++ wrappers.
 * It enables automatic unwrapping of wrapper objects when passed to set_properties().
 * Empty base optimization ensures zero runtime overhead.
 * 
 * All wrapper classes (ObjectBase, MiniObjectBase, and their derivatives) inherit from
 * this base class, allowing generic code to detect and automatically unwrap them.
 * 
 * @warning This is NOT a polymorphic base class - it has no virtual destructor.
 *          Do not delete derived objects through GstWrapperBase pointers.
 *          This class exists solely as a compile-time type marker for template metaprogramming.
 */
struct GstWrapperBase {};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__WRAPPER_BASE_HPP */
