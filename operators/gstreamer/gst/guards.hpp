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

#ifndef HOLOSCAN__GSTREAMER__GST__GUARDS_HPP
#define HOLOSCAN__GSTREAMER__GST__GUARDS_HPP

#include <memory>
#include <gst/gst.h>
#include <gst/cuda/gstcudacontext.h>

#include "gst/object.hpp"

namespace holoscan {
namespace gst {

// ============================================================================
// RAII Guards for GStreamer Objects
// ============================================================================

/**
 * @brief RAII wrapper for GstMessage with automatic cleanup
 */
using GstMessageGuard = std::shared_ptr<GstMessage>;

/**
 * @brief Create a RAII guard for a GstMessage that automatically calls gst_message_unref
 * @param message The GstMessage to wrap (takes ownership)
 * @return Shared pointer that will automatically unref the message when destroyed
 */
GstMessageGuard make_gst_message_guard(GstMessage* message);

/**
 * @brief RAII wrapper for GError with automatic cleanup
 */
using GstErrorGuard = std::shared_ptr<GError>;

/**
 * @brief Create a RAII guard for a GError that automatically calls g_error_free
 * @param error The GError to wrap (takes ownership)
 * @return Shared pointer that will automatically free the error when destroyed
 */
GstErrorGuard make_gst_error_guard(GError* error);

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__GUARDS_HPP */

