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

#include "message.hpp"

namespace holoscan {
namespace gst {

Error Message::parse_error() const {
  ::GError* error = nullptr;
  gst_message_parse_error(get(), &error, nullptr);
  return Error(error);
}

Error Message::parse_error(std::string& debug_info) const {
  ::GError* error = nullptr;
  gchar* debug = nullptr;
  gst_message_parse_error(get(), &error, &debug);

  // Copy debug info to output string.
  if (debug) {
    debug_info = debug;
    g_free(debug);
  } else {
    debug_info.clear();
  }

  return Error(error);
}

}  // namespace gst
}  // namespace holoscan
