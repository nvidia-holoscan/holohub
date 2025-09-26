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

#pragma once

#include <string>

// Forward declaration for Frame class - actual definition comes from StreamingServer headers
class Frame;

namespace holoscan::ops::debug_utils {

#ifdef HOLOSCAN_DEBUG_FRAME_WRITING
/**
 * @brief Utility function to write Frame to disk for debugging purposes
 *
 * This function writes frame data in multiple formats for debugging:
 * - Raw binary data (.bin)
 * - Metadata file (.txt) with frame properties and pixel samples
 * - PPM image file (.ppm) for visual inspection
 *
 * @param frame The frame to write to disk
 * @param filename_prefix Prefix for the output filenames
 * @param frame_number Optional frame number to include in filename (-1 to omit)
 * @return true if successful, false otherwise
 *
 * @note This function is only available when compiled with -DHOLOSCAN_DEBUG_FRAME_WRITING
 *       to avoid performance impact and disk I/O in production builds.
 */
bool writeFrameToDisk(const Frame& frame, const std::string& filename_prefix,
                      int frame_number = -1);
#endif  // HOLOSCAN_DEBUG_FRAME_WRITING

}  // namespace holoscan::ops::debug_utils
