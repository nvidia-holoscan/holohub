/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_VIDEO_ENCODER_REQUEST_VIDEO_ENCODER_UTILS
#define HOLOSCAN_OPERATORS_VIDEO_ENCODER_REQUEST_VIDEO_ENCODER_UTILS

namespace nvidia::gxf {
enum struct EncoderInputFormat {
  kNV12 = 0,          // input format is NV12;
  kNV24 = 1,          // input format is NV24;
  kYUV420PLANAR = 3,  // input format is YUV420 planar;
  kUnsupported = 4    // Unsupported parameter
};

enum struct EncoderConfig {
  kIFrameCQP = 0,   // I frame only, CQP mode;
  kPFrameCQP = 1,   // IPP GOP, CQP
  kCustom = 2,      // Custom parameters
  kUnsupported = 3  // Unsupported parameter
};

}  // namespace nvidia::gxf

#endif /* HOLOSCAN_OPERATORS_VIDEO_ENCODER_REQUEST_VIDEO_ENCODER_UTILS */
