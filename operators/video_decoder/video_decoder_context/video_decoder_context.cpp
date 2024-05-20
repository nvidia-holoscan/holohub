/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "video_decoder_context.hpp"

namespace holoscan::ops {

void VideoDecoderContext::setup(ComponentSpec& spec) {
  spec.param(response_scheduling_term_,
             "async_scheduling_term",
             "Asynchronous Scheduling Condition",
             "Asynchronous Scheduling Condition");
  spec.param(device_id_,
             "device_id",
             "Cuda device id",
             "A valid device id, range is 0 to (cudaGetDeviceCount() - 1)",
             0);
}

}  // namespace holoscan::ops
