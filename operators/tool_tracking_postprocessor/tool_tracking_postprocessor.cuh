/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>

#include <array>
#include <cstdint>

namespace holoscan::ops {
namespace tool_tracking_postprocessor {

void cuda_postprocess(uint32_t width, uint32_t height, const std::array<float, 3>& color,
                      bool first, const float* input, float4* output, cudaStream_t cuda_stream);

}  // namespace tool_tracking_postprocessor
}  // namespace holoscan::ops
