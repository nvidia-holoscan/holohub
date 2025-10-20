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

import holoscan;

[holoscan::input("input")]
[holoscan::output("output")]
RWStructuredBuffer<DATA_TYPE> buffer;

[holoscan::parameter("gamma=2.2")]
float gamma;

[holoscan::size_of("input")]
uint3 size;

[holoscan::strides_of("input")]
uint64_t3 strides;

#ifdef NORMALIZE
static const float range = (1 << (sizeof(DATA_TYPE) * 8)) - 1;
DATA_TYPE apply(DATA_TYPE value) {
  float fvalue = (float)value / range;
  fvalue = pow(fvalue, gamma);
  fvalue = fvalue * range + 0.5f;
  return DATA_TYPE(fvalue);
}
#else
float apply(float value) {
  return pow(value, gamma);
}
#endif

[holoscan::invocations::size_of("input")]
[shader("compute")]
void gamma_correction(uint3 gid: SV_DispatchThreadID) {
  if ((gid.x >= size.x) || (gid.y >= size.y)) {
    return;
  }

  const uint64_t offset = gid.x * strides.x + gid.y * strides.y;

  // apply gamma correction to each component except alpha
  buffer[offset] = apply(buffer[offset]);
#if COMPONENT_COUNT > 1
  buffer[offset + 1] = apply(buffer[offset + 1]);
#endif
#if COMPONENT_COUNT > 2
  buffer[offset + 2] = apply(buffer[offset + 2]);
#endif
}
