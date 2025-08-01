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

[holoscan::input("in:probs")]
StructuredBuffer<float> probs;

[holoscan::input("in:scaled_coords")]
StructuredBuffer<float2> scaled_coords;

[holoscan::input("in:binary_masks")]
StructuredBuffer<float> binary_masks;

[holoscan::output("out:scaled_coords")]
[holoscan::alloc::size_of("in:scaled_coords.x")]
RWStructuredBuffer<float3> filtered_scaled_coords;

[holoscan::output("out:mask")]
// The inference operator omits the component count for single component buffers, e.g. the mask has
// a shape of (width, height). Therefore the swizzle is `cx` resulting in a allocation size of
// (4, width, height) (the 4 is automatically added by the operator since the data is a float4).
[holoscan::alloc::size_of("in:binary_masks.cx")]
[holoscan::zeros()]
RWStructuredBuffer<float4> colored_mask;

[holoscan::parameter("min_prob=0.5")]
float min_prob;

[holoscan::size_of("out:mask")]
uint3 binary_mask_size;

static const float3 colors[12] = {
  float3(0.12f, 0.47f, 0.71f),
  float3(0.20f, 0.63f, 0.17f),
  float3(0.89f, 0.10f, 0.11f),
  float3(1.00f, 0.50f, 0.00f),
  float3(0.42f, 0.24f, 0.60f),
  float3(0.69f, 0.35f, 0.16f),
  float3(0.65f, 0.81f, 0.89f),
  float3(0.70f, 0.87f, 0.54f),
  float3(0.98f, 0.60f, 0.60f),
  float3(0.99f, 0.75f, 0.44f),
  float3(0.79f, 0.70f, 0.84f),
  float3(1.00f, 1.00f, 0.60f),
};

// Apply the same workaround here as for the `colored_mask` buffer, see above.
[holoscan::invocations::size_of("in:binary_masks.cxy")]
[shader("compute")]
void filter_binary_mask_kernel(uint3 gid: SV_DispatchThreadID) {
  if ((gid.x >= binary_mask_size.x) || (gid.y >= binary_mask_size.y) ||
      (gid.z >= binary_mask_size.z)) {
    return;
  }

  const uint index = gid.z;

  // check if the probability meets the minimum probability
  if (probs[index] < min_prob) {
    return;
  }

  const uint offset = (gid.y * binary_mask_size.x) + gid.x;
  float value = binary_masks[offset];

  const float minV = 0.3f;
  const float maxV = 0.99f;
  const float range = maxV - minV;
  value = min(max(value, minV), maxV);
  value -= minV;
  value /= range;
  value *= 0.7f;

  const float4 dst = colored_mask[offset];
  colored_mask[offset] = float4((1.0f - value) * dst.x + colors[index].x * value,
                                (1.0f - value) * dst.y + colors[index].y * value,
                                (1.0f - value) * dst.z + colors[index].z * value,
                                (1.0f - value) * dst.w + 1.f * value);
}

[holoscan::invocations::size_of("in:scaled_coords")]
[shader("compute")]
void filter_coordinates_kernel(uint3 gid: SV_DispatchThreadID) {
  // the third component of the coordinate is the size of the crosses and the text
  constexpr float ITEM_SIZE = 0.05f;

  // check if the probability meets the minimum probability
  if (probs[gid.x] > min_prob) {
    filtered_scaled_coords[gid.x] =
        float3(scaled_coords[gid.x].x, scaled_coords[gid.x].y, ITEM_SIZE);
  } else {
    // move outside of the screen
    filtered_scaled_coords[gid.x] = float3(-1.f, -1.f, ITEM_SIZE);
  }
}
