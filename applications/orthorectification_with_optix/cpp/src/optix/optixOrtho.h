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

struct Params {
  OptixTraversableHandle handle;
  cudaTextureObject_t sensor_tex;
  uchar4 *image;
  unsigned int image_width;
  unsigned int image_height;
  float2 image_corner_coords;
  float image_gsd;
  float sensor_focal_length;
  float terrain_zmax;
  float3 sensor_pos;
  float3 sensor_up;
  float3 sensor_right;
  float3 sensor_focal_plane_origin;
  float2 sensor_focal_plane_size;
};

struct RayGenData {
  // No data needed
};

struct MissData {
  float3 bg_color;
};

struct HitGroupData {
  // No data needed
};
