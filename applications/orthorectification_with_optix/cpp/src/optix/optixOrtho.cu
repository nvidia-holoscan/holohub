/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <optix.h>
#include <builtin_types.h>
#include <optix_device.h>

#include "optixOrtho.h"
#include <cuda/helpers.h>

#include <sutil/vec_math.h>

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ void setPayload(float3 p) {
  optixSetPayload_0(float_as_int(p.x));
  optixSetPayload_1(float_as_int(p.y));
  optixSetPayload_2(float_as_int(p.z));
}

static __forceinline__ __device__ float2
computeSensorUV(const float3 hit_point) {
  const float3 origin_to_hit_point =
      hit_point - params.sensor_focal_plane_origin;
  const float dist_x = length(cross(params.sensor_up, origin_to_hit_point));
  const float dist_y = length(cross(params.sensor_right, origin_to_hit_point));
  return make_float2(dist_x, dist_y) / params.sensor_focal_plane_size;
}

static __device__ __inline__ float3 traceSensorRay(float3 ray_origin,
                                                   float3 ray_direction) {
  unsigned int p0 = 0, p1 = 0, p2 = 0;
  optixTrace(params.handle, ray_origin, ray_direction,
             0.0f,                      // Min intersection distance
             1e16f,                     // Max intersection distance
             0.0f,                      // rayTime -- used for motion blur
             OptixVisibilityMask(255),  // Specify always visible
             OPTIX_RAY_FLAG_NONE,
             0,  // SBT offset   -- See SBT discussion
             1,  // SBT stride   -- See SBT discussion
             0,  // missSBTIndex -- See SBT discussion
             p0, p1, p2);

  const float3 sensor_col =
      make_float3(int_as_float(p0), int_as_float(p1), int_as_float(p2));
  return sensor_col;
}

extern "C" __global__ void __raygen__rg() {
  // Lookup our location within the launch grid
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();

  // for nearest
  const float3 ray_origin = make_float3(
      idx.x * params.image_gsd + params.image_corner_coords.x,
      idx.y * params.image_gsd + params.image_corner_coords.y, -10.f);

  const float3 ray_direction = make_float3(0.f, 0.f, 1.f);

  // Trace the ray against our scene hierarchy
  unsigned int p0, p1, p2;
  optixTrace(params.handle, ray_origin, ray_direction,
             0.0f,                      // Min intersection distance
             1e16f,                     // Max intersection distance
             0.0f,                      // rayTime -- used for motion blur
             OptixVisibilityMask(255),  // Specify always visible
             OPTIX_RAY_FLAG_NONE,
             0,  // SBT offset   -- See SBT discussion
             1,  // SBT stride   -- See SBT discussion
             0,  // missSBTIndex -- See SBT discussion
             p0, p1, p2);

  if ((p0 > 0) && (p1 > 0) && (p2 > 0)) {
    const float3 result =
        make_float3(int_as_float(p0), int_as_float(p1), int_as_float(p2));

    const uchar4 clr = make_color(result);
    // Record results in the output raster
    params.image[idx.y * params.image_width + idx.x] = clr;
  }
}

extern "C" __global__ void __miss__ms() {
  const float3 sensor_col = make_float3(0., 0., 0.);
  setPayload(sensor_col);
}

extern "C" __global__ void __closesthit__terrain_ch() {
  // When built-in triangle intersection is used, a number of fundamental
  // attributes are provided by the OptiX API, indlucing barycentric
  // coordinates.
  const float3 ray_orig = optixGetWorldRayOrigin();
  const float3 ray_dir = optixGetWorldRayDirection();  // incident direction
  const float ray_t = optixGetRayTmax();

  // Lookup our location within the launch grid
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();

  const float3 hit_point = ray_orig + ray_t * ray_dir;
  const int index = idx.y * params.image_width + idx.x;

  if (hit_point.z < params.terrain_zmax) {  // We hit terrain, cast ray to sensor
    if (ray_orig.z > 0.f) {
      const float3 sensor_col = make_float3(0., 0., 0.);
      setPayload(sensor_col);
    } else {
      // // if you want to cast from sensor back to terrain
      // const float3 to_terrain = normalize(params.sensor_pos - hit_point);
      // const float3 sensor_col = traceSensorRay(params.sensor_pos,
      // to_terrain);

      // from terrain to sensor
      const float3 to_sensor = normalize(params.sensor_pos - hit_point);
      const float3 sensor_col =
          traceSensorRay(hit_point + 0.01f * to_sensor, to_sensor);

      setPayload(sensor_col);
    }
  } else {  // We hit the sensor plane
    const float2 sensor_uv = computeSensorUV(hit_point);
    // for nearest lookup
    const uchar4 sensor_rgba =
        tex2D<uchar4>(params.sensor_tex, sensor_uv.x, sensor_uv.y);
    const float3 sensor_col =
        make_float3(sensor_rgba.x, sensor_rgba.y, sensor_rgba.z) / 255.;

    // // for linear lookup
    // const float4 sensor_rgba = tex2D<float4>(params.sensor_tex, sensor_uv.x,
    // sensor_uv.y); const float3 sensor_col = make_float3(sensor_rgba.x,
    // sensor_rgba.y, sensor_rgba.z);

    setPayload(sensor_col);
  }
}
