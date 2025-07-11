/SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

// Ported from NVIDIA DeepMap SDK

#ifndef VELODYNE_CONVERT_XYZ_HPP
#define VELODYNE_CONVERT_XYZ_HPP

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <iostream>
#include <vector>

#include "velodyne_constants.hpp"

namespace data_collection {
namespace sensors {

/// @brief Defines one point in the 3D coordinate system relative to the VLP-16 sensor.
///
/// Values are assumed to be lengths reported in meters.
struct PointXYZ {
  float x;
  float y;
  float z;
};

/// @brief Points corresponding to a single azimuth in a single VLP-16 packet.
struct Lines {
  PointXYZ lines_[kVelodyneRecords];
};

/// @brief Sets of points corresponding to all azimuths in a single VLP-16 packet.
struct Blocks {
  Lines blocks_[kVelodyneBlocks];
};

/// @brief Helper class to convert VLP-16 packets to a list of XYZ points in device memory.
///
/// This class is used to convert raw Velodyne packets to a list of XYZ points via GPU.
/// It maintains an internal table of sin and cos values for yaw and pitch angles
/// to reuse across packet transformations.
/// Code is adapted from the NVIDIA Isaac DeepMap SDK.
///
/// @see https://developer.nvidia.com/isaac
class VelodyneConvertXYZHelper {
 public:
  // Destructor of class. Mainly free the memory space.
  ~VelodyneConvertXYZHelper();

  /// @brief Convert a raw Velodyne VLP-16 packet to a list of XYZ points.
  /// @param packet The 1206-byte packet in host memory to convert.
  /// @param gpu_xyz_destination The device memory destination for 384 XYZ points.
  ///
  /// One Velodyne VLP-16 packet describes:
  /// - 12 data blocks, each representing one azimuth angle;
  /// - 2 firing sequences per data block;
  /// - 16 spherical data points per firing sequence.
  ///
  /// This function copies a VLP-16 packet to device memory and
  /// converts it to a list of XYZ points in device memory.
  ///
  /// This function does NOT:
  /// - Encode additional information such as intensity;
  /// - Discern the arc covered by the packet
  void ConvertRawPacketToDeviceXYZ(const data_collection::sensors::RawVelodynePacket* packet,
                                   PointXYZ* gpu_xyz_intensity_destination);

 private:
  // Initialize all yaw table and pitch table, and copy them into Gpu
  // constants. This function will be called when first time call
  // RawVelodynePacketToXYZIntensityGpu.
  void InitSinAndCosTable();

  // sin yaw. Cross 360 degrees. GPU data. resolution 0.01 degrees.
  double* d_sin_rot_table_;
  // cos yaw. Cross 360 degrees. GPU data. resolution 0.01 degrees.
  double* d_cos_rot_table_;
  // Raw Velodyne packets. GPU data.
  RawVelodynePacket* d_packet_;
  // If this flag is false. When calling RawVelodynePacketToXYZIntensityGpu will
  // tries to init all tables.
  bool initialized_ = false;
};
}  // namespace sensors
}  // namespace data_collection

#endif  // VELODYNE_CONVERT_XYZ_HPP
