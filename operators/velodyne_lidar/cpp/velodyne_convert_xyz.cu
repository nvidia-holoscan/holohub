/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "velodyne_constants.hpp"
#include "velodyne_convert_xyz.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

#include <holoscan/holoscan.hpp>

// Trigonometry tables for spherical -> Cartesian conversion.
__constant__ double d_vlp16_sin_pitch[data_collection::sensors::kVLP16LineCount];
__constant__ double d_vlp16_cos_pitch[data_collection::sensors::kVLP16LineCount];

#define CUDA_TRY(stmt)                                                                  \
  {                                                                                     \
    cudaError_t cuda_status = stmt;                                                     \
    if (cudaSuccess != cuda_status) {                                                   \
      HOLOSCAN_LOG_ERROR("Runtime call {} in line {} of file {} failed with '{}' ({})", \
                         #stmt,                                                         \
                         __LINE__,                                                      \
                         __FILE__,                                                      \
                         cudaGetErrorString(cuda_status),                               \
                         static_cast<int>(cuda_status));                                \
      throw std::runtime_error("Unable to copy device to host");                        \
    }                                                                                   \
  }

namespace data_collection {
namespace sensors {

__device__ uint16_t ConvertLittleEndianToBigEndian(uint16_t small_endian) {
  const uint8_t* bits = reinterpret_cast<const uint8_t*>(&small_endian);
  return (bits[1] << 8) + bits[0];
}

/// @brief  Convert a raw Velodyne VLP-16 packet to a list of XYZ points.
/// @param d_packet The 1206-byte packet in device memory to convert.
/// @param d_xyz_list The device memory destination for 384 XYZ points.
/// @param d_cos_rot_table Precomputed values for cosines of azimuth angles.
/// @param d_sin_rot_table Precomputed values for sines of azimuth angles.
/// @note The VLP-16 firing sequence does not follow the physical order of lasers,
///       but instead "jumps around". See the Velodyne VLP-16 User Manual for details.
__global__ void ConvertRawPacketToXYZ(data_collection::sensors::RawVelodynePacket* d_packet,
                                      Blocks* d_xyz_list, double* d_cos_rot_table,
                                      double* d_sin_rot_table) {
  // x: 0 -- 31, y: 0 -- 11
  // i: 0 -- 31, j: 0 -- 11
  int i = threadIdx.x;
  int j = threadIdx.y;

  // out of size
  if (i >= kVelodyneRecords || j >= kVelodyneBlocks) { return; }

  // Convert from the default Velodyne little endian format to the default
  // IGX big endian format.
  uint16_t azimuth =
      ConvertLittleEndianToBigEndian(d_packet->blocks_[j].azimuth_hundredths_degrees_);
  float range_meters =
      ConvertLittleEndianToBigEndian(d_packet->blocks_[j].records_[i].distance_two_millimeters_) *
      kRawVelodyneDefaultDistanceAccuracy * kMillimetersToMeters;

  // Use pre-computed lookup tables to translate from spherical coordinates
  // to Cartesian coordinates.
  d_xyz_list->blocks_[j].lines_[i].x =
      range_meters * d_vlp16_cos_pitch[i] * d_sin_rot_table[azimuth];
  d_xyz_list->blocks_[j].lines_[i].y =
      range_meters * d_vlp16_cos_pitch[i] * d_cos_rot_table[azimuth];
  d_xyz_list->blocks_[j].lines_[i].z = range_meters * d_vlp16_sin_pitch[i];
}

/// @brief Pre-compute sine and cosine values for rapid lookup.
void VelodyneConvertXYZHelper::InitSinAndCosTable() {
  HOLOSCAN_LOG_DEBUG("Initializing sin and cos tables in GPU.");
  auto sin_rot_table = std::vector<double>(kVelodyneMaxAzimuth);
  auto cos_rot_table = std::vector<double>(kVelodyneMaxAzimuth);

  for (int i = 0; i < kVelodyneMaxAzimuth; i++) {
    sin_rot_table[i] = sin(RawVelodyneAngleToRadians(i));
    cos_rot_table[i] = cos(RawVelodyneAngleToRadians(i));
  }

  auto vlp16_sin_pitch_table = std::vector<double>(kVLP16LineCount);
  auto vlp16_cos_pitch_table = std::vector<double>(kVLP16LineCount);

  for (int i = 0; i < kVLP16LineCount; i++) {
    vlp16_sin_pitch_table[i] =
        sin(data_collection::sensors::kVLP16PitchDegrees[i] * kDegreesToRadians);
    vlp16_cos_pitch_table[i] =
        cos(data_collection::sensors::kVLP16PitchDegrees[i] * kDegreesToRadians);
  }

  auto size_of_rot_table_bytes = sin_rot_table.size() * sizeof(double);
  CUDA_TRY(cudaMalloc((void**)&d_sin_rot_table_, size_of_rot_table_bytes));
  CUDA_TRY(cudaMemcpy(
      d_sin_rot_table_, sin_rot_table.data(), size_of_rot_table_bytes, cudaMemcpyHostToDevice));

  CUDA_TRY(cudaMalloc((void**)&d_cos_rot_table_, size_of_rot_table_bytes));
  CUDA_TRY(cudaMemcpy(
      d_cos_rot_table_, cos_rot_table.data(), size_of_rot_table_bytes, cudaMemcpyHostToDevice));

  auto size_of_pitch_table = vlp16_sin_pitch_table.size() * sizeof(double);
  CUDA_TRY(
      cudaMemcpyToSymbol(d_vlp16_sin_pitch, vlp16_sin_pitch_table.data(), size_of_pitch_table));
  CUDA_TRY(
      cudaMemcpyToSymbol(d_vlp16_cos_pitch, vlp16_cos_pitch_table.data(), size_of_pitch_table));

  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_packet_), sizeof(RawVelodynePacket)));

  HOLOSCAN_LOG_DEBUG("Finished initializing sin and cos tables.\n");
}

void VelodyneConvertXYZHelper::ConvertRawPacketToDeviceXYZ(
    const data_collection::sensors::RawVelodynePacket* packet, PointXYZ* gpu_xyz_destination) {
  if (!initialized_) {
    initialized_ = true;
    InitSinAndCosTable();
  }

  size_t packet_size = sizeof(RawVelodynePacket);
  CUDA_TRY(cudaMemcpy(d_packet_, packet, packet_size, cudaMemcpyHostToDevice));

  // Defines compute resource's size.
  dim3 block(kVelodyneRecords, kVelodyneBlocks);
  dim3 grid(1, 1);

  ConvertRawPacketToXYZ<<<grid, block>>>(d_packet_,
                                         reinterpret_cast<Blocks*>(gpu_xyz_destination),
                                         d_cos_rot_table_,
                                         d_sin_rot_table_);

  return;
}

VelodyneConvertXYZHelper::~VelodyneConvertXYZHelper() {
  cudaFree(d_sin_rot_table_);
  cudaFree(d_cos_rot_table_);
  cudaFree(d_packet_);
}

}  // namespace sensors
}  // namespace data_collection
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "velodyne_constants.hpp"
#include "velodyne_convert_xyz.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

#include <holoscan/holoscan.hpp>

// Trigonometry tables for spherical -> Cartesian conversion.
__constant__ double d_vlp16_sin_pitch[data_collection::sensors::kVLP16LineCount];
__constant__ double d_vlp16_cos_pitch[data_collection::sensors::kVLP16LineCount];

#define CUDA_TRY(stmt)                                                                  \
  {                                                                                     \
    cudaError_t cuda_status = stmt;                                                     \
    if (cudaSuccess != cuda_status) {                                                   \
      HOLOSCAN_LOG_ERROR("Runtime call {} in line {} of file {} failed with '{}' ({})", \
                         #stmt,                                                         \
                         __LINE__,                                                      \
                         __FILE__,                                                      \
                         cudaGetErrorString(cuda_status),                               \
                         static_cast<int>(cuda_status));                                \
      throw std::runtime_error("Unable to copy device to host");                        \
    }                                                                                   \
  }

namespace data_collection {
namespace sensors {

__device__ uint16_t ConvertLittleEndianToBigEndian(uint16_t small_endian) {
  const uint8_t* bits = reinterpret_cast<const uint8_t*>(&small_endian);
  return (bits[1] << 8) + bits[0];
}

/// @brief  Convert a raw Velodyne VLP-16 packet to a list of XYZ points.
/// @param d_packet The 1206-byte packet in device memory to convert.
/// @param d_xyz_list The device memory destination for 384 XYZ points.
/// @param d_cos_rot_table Precomputed values for cosines of azimuth angles.
/// @param d_sin_rot_table Precomputed values for sines of azimuth angles.
/// @note The VLP-16 firing sequence does not follow the physical order of lasers,
///       but instead "jumps around". See the Velodyne VLP-16 User Manual for details.
__global__ void ConvertRawPacketToXYZ(data_collection::sensors::RawVelodynePacket* d_packet,
                                      Blocks* d_xyz_list, double* d_cos_rot_table,
                                      double* d_sin_rot_table) {
  // x: 0 -- 31, y: 0 -- 11
  // i: 0 -- 31, j: 0 -- 11
  int i = threadIdx.x;
  int j = threadIdx.y;

  // out of size
  if (i >= kVelodyneRecords || j >= kVelodyneBlocks) { return; }

  // Convert from the default Velodyne little endian format to the default
  // IGX big endian format.
  uint16_t azimuth =
      ConvertLittleEndianToBigEndian(d_packet->blocks_[j].azimuth_hundredths_degrees_);
  float range_meters =
      ConvertLittleEndianToBigEndian(d_packet->blocks_[j].records_[i].distance_two_millimeters_) *
      kRawVelodyneDefaultDistanceAccuracy * kMillimetersToMeters;

  // Use pre-computed lookup tables to translate from spherical coordinates
  // to Cartesian coordinates.
  d_xyz_list->blocks_[j].lines_[i].x =
      range_meters * d_vlp16_cos_pitch[i] * d_sin_rot_table[azimuth];
  d_xyz_list->blocks_[j].lines_[i].y =
      range_meters * d_vlp16_cos_pitch[i] * d_cos_rot_table[azimuth];
  d_xyz_list->blocks_[j].lines_[i].z = range_meters * d_vlp16_sin_pitch[i];
}

/// @brief Pre-compute sine and cosine values for rapid lookup.
void VelodyneConvertXYZHelper::InitSinAndCosTable() {
  HOLOSCAN_LOG_DEBUG("Initializing sin and cos tables in GPU.");
  auto sin_rot_table = std::vector<double>(kVelodyneMaxAzimuth);
  auto cos_rot_table = std::vector<double>(kVelodyneMaxAzimuth);

  for (int i = 0; i < kVelodyneMaxAzimuth; i++) {
    sin_rot_table[i] = sin(RawVelodyneAngleToRadians(i));
    cos_rot_table[i] = cos(RawVelodyneAngleToRadians(i));
  }

  auto vlp16_sin_pitch_table = std::vector<double>(kVLP16LineCount);
  auto vlp16_cos_pitch_table = std::vector<double>(kVLP16LineCount);

  for (int i = 0; i < kVLP16LineCount; i++) {
    vlp16_sin_pitch_table[i] =
        sin(data_collection::sensors::kVLP16PitchDegrees[i] * kDegreesToRadians);
    vlp16_cos_pitch_table[i] =
        cos(data_collection::sensors::kVLP16PitchDegrees[i] * kDegreesToRadians);
  }

  auto size_of_rot_table_bytes = sin_rot_table.size() * sizeof(double);
  CUDA_TRY(cudaMalloc((void**)&d_sin_rot_table_, size_of_rot_table_bytes));
  CUDA_TRY(cudaMemcpy(
      d_sin_rot_table_, sin_rot_table.data(), size_of_rot_table_bytes, cudaMemcpyHostToDevice));

  CUDA_TRY(cudaMalloc((void**)&d_cos_rot_table_, size_of_rot_table_bytes));
  CUDA_TRY(cudaMemcpy(
      d_cos_rot_table_, cos_rot_table.data(), size_of_rot_table_bytes, cudaMemcpyHostToDevice));

  auto size_of_pitch_table = vlp16_sin_pitch_table.size() * sizeof(double);
  CUDA_TRY(
      cudaMemcpyToSymbol(d_vlp16_sin_pitch, vlp16_sin_pitch_table.data(), size_of_pitch_table));
  CUDA_TRY(
      cudaMemcpyToSymbol(d_vlp16_cos_pitch, vlp16_cos_pitch_table.data(), size_of_pitch_table));

  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_packet_), sizeof(RawVelodynePacket)));

  HOLOSCAN_LOG_DEBUG("Finished initializing sin and cos tables.\n");
}

void VelodyneConvertXYZHelper::ConvertRawPacketToDeviceXYZ(
    const data_collection::sensors::RawVelodynePacket* packet, PointXYZ* gpu_xyz_destination) {
  if (!initialized_) {
    initialized_ = true;
    InitSinAndCosTable();
  }

  size_t packet_size = sizeof(RawVelodynePacket);
  CUDA_TRY(cudaMemcpy(d_packet_, packet, packet_size, cudaMemcpyHostToDevice));

  // Defines compute resource's size.
  dim3 block(kVelodyneRecords, kVelodyneBlocks);
  dim3 grid(1, 1);

  ConvertRawPacketToXYZ<<<grid, block>>>(d_packet_,
                                         reinterpret_cast<Blocks*>(gpu_xyz_destination),
                                         d_cos_rot_table_,
                                         d_sin_rot_table_);

  return;
}

VelodyneConvertXYZHelper::~VelodyneConvertXYZHelper() {
  cudaFree(d_sin_rot_table_);
  cudaFree(d_cos_rot_table_);
  cudaFree(d_packet_);
}

}  // namespace sensors
}  // namespace data_collection
