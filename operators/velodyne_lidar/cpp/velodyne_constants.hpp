/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
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

#ifndef VELODYNE_CONSTANTS_HPP
#define VELODYNE_CONSTANTS_HPP

#include <stdint.h>

namespace data_collection {
namespace sensors {

/// @brief VLP-16 lidar has 16 emitters stacked vertically
const uint8_t kVLP16LineCount = 16;

// VLP16 pitch degree angles by index
constexpr float kVLP16PitchDegrees[kVLP16LineCount] = {
    -15.00,
    1.00,
    -13.00,
    3.00,
    -11.00,
    5.00,
    -9.00,
    7.00,
    -7.00,
    9.00,
    -5.00,
    11.00,
    -3.00,
    13.00,
    -1.00,
    15.00,
};

// Velodyne device types
enum __attribute__((__packed__)) VelodyneModel {
  VLP16 = 0x22,
  HDL32 = 0x21,
  VLP32C = 0x28,
};

// Velodyne packet structure for 16 and 32 line Lidars
const uint32_t kVelodyneRecords = 32;
const uint32_t kVelodyneBlocks = 12;

constexpr uint8_t kVLP16ColumnsPerBlock = 2;
constexpr uint32_t kVLP16PointsPerPacket = kVelodyneRecords * kVelodyneBlocks;

// Expected size of a VLP-16 UDP packet payload, ignoring the UDP header.
constexpr uint16_t kVLP16PacketSize = 1206;

// Packet azimuth (this is in packet units, which is in 100ths of degrees)
// This is not in floating point math because the packet values are in uint16_t
// and line up with this math.
const uint16_t kVelodyneMaxAzimuth = 36000;

// We need these structs to have no padding inserted into them so they can be
// serialized and deserialized easily.  We could do this on a per-struct basis
// but it's needed for all of these structs and reduces clutter.
#pragma pack(push, 1)

struct RawVelodyneRecord {
  // Distance in mm but in 2mm increments.  1 = 2mm
  uint16_t distance_two_millimeters_;
  // Intensity/reflectivity
  uint8_t intensity_;
};

struct RawVelodyneBlock {
  // Special Velodyne magic number FFEE
  uint16_t header_;
  // Angle in rotation of the lidar: in 100s of a degree
  // ie: 100.25 degrees is 10025
  uint16_t azimuth_hundredths_degrees_;
  // Each one of these is nanoseconds apart, and that time delta is determined
  // by the device parameters.
  RawVelodyneRecord records_[kVelodyneRecords];
};

// Sensor setting values:
// Return types
enum __attribute__((__packed__)) VelodyneReturnType {
  STRONG = 0x37,
  LAST = 0x38,
  DUAL = 0x39,
};

// Full binary packet from the Velodyne device
// (There is a UDP header, but that is ignored and consumed by the socket
// interface)
struct RawVelodynePacket {
  // There are 12 "firerings" per packet
  RawVelodyneBlock blocks_[kVelodyneBlocks];
  // Timestamp in minutes on the hour, utc
  uint32_t timestamp_microseconds_on_hour_;
  // Look at the constants defined right before this struct for values
  // Sensor settings: return can be strong, last, or dual
  union {
    VelodyneReturnType return_type_;
    uint8_t status_type_;
  };
  // Sensor info: model can be dlp16 or hdl32
  union {
    VelodyneModel model_;
    uint8_t status_value_;
  };
};

// End of packing attribute change.
#pragma pack(pop)  // see pack(push, 1) above

// VLP-16 reports in centidegrees
const float kBlockAngleToDegrees = 0.01;

// Math functions take radians but the angles from Velodyne are in degrees
const float kTwoPI = 3.14159265359 * 2.0;
const float kDegreesToRadians = kTwoPI / 360.0;

inline float RawVelodyneAngleToRadians(uint16_t block_angle) {
  return (block_angle * kBlockAngleToDegrees) * kDegreesToRadians;
}

// Distance is in 2mm increments by default
const uint8_t kRawVelodyneDefaultDistanceAccuracy = 2;
// Convert the 2mm distances to meters
const float kMillimetersToMeters = 0.001;

}  // namespace sensors
}  // namespace data_collection

#endif  // VELODYNE_CONSTANTS_HPP
