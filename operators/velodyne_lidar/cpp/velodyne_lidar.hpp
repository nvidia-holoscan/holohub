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

#ifndef HOLOSCAN_OPERATORS_VELODYNE_LIDAR_HPP
#define HOLOSCAN_OPERATORS_VELODYNE_LIDAR_HPP

#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

#include <basic_network_operator_common.h>
#include "velodyne_convert_xyz.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to process Velodyne VLP-16 lidar sensor input.
 *
 * This operator receives packets from a Velodyne VLP-16 lidar and
 * processes them into a point cloud of fixed size in Cartesian space.
 *
 * The operator performs the following steps:
 * 1. Interpret a fixed-size UDP packet as a Velodyne VLP-16 lidar packet,
 *    which contains 12 data blocks (azimuths) and 32 spherical data points per block.
 * 2. Transform the spherical data points into Cartesian coordinates (x, y, z)
 *    and add them to the output point cloud tensor, overwriting a previous cloud segment.
 * 3. Output the point cloud tensor and update the tensor insertion pointer to prepare
 *    for the next incoming packet.
 *
 * We recommend relying on HoloHub networking operators to receive Velodyne VLP-16 lidar packets
 * over UDP/IP and forward them to this operator.
 *
 * This operator was developed in part with support from the NVIDIA ISAAC team and adapts portions
 * of the NVIDIA DeepMap SDK.
 *
 * @see holoscan::ops::AdvNetworkOpRx
 * @see holoscan::ops::BasicNetworkOpRx
 * @see https://developer.nvidia.com/isaac
 * @see https://data.ouster.io/downloads/velodyne/user-manual/vlp-16-user-manual-revf.pdf
 */
class VelodyneLidarOp : public Operator {
 public:
  using PointXYZ = data_collection::sensors::PointXYZ;

  /// Velodyne VLP-16 lidar returns points in 3D Cartesian space.
  static const uint8_t CLOUD_DIMENSION = 3;

  static constexpr auto VLP16_PACKET_CLOUD_SIZE = data_collection::sensors::kVLP16PointsPerPacket;
  static constexpr auto VLP16_PACKET_SIZE = data_collection::sensors::kVLP16PacketSize;

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VelodyneLidarOp);

  VelodyneLidarOp() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override {
    // The incoming UDP packet to possibly translate to a VLP-16 cloud.
    spec.input<std::shared_ptr<NetworkOpBurstParams>>("burst_in");
    // The outgoing point cloud tensor.
    spec.output<holoscan::TensorMap>("cloud_out");

    // User parameter to set the size of the point cloud tensor buffer.
    // It typically takes between 80 and 82 packets to capture one
    // full revolution of the VLP-16 lidar sensor.
    spec.param<size_t>(packet_buffer_size_,
                       "packet_buffer_size",
                       "Packet buffer size",
                       "Ring buffer size for incoming packets",
                       80);
  };

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override;

 protected:
  // @brief Returns the shape of the output point cloud tensor.
  //
  // The point cloud tensor shape is fixed and determined by the packet buffer size
  // user parameter at the time the operator is initialized.
  nvidia::gxf::Shape output_cloud_shape();

 private:
  // Size of the point cloud tensor buffer in terms of discrete packets.
  Parameter<size_t> packet_buffer_size_;

  // Helper class to convert Velodyne packets to Cartesian data points
  data_collection::sensors::VelodyneConvertXYZHelper velodyne_helper_;

  // Internal buffer to store the latest point cloud state.
  //
  // The operator performs a one-time cloud allocation of length
  // (packet_buffer_size_ packets) * (12 data blocks / packet) * (32 points / data block).
  // Packets are translated to clouds of 384 points and inserted into the buffer,
  // overwriting some previous sub-cloud.
  std::shared_ptr<holoscan::Tensor> cloud_tensor_;

  // An increasing index to keep track of the number of packets processed.
  // Used to determine where points from the next packet should be inserted into the cloud buffer.
  size_t packet_buffer_index_ = 0;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_VELODYNE_LIDAR_HPP */
