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

#include "velodyne_lidar.hpp"

#include <holoscan/holoscan.hpp>

#include <basic_network_operator_common.h>

#include "velodyne_constants.hpp"
#include "velodyne_convert_xyz.hpp"

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

namespace holoscan::ops {

nvidia::gxf::Shape VelodyneLidarOp::output_cloud_shape() {
  return {static_cast<int>(VLP16_PACKET_CLOUD_SIZE * packet_buffer_size_.get()),
          CLOUD_DIMENSION};
}

void VelodyneLidarOp::initialize() {
  holoscan::Operator::initialize();

  // Reserve and initialize space for the cloud tensor on the device
  float* device_xyz_intensity_buffer_;
  auto cloud_shape = output_cloud_shape();
  CUDA_TRY(cudaMalloc(&device_xyz_intensity_buffer_, cloud_shape.size() * sizeof(float)));
  CUDA_TRY(cudaMemset(device_xyz_intensity_buffer_, 0, cloud_shape.size() * sizeof(float)));

  // Construct Tensor from allocated device memory
  auto primitive_type = nvidia::gxf::PrimitiveType::kFloat32;
  auto gxf_cloud_tensor = std::make_shared<nvidia::gxf::Tensor>();
  gxf_cloud_tensor->wrapMemory(cloud_shape,
                               primitive_type,
                               nvidia::gxf::PrimitiveTypeSize(primitive_type),
                               nvidia::gxf::ComputeTrivialStrides(
                                   cloud_shape, nvidia::gxf::PrimitiveTypeSize(primitive_type)),
                               nvidia::gxf::MemoryStorageType::kDevice,
                               reinterpret_cast<void*>(device_xyz_intensity_buffer_),
                               [orig_pointer = device_xyz_intensity_buffer_](void*) mutable {
                                 CUDA_TRY(cudaFree(orig_pointer));
                                 return nvidia::gxf::Success;
                               });
  auto maybe_dl_ctx = gxf_cloud_tensor->toDLManagedTensorContext();
  if (!maybe_dl_ctx) {
    HOLOSCAN_LOG_ERROR(
        "failed to get std::shared_ptr<DLManagedTensorContext> from nvidia::gxf::Tensor");
  }
  cloud_tensor_ = std::make_shared<Tensor>(maybe_dl_ctx.value());
}

void VelodyneLidarOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                              holoscan::ExecutionContext& context) {
  // Advance our cloud ring buffer by one packet's cloud size on each tick, regardless of whether
  // the packet can be parsed.
  packet_buffer_index_ = (packet_buffer_index_ + 1) % packet_buffer_size_;

  auto data = op_input.receive<std::shared_ptr<NetworkOpBurstParams>>("burst_in");
  HOLOSCAN_LOG_DEBUG("First packet byte: " + std::to_string((*data)->data[0]) +
                     ", Last packet byte: " + std::to_string((*data)->data[(*data)->len - 1]) +
                     ", Length: " + std::to_string((*data)->len) +
                     ", Num packets: " + std::to_string((*data)->num_pkts));

  if ((*data)->len != VLP16_PACKET_SIZE) {
    HOLOSCAN_LOG_ERROR(
        "Received data length does not match expected VLP16 packet size. Expected: " +
        std::to_string(VLP16_PACKET_SIZE) + ", Received: " + std::to_string((*data)->len));
    return;
  }

  velodyne_helper_.ConvertRawPacketToDeviceXYZ(
      reinterpret_cast<data_collection::sensors::RawVelodynePacket*>((*data)->data),
      reinterpret_cast<data_collection::sensors::PointXYZ*>(
          reinterpret_cast<float*>(cloud_tensor_->data()) +
          (packet_buffer_index_ * VLP16_PACKET_CLOUD_SIZE * CLOUD_DIMENSION)));
  HOLOSCAN_LOG_DEBUG("Done processing Velodyne packet.");

  TensorMap out_message;
  out_message.insert({"xyz", cloud_tensor_});
  op_output.emit(out_message);
}

}  // namespace holoscan::ops
