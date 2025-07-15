/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "openigtlink_rx.hpp"

#include "igtlImageMessage.h"

#ifndef CUDA_TRY
#define CUDA_TRY(stmt)                                                                     \
  ({                                                                                       \
    cudaError_t _holoscan_cuda_err = stmt;                                                 \
    if (cudaSuccess != _holoscan_cuda_err) {                                               \
      GXF_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", \
                    #stmt,                                                                 \
                    __LINE__,                                                              \
                    __FILE__,                                                              \
                    cudaGetErrorString(_holoscan_cuda_err),                                \
                    _holoscan_cuda_err);                                                   \
    }                                                                                      \
    _holoscan_cuda_err;                                                                    \
  })
#endif

namespace holoscan::ops {

void OpenIGTLinkRxOp::setup(OperatorSpec& spec) {
  auto& out_tensor = spec.output<gxf::Entity>("out_tensor");

  spec.param(out_, "out", "Output", "Output channel.", &out_tensor);

  spec.param(
    port_,
    "port",
    "Port",
    "Port number of server.",
    0);

  spec.param(
    out_tensor_name_,
    "out_tensor_name",
    "OutTensorName",
    "Name of output tensor.",
    std::string(""));

  spec.param(
    flip_width_height_,
    "flip_width_height",
    "FlipWidthHeight",
    "Flip width and height (necessary for receiving from 3D Slicer).",
    true);
  spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
}

void OpenIGTLinkRxOp::start() {
  // Create client socket
  server_socket_ = igtl::ServerSocket::New();
  HOLOSCAN_LOG_INFO("Creating OpenIGTLink server socket...");
  int r = server_socket_->CreateServer(port_.get());
  if (r < 0) {
    throw std::runtime_error("Cannot create server socket.");
  }
  socket_ = server_socket_->WaitForConnection(1000);
  HOLOSCAN_LOG_INFO("Creating server socket successful");
  // Create timer
  time_stamp_ = igtl::TimeStamp::New();
}

void OpenIGTLinkRxOp::stop() {
  // Close connection
  socket_->CloseSocket();
  server_socket_->CloseSocket();
}

void OpenIGTLinkRxOp::compute(InputContext& op_input, OutputContext& op_output,
              ExecutionContext& context) {
  auto entity = nvidia::gxf::Entity::New(context.context());
  if (!entity) {
    throw std::runtime_error("Failed to allocate message for output tensor.");
  }

  // Allocate output tensor
  auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
    context.context(),
    allocator_->gxf_cid());

  auto tensor = entity.value().add<nvidia::gxf::Tensor>(out_tensor_name_.get().c_str());
  if (!tensor) {
    throw std::runtime_error("Failed to allocate output tensor.");
  }

  if (socket_.IsNotNull()) {
    igtl::MessageHeader::Pointer header;
    while (1) {
      header = igtl::MessageHeader::New();
      header->InitPack();
      bool timeout = false;
      igtlUint64 r = socket_->Receive(header->GetPackPointer(), header->GetPackSize(), timeout);
      if (r == 0) {
        throw std::runtime_error("Failed to receive message.");
      }
      if (r != header->GetPackSize()) {
        throw std::runtime_error("Packet size zero.");
      }

      // Deserialize the header
      header->Unpack();
      if (header->GetHeaderVersion() != IGTL_HEADER_VERSION_2) {
        throw std::runtime_error("Version of the client and server doesn't match.");
      }

      // Get time stamp
      igtlUint32 sec;
      igtlUint32 nanosec;
      header->GetTimeStamp(time_stamp_);
      time_stamp_->GetTimeStamp(&sec, &nanosec);

      if (strcmp(header->GetDeviceType(), "IMAGE") == 0) {
        // OBS: Could be GetDeviceName()
        break;
      } else {
        HOLOSCAN_LOG_INFO("Skipping : {}", header->GetDeviceType());
        socket_->Skip(header->GetBodySizeToRead(), 0);
      }
    }

    igtl::ImageMessage::Pointer image_msg;
    int scalar_type;
    int size[3];
    int num_components;

    // Receive image data
    image_msg = igtl::ImageMessage::New();
    image_msg->SetMessageHeader(header);
    image_msg->AllocatePack();

    bool timeout = false;
    socket_->Receive(image_msg->GetPackBodyPointer(), image_msg->GetPackBodySize(), timeout);
    int c = image_msg->Unpack(1);

    if (c & igtl::MessageHeader::UNPACK_BODY) {
      scalar_type = image_msg->GetScalarType();
      image_msg->GetDimensions(size);
      num_components = image_msg->GetNumComponents();
    } else {
      throw std::runtime_error("Unpacking body failed");
    }

    // Holoscan data type from IGT scalar type
    nvidia::gxf::PrimitiveType dtype;
    if (scalar_type == igtl::ImageMessage::TYPE_INT8) {
      dtype = nvidia::gxf::PrimitiveType::kInt8;
    } else if (scalar_type == igtl::ImageMessage::TYPE_UINT8) {
      dtype = nvidia::gxf::PrimitiveType::kUnsigned8;
    } else if (scalar_type == igtl::ImageMessage::TYPE_INT16) {
      dtype = nvidia::gxf::PrimitiveType::kInt16;
    } else if (scalar_type == igtl::ImageMessage::TYPE_UINT16) {
      dtype = nvidia::gxf::PrimitiveType::kUnsigned16;
    } else if (scalar_type == igtl::ImageMessage::TYPE_INT32) {
      dtype = nvidia::gxf::PrimitiveType::kInt32;
    } else if (scalar_type == igtl::ImageMessage::TYPE_UINT32) {
      dtype = nvidia::gxf::PrimitiveType::kUnsigned32;
    } else if (scalar_type == igtl::ImageMessage::TYPE_FLOAT32) {
      dtype = nvidia::gxf::PrimitiveType::kFloat32;
    } else if (scalar_type == igtl::ImageMessage::TYPE_FLOAT64) {
      dtype = nvidia::gxf::PrimitiveType::kFloat64;
    } else {
      throw std::runtime_error("Unsupported data type.");
    }
    // Shape
    nvidia::gxf::Shape shape;
    if (flip_width_height_) {
      shape = {size[1], size[0], num_components};
    } else {
      shape = {size[0], size[1], num_components};
    }
    // Size
    const uint64_t bytes_per_element = nvidia::gxf::PrimitiveTypeSize(dtype);
    auto strides = nvidia::gxf::ComputeTrivialStrides(shape, bytes_per_element);
    int bytes_size = size[0] * size[1] * num_components * bytes_per_element;
    // Allocate tensor
    auto reshape_result = tensor.value()->reshapeCustom(
        shape, dtype, bytes_per_element, strides, nvidia::gxf::MemoryStorageType::kDevice,
        allocator.value());
    if (!reshape_result) {
      throw std::runtime_error("Failed to generate tensor.");
    }
    // Copy data from OpenIGTLink message to tensor
    CUDA_TRY(cudaMemcpy(
      (void*)tensor.value()->pointer(),
      image_msg->GetScalarPointer(),
      bytes_size,
      cudaMemcpyHostToDevice));
  }

  // Emit output message
  auto result = gxf::Entity(std::move(entity.value()));
  op_output.emit(result);
}

}  // namespace holoscan::ops
