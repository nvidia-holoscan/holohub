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

#include "openigtlink_tx.hpp"

#include "holoscan/operators/holoviz/buffer_info.hpp"
#include "gxf/multimedia/video.hpp"

#include "igtl_util.h"

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

void identity_matrix(igtl::Matrix4x4& matrix) {
  // Populate IGTL matrix as identity
  matrix[0][0] = 1.0;  matrix[0][1] = 0.0;  matrix[0][2] = 0.0; matrix[0][3] = 0.0;
  matrix[1][0] = 0.0;  matrix[1][1] = 1.0;  matrix[1][2] = 0.0; matrix[1][3] = 0.0;
  matrix[2][0] = 0.0;  matrix[2][1] = 0.0;  matrix[2][2] = 1.0; matrix[2][3] = 0.0;
  matrix[3][0] = 0.0;  matrix[3][1] = 0.0;  matrix[3][2] = 0.0; matrix[3][3] = 1.0;
}

namespace holoscan::ops {

void OpenIGTLinkTxOp::setup(OperatorSpec& spec) {
  spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});
  spec.param(
    host_name_,
    "host_name",
    "HostName",
    "Host name.",
    std::string(""));
  spec.param(
    port_,
    "port",
    "Port",
    "Port number of server.",
    0);
  spec.param(
    device_name_,
    "device_name",
    "DeviceName",
    "OpenIGTLink device name",
    std::string("Holoscan"));
  spec.param(
    input_names_,
    "input_names",
    "InputNames",
    "Names of input messages.",
    std::vector<std::string>{});
}

void OpenIGTLinkTxOp::start() {
  // Create client socket
  client_socket_ = igtl::ClientSocket::New();
  HOLOSCAN_LOG_INFO("Connecting to OpenIGTLink server...");
  int r = client_socket_->ConnectToServer(host_name_.get().c_str(), port_.get());
  if (r < 0) {
    throw std::runtime_error("Cannot connect to server.");
  }
  HOLOSCAN_LOG_INFO("Connection successful");
  // Create timer
  time_stamp_ = igtl::TimeStamp::New();
}

void OpenIGTLinkTxOp::stop() {
  // Close connection
  client_socket_->CloseSocket();
}

void OpenIGTLinkTxOp::compute(InputContext& op_input, OutputContext& op_output,
              ExecutionContext& context) {
  std::vector<gxf::Entity> messages_h =
      op_input.receive<std::vector<gxf::Entity>>("receivers").value();

  std::vector<nvidia::gxf::Entity> messages;
  messages.reserve(messages_h.size());
  for (auto& message_h : messages_h) {
    // cast each holoscan::gxf:Entity to its base class
    nvidia::gxf::Entity message = static_cast<nvidia::gxf::Entity>(message_h);
    messages.push_back(message);
  }

  for (int i=0; i < input_names_.get().size(); ++i) {
    // Loop over input messages
    auto message = messages.begin();
    bool found = false;
    std::string name = input_names_.get()[i];

    // Check if message is in supported message types
    while (message != messages.end()) {
      // Message can be either tensor or video buffer
      auto maybe_input_tensor = message->get<nvidia::gxf::Tensor>(name.c_str());
      auto maybe_input_video = message->get<nvidia::gxf::VideoBuffer>(name.c_str());

      // Get buffer info
      BufferInfo buffer_info;
      gxf_result_t result;
      if (maybe_input_tensor) {
        result = buffer_info.init(maybe_input_tensor.value());
      } else if (maybe_input_video) {
        result = buffer_info.init(maybe_input_video.value());
      } else {
        ++message;
        continue;
      }
      found = true;

      if (result != GXF_SUCCESS) {
        throw std::runtime_error(
          fmt::format("Unsupported buffer format tensor/video buffer '{}'", name));
      }

      // If the buffer is empty, skip processing it
      if (buffer_info.bytes_size == 0) { continue; }

      // Get time stamp
      time_stamp_->GetTime();
      // Image properties
      int size[] = {
        static_cast<int>(buffer_info.width),
        static_cast<int>(buffer_info.height),
        1
      };
      float spacing[]  = {1.0, 1.0, 1.0};
      int endian = igtl::ImageMessage::ENDIAN_BIG;
      if (igtl_is_little_endian()) {
        endian = igtl::ImageMessage::ENDIAN_LITTLE;
      }
      // IGT scalar type from Holoscan data type
      int scalar_type;
      if (buffer_info.element_type == nvidia::gxf::PrimitiveType::kInt8) {
        scalar_type = igtl::ImageMessage::TYPE_INT8;
      } else if (buffer_info.element_type == nvidia::gxf::PrimitiveType::kUnsigned8) {
        scalar_type = igtl::ImageMessage::TYPE_UINT8;
      } else if (buffer_info.element_type == nvidia::gxf::PrimitiveType::kInt16) {
        scalar_type = igtl::ImageMessage::TYPE_INT16;
      } else if (buffer_info.element_type == nvidia::gxf::PrimitiveType::kUnsigned16) {
        scalar_type = igtl::ImageMessage::TYPE_UINT16;
      } else if (buffer_info.element_type == nvidia::gxf::PrimitiveType::kInt32) {
        scalar_type = igtl::ImageMessage::TYPE_INT32;
      } else if (buffer_info.element_type == nvidia::gxf::PrimitiveType::kUnsigned32) {
        scalar_type = igtl::ImageMessage::TYPE_UINT32;
      } else if (buffer_info.element_type == nvidia::gxf::PrimitiveType::kFloat32) {
        scalar_type = igtl::ImageMessage::TYPE_FLOAT32;
      } else if (buffer_info.element_type == nvidia::gxf::PrimitiveType::kFloat64) {
        scalar_type = igtl::ImageMessage::TYPE_FLOAT64;
      } else {
        throw std::runtime_error("Unsupported scalar type.");
      }
      // Create OpenIGTLink image message
      igtl::ImageMessage::Pointer image_msg = igtl::ImageMessage::New();
      image_msg->SetDimensions(size);
      image_msg->SetSpacing(spacing);
      image_msg->SetScalarType(scalar_type);
      image_msg->SetEndian(endian);
      image_msg->SetDeviceName(device_name_.get());
      image_msg->SetTimeStamp(time_stamp_);
      image_msg->SetNumComponents(buffer_info.components);
      image_msg->AllocateScalars();
      // Copy image data to message
      if (buffer_info.storage_type == nvidia::gxf::MemoryStorageType::kDevice) {
        // Copy device to host
        CUDA_TRY(cudaMemcpy(
          image_msg->GetScalarPointer(),
          static_cast<const void*>(buffer_info.buffer_ptr),
          buffer_info.bytes_size,
          cudaMemcpyDeviceToHost));
      } else {
        // Copy host to host
        std::memcpy(
          image_msg->GetScalarPointer(),
          static_cast<const void*>(buffer_info.buffer_ptr),
          image_msg->GetImageSize());
      }
      // Set orientation matrix to identity
      igtl::Matrix4x4 matrix;
      identity_matrix(matrix);
      image_msg->SetMatrix(matrix);
      // Send message
      image_msg->Pack();
      client_socket_->Send(image_msg->GetPackPointer(), image_msg->GetPackSize());

      break;
    }

    if (!found) {
      throw std::runtime_error(
        fmt::format("Tensor named `{}` not found in input messages.", name));
    }
  }
}

}  // namespace holoscan::ops
