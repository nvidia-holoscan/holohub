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

#include "tensor_proto.hpp"

namespace holoscan::ops {

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

void TensorProto::gxf_time_to_proto(const nvidia::gxf::Entity& gxf_entity,
                                    ::holoscan::entity::Timestamp* timestamp) {
  auto gxf_timestamp = gxf_entity.get<nvidia::gxf::Timestamp>();
  if (gxf_timestamp) {
    timestamp->set_acqtime((*gxf_timestamp)->acqtime);
    timestamp->set_pubtime((*gxf_timestamp)->pubtime);
  }
}

void TensorProto::proto_to_gxf_time(nvidia::gxf::Entity& gxf_entity,
                                    const ::holoscan::entity::Timestamp& timestamp) {
  auto gxf_timestamp = gxf_entity.add<nvidia::gxf::Timestamp>("timestamp");
  (*gxf_timestamp)->acqtime = timestamp.acqtime();
  (*gxf_timestamp)->pubtime = timestamp.pubtime();
}

void TensorProto::gxf_tensor_to_proto(
    const nvidia::gxf::Entity& gxf_entity,
    google::protobuf::Map<std::string, ::holoscan::entity::Tensor>* tensor_map,
    const cudaStream_t cuda_stream) {
  auto tensors = gxf_entity.findAll<nvidia::gxf::Tensor, 4>();
  if (!tensors) { throw std::runtime_error("Tensor not found"); }

  for (auto tensor : tensors.value()) {
    holoscan::entity::Tensor& tensor_proto = (*tensor_map)[tensor->name()];
    for (uint32_t i = 0; i < (*tensor)->shape().rank(); i++) {
      tensor_proto.add_dimensions((*tensor)->shape().dimension(i));
    }
    switch ((*tensor)->element_type()) {
      case nvidia::gxf::PrimitiveType::kUnsigned8:
        tensor_proto.set_primitive_type(holoscan::entity::Tensor::kUnsigned8);
        copy_data_to_proto<uint8_t>(tensor.value(), tensor_proto, cuda_stream);
        break;
      case nvidia::gxf::PrimitiveType::kUnsigned16:
        tensor_proto.set_primitive_type(holoscan::entity::Tensor::kUnsigned16);
        copy_data_to_proto<uint16_t>(tensor.value(), tensor_proto, cuda_stream);
        break;
      case nvidia::gxf::PrimitiveType::kFloat32:
        tensor_proto.set_primitive_type(holoscan::entity::Tensor::kFloat32);
        copy_data_to_proto<float>(tensor.value(), tensor_proto, cuda_stream);
        break;
      default:
        throw std::runtime_error("Unsupported primitive type");
    }
  }
}

template <typename T>
void TensorProto::copy_data_to_proto(const nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor,
                                     ::holoscan::entity::Tensor& tensor_proto,
                                     const cudaStream_t cuda_stream) {
  if ((*tensor).storage_type() == nvidia::gxf::MemoryStorageType::kDevice) {
    void* in_data_ptr = (*tensor).pointer();
    size_t data_size = (*tensor).bytes_size();
    std::vector<T> in_data(data_size);
    CUDA_TRY(cudaMemcpyAsync(
        in_data.data(), in_data_ptr, data_size, cudaMemcpyDeviceToHost, cuda_stream));
    tensor_proto.set_data(in_data.data(), data_size);
    tensor_proto.set_memory_storage_type(holoscan::entity::Tensor::kDevice);
  } else {
    tensor_proto.set_data((*tensor).pointer(), (*tensor).size());
    tensor_proto.set_memory_storage_type(holoscan::entity::Tensor::kHost);
  }
}

void TensorProto::proto_to_gxf_tensor(
    nvidia::gxf::Entity& gxf_entity,
    const google::protobuf::Map<std::string, ::holoscan::entity::Tensor>& tensor_map,
    nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator, const cudaStream_t cuda_stream) {
  for (auto tensor_entry : tensor_map) {
    const holoscan::entity::Tensor& tensor_proto = tensor_entry.second;
    auto tensor = gxf_entity.add<nvidia::gxf::Tensor>(tensor_entry.first.c_str());
    if (!tensor) { throw std::runtime_error("Failed to create tensor"); }

    nvidia::gxf::Shape shape({tensor_proto.dimensions().begin(), tensor_proto.dimensions().end()});
    switch (tensor_proto.primitive_type()) {
      case holoscan::entity::Tensor::kUnsigned8:
        tensor.value()->reshape<uint8_t>(
            shape, memory_storage_type(tensor_proto.memory_storage_type()), allocator);
        copy_data_to_tensor<uint8_t>(tensor_proto, tensor.value(), cuda_stream);
        break;
      case holoscan::entity::Tensor::kUnsigned16:
        tensor.value()->reshape<uint16_t>(
            shape, memory_storage_type(tensor_proto.memory_storage_type()), allocator);
        copy_data_to_tensor<uint16_t>(tensor_proto, tensor.value(), cuda_stream);
        break;
      case holoscan::entity::Tensor::kFloat32:
        tensor.value()->reshape<float>(
            shape, memory_storage_type(tensor_proto.memory_storage_type()), allocator);
        copy_data_to_tensor<float>(tensor_proto, tensor.value(), cuda_stream);
        break;
      default:
        throw std::runtime_error("Unsupported primitive type");
    }
  }
}
nvidia::gxf::MemoryStorageType TensorProto::memory_storage_type(
    holoscan::entity::Tensor_MemoryStorageType mem_storage_type) {
  switch (mem_storage_type) {
    case holoscan::entity::Tensor::kDevice:
      return nvidia::gxf::MemoryStorageType::kDevice;
    case holoscan::entity::Tensor::kHost:
      return nvidia::gxf::MemoryStorageType::kHost;
    case holoscan::entity::Tensor::kSystem:
      return nvidia::gxf::MemoryStorageType::kSystem;
    default:
      throw std::runtime_error("Unsupported memory storage type");
  }
}

template <typename T>
void TensorProto::copy_data_to_tensor(const ::holoscan::entity::Tensor& tensor_proto,
                                      nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor,
                                      const cudaStream_t cuda_stream) {
  if (tensor_proto.memory_storage_type() == holoscan::entity::Tensor::kDevice) {
    CUDA_TRY(cudaMemcpyAsync((*tensor).pointer(),
                             tensor_proto.data().data(),
                             tensor_proto.data().size(),
                             cudaMemcpyHostToDevice,
                             cuda_stream));
  } else {
    std::copy(tensor_proto.data().begin(), tensor_proto.data().end(), (*tensor).pointer());
  }
}

void TensorProto::tensor_to_entity_request(const nvidia::gxf::Entity& gxf_entity,
                                           std::shared_ptr<EntityRequest> request,
                                           const cudaStream_t cuda_stream) {
  TensorProto::gxf_time_to_proto(gxf_entity, request->mutable_timestamp());
  TensorProto::gxf_tensor_to_proto(gxf_entity, request->mutable_tensors(), cuda_stream);
}

void TensorProto::tensor_to_entity_response(const nvidia::gxf::Entity& gxf_entity,
                                            std::shared_ptr<EntityResponse> response,
                                            const cudaStream_t cuda_stream) {
  TensorProto::gxf_time_to_proto(gxf_entity, response->mutable_timestamp());
  TensorProto::gxf_tensor_to_proto(gxf_entity, response->mutable_tensors(), cuda_stream);
}

void TensorProto::entity_request_to_tensor(
    const EntityRequest* entity_request, nvidia::gxf::Entity& gxf_entity,
    nvidia::gxf::Handle<nvidia::gxf::Allocator> gxf_allocator, const cudaStream_t cuda_stream) {
  TensorProto::proto_to_gxf_time(gxf_entity, entity_request->timestamp());
  TensorProto::proto_to_gxf_tensor(
      gxf_entity, entity_request->tensors(), gxf_allocator, cuda_stream);
}

void TensorProto::entity_response_to_tensor(
    const EntityResponse& entity_response, nvidia::gxf::Entity& gxf_entity,
    nvidia::gxf::Handle<nvidia::gxf::Allocator> gxf_allocator, const cudaStream_t cuda_stream) {
  TensorProto::proto_to_gxf_time(gxf_entity, entity_response.timestamp());
  TensorProto::proto_to_gxf_tensor(
      gxf_entity, entity_response.tensors(), gxf_allocator, cuda_stream);
}
}  // namespace holoscan::ops
