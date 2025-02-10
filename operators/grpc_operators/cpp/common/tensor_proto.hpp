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

#ifndef COMMON_TENSOR_PROTO_HPP
#define COMMON_TENSOR_PROTO_HPP

#include <cuda_runtime.h>
#include <gxf/std/tensor.hpp>
#include <memory>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/holoscan.hpp>

#include "holoscan.pb.h"

using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

namespace holoscan::ops {

/* TensorProto class
 * This class is responsible for converting data between GXF Entities and protobuf messages.
 */
class TensorProto {
 public:
  /**
   * @brief Converts a GXF entity to an EntityRequest.
   *
   * This static function takes a GXF entity and converts it into an EntityRequest,
   * which is a shared pointer to an EntityRequest object.
   *
   * @param gxf_entity The GXF entity to be converted.
   * @param request A shared pointer to an EntityRequest where the converted entity will be stored.
   */
  static void tensor_to_entity_request(const nvidia::gxf::Entity& gxf_entity,
                                       std::shared_ptr<EntityRequest> request,
                                       const cudaStream_t cuda_stream);
  /**
   * @brief Converts a GXF entity to an EntityResponse.
   *
   * This function takes a GXF entity and converts it into a shared pointer
   * of EntityResponse.
   *
   * @param gxf_entity The GXF entity to be converted.
   * @param response A shared pointer to the EntityResponse where the converted
   *                 entity will be stored.
   */
  static void tensor_to_entity_response(const nvidia::gxf::Entity& gxf_entity,
                                        std::shared_ptr<EntityResponse> response,
                                        const cudaStream_t cuda_stream);
  /**
   * @brief Converts an EntityRequest to a GXF entity.
   *
   * This function takes an EntityRequest object and converts it into a GXF tensor entity.
   * It uses the provided GXF allocator handle to allocate necessary resources.
   *
   * @param entity_request Pointer to the EntityRequest object to be converted.
   * @param gxf_entity Reference to the GXF entity where the tensor will be stored.
   * @param gxf_allocator Handle to the GXF allocator used for resource allocation.
   * @param cuda_stream CUDA stream used for memory operations.
   */
  static void entity_request_to_tensor(const EntityRequest* entity_request,
                                       nvidia::gxf::Entity& gxf_entity,
                                       nvidia::gxf::Handle<nvidia::gxf::Allocator> gxf_allocator,
                                       const cudaStream_t cuda_stream);
  /**
   * @brief Converts an EntityResponse to a GXF entity.
   *
   * This function takes an EntityResponse object and converts it into a GXF tensor,
   * storing the result in the provided GXF entity. It also uses the specified GXF
   * allocator for memory management.
   *
   * @param entity_request The EntityResponse object to be converted.
   * @param gxf_entity The GXF entity where the tensor will be stored.
   * @param gxf_allocator The GXF allocator used for memory management.
   */
  static void entity_response_to_tensor(const EntityResponse& entity_request,
                                        nvidia::gxf::Entity& gxf_entity,
                                        nvidia::gxf::Handle<nvidia::gxf::Allocator> gxf_allocator,
                                        const cudaStream_t cuda_stream);

 private:
  template <typename T>
  static void copy_data_to_proto(const nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor,
                                 ::holoscan::entity::Tensor& tensor_proto,
                                 const cudaStream_t cuda_stream);
  template <typename T>
  static void copy_data_to_tensor(const ::holoscan::entity::Tensor& tensor_proto,
                                  nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor,
                                  const cudaStream_t cuda_stream);
  static void gxf_time_to_proto(const nvidia::gxf::Entity& gxf_entity,
                                ::holoscan::entity::Timestamp* timestamp);
  static void gxf_tensor_to_proto(
      const nvidia::gxf::Entity& gxf_entity,
      google::protobuf::Map<std::string, ::holoscan::entity::Tensor>* tensor_map,
      const cudaStream_t cuda_stream);
  static void proto_to_gxf_time(nvidia::gxf::Entity& gxf_entity,
                                const ::holoscan::entity::Timestamp& timestamp);
  static void proto_to_gxf_tensor(
      nvidia::gxf::Entity& gxf_entity,
      const google::protobuf::Map<std::string, ::holoscan::entity::Tensor>& tensor_map,
      nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator, const cudaStream_t cuda_stream);
  static nvidia::gxf::MemoryStorageType memory_storage_type(
      holoscan::entity::Tensor_MemoryStorageType mem_storage_type);
};

}  // namespace holoscan::ops

#endif /* COMMON_TENSOR_PROTO_HPP */
