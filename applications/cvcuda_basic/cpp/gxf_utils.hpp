/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOHUB_APPLICATIONS_CVCUDA_BASIC_GXF_UTILS_HPP
#define HOLOHUB_APPLICATIONS_CVCUDA_BASIC_GXF_UTILS_HPP

#include <memory>

#include <gxf/std/tensor.hpp>
#include <holoscan/core/domain/tensor.hpp>

namespace holoscan {

/**
 * @brief Generate output message containing a single tensor
 *
 * @param context The GXF context.
 * @param shape The element type of the tensor.
 * @param element_type The element type of the tensor.
 * @param storage_type The storage type of the tensor.
 * @return Pair containing the GXF entity as well as a std::shared_ptr<void*> corresponding to
 * the tensor data.
 */
std::pair<nvidia::gxf::Entity, std::shared_ptr<void*>> create_out_message_with_tensor(
    gxf_context_t context, nvidia::gxf::Shape shape, nvidia::gxf::PrimitiveType element_type,
    nvidia::gxf::MemoryStorageType storage_type = nvidia::gxf::MemoryStorageType::kDevice);

}  // namespace holoscan

#endif /* HOLOHUB_APPLICATIONS_CVCUDA_BASIC_GXF_UTILS_HPP */
