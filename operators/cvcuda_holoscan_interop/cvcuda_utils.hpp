/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights
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

#ifndef HOLOHUB_OPERATORS_CVCUDA_HOLOSCAN_INTEROP_CVCUDA_UTILS_HPP
#define HOLOHUB_OPERATORS_CVCUDA_HOLOSCAN_INTEROP_CVCUDA_UTILS_HPP

#include <dlpack/dlpack.h>

#include <memory>

#include <gxf/std/tensor.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <nvcv/DataType.hpp>  // nvcv::DataType
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

namespace holoscan {

/**
 * @brief Convert from a DLPack data type structure to a corresponding CV-CUDA data type
 *
 * @param dtype DLPack data type.
 * @param num_channels The number of channels (optional). If specified, overrides `dtype.lanes`.
 * @return CV-CUDA data type.
 */
nvcv::DataType dldatatype_to_nvcvdatatype(DLDataType dtype, int num_channels = 0);

nvcv::TensorDataStridedCuda::Buffer nhwc_buffer_from_holoscan_tensor(
    std::shared_ptr<holoscan::Tensor> tensor, std::shared_ptr<void*>& holoscan_tensor_data);

void validate_cvcuda_tensor(nvcv::Tensor tensor);

}  // namespace holoscan

#endif /* HOLOHUB_OPERATORS_CVCUDA_HOLOSCAN_INTEROP_CVCUDA_UTILS_HPP */
