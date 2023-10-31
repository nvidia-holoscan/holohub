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

#ifndef HOLOHUB_APPLICATIONS_CVCUDA_BASIC_CVCUDA_UTILS_HPP
#define HOLOHUB_APPLICATIONS_CVCUDA_BASIC_CVCUDA_UTILS_HPP

#include <dlpack/dlpack.h>

#include <memory>

#include "gxf_utils.hpp"
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

/**
 * @brief Return the GXF tensor primitive type corresponding to a CV-CUDA dtype.
 *
 * The GXF primitive type does not include information on the number of interleaved channels.
 *
 * @param dtype CV-CUDA data type.
 * @return The GXF tensor primitive type
 */
nvidia::gxf::PrimitiveType nvcvdatatype_to_gxfprimitivetype(nvcv::DataType dtype);

/**
 * @brief Convert a holoscan::Tensor to a CV-CUDA tensor and corresponding Buffer class.
 *
 * This function only supports holoscan tensors with C-ordered memory layout and shape
 * corresponding to one of the following:
 * - (height, width)
 * - (height, width, channels)
 * - (batch, height, width, channels)
 *
 * @param in_tensor The holoscan tensor from which the CV-CUDA tensor will be initialized.
 * @return CV-CUDA tensor and corresponding buffer
 */
std::pair<nvcv::Tensor, nvcv::TensorDataStridedCuda::Buffer> holoscan_tensor_to_cvcuda_NHWC(
    std::shared_ptr<holoscan::Tensor> in_tensor);

/**
 * @brief Generate output message containing a single Holoscan tensor like reference_nhwc_tensor
 *
 * Note: If dimensions N or C have size 1, the corresponding dimensions will be dropped from
 * the output tensor.
 *
 * @param context The GXF context.
 * @param reference_nhwc_tensor The reference CV-CUDA tensor.
 * @return A GXF entity containing a single tensor like reference_nhwc_tensor.
 */
std::pair<nvidia::gxf::Entity, std::shared_ptr<void*>> create_out_message_with_tensor_like(
    gxf_context_t context, nvcv::Tensor reference_nhwc_tensor);

}  // namespace holoscan

#endif /* HOLOHUB_APPLICATIONS_CVCUDA_BASIC_CVCUDA_UTILS_HPP */
