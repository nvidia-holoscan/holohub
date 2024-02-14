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

#include <nvcv/Tensor.hpp>

#include "cvcuda_utils.hpp"
#include "gxf_utils.hpp"
#include "holoscan/holoscan.hpp"
#include "holoscan_to_cvcuda.hpp"

namespace holoscan::ops {

nvcv::Tensor HoloscanToCvCuda::to_cvcuda_NHWC_tensor(std::shared_ptr<holoscan::Tensor> in_tensor,
                                                     std::shared_ptr<void*>& holoscan_tensor_data) {
  // The output tensor will always be created in NHWC format even if no batch dimension existed
  // on the GXF tensor.
  int ndim = in_tensor->ndim();
  int batch_size, image_height, image_width, num_channels;
  auto in_shape = in_tensor->shape();
  if (ndim == 4) {
    batch_size = in_shape[0];
    image_height = in_shape[1];
    image_width = in_shape[2];
    num_channels = in_shape[3];
  } else if (ndim == 3) {
    batch_size = 1;
    image_height = in_shape[0];
    image_width = in_shape[1];
    num_channels = in_shape[2];
  } else if (ndim == 2) {
    batch_size = 1;
    image_height = in_shape[0];
    image_width = in_shape[1];
    num_channels = 1;
  } else {
    throw std::runtime_error(
        "expected a tensor with (height, width) or (height, width, channels) or "
        "(batch, height, width, channels) dimensions");
  }

  // buffer with strides defined for NHWC format
  auto in_buffer = holoscan::nhwc_buffer_from_holoscan_tensor(in_tensor, holoscan_tensor_data);
  nvcv::TensorShape cv_tensor_shape{{batch_size, image_height, image_width, num_channels},
                                    NVCV_TENSOR_NHWC};
  nvcv::DataType cv_dtype = dldatatype_to_nvcvdatatype(in_tensor->dtype());

  // Create a tensor buffer to store the data pointer and pitch bytes for each plane
  nvcv::TensorDataStridedCuda in_data(cv_tensor_shape, cv_dtype, in_buffer);

  // TensorWrapData allows for interoperation of external tensor representations with CVCUDA
  // Tensor.
  nvcv::Tensor cv_in_tensor = nvcv::TensorWrapData(in_data);
  return cv_in_tensor;
}

void HoloscanToCvCuda::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("input");
  // Keeping the output generic as gxf::Entity, so that we can also receive other data types in the
  // future
  spec.output<nvcv::Tensor>("output");
}

void HoloscanToCvCuda::compute(InputContext& op_input, OutputContext& op_output,
                               ExecutionContext&) {
  auto maybe_tensormap = op_input.receive<holoscan::TensorMap>("input");
  if (!maybe_tensormap.has_value()) {
    HOLOSCAN_LOG_ERROR("Failed to receive input message TensorMap");
    return;
  }

  auto tensormap = maybe_tensormap.value();

  HOLOSCAN_LOG_DEBUG("input tensor size: {}", tensormap.size());
  if (tensormap.size() > 1) {
    throw std::runtime_error("expected a single tensor in the input message but received " +
                             std::to_string(tensormap.size()) + " tensors");
  }
  auto holoscan_tensor = tensormap.begin()->second;

  // std::cout << "holoscan_tensor shape: ";
  // for (int i = 0; i < holoscan_tensor->ndim(); i++) {
  //   std::cout << holoscan_tensor->shape()[i] << ",";
  // }
  std::cout << std::endl;

  validate_holoscan_tensor(holoscan_tensor);

  const auto& cv_tensor = to_cvcuda_NHWC_tensor(holoscan_tensor, holoscan_tensor_data_);

  op_output.emit(cv_tensor, "output");
}

}  // namespace holoscan::ops