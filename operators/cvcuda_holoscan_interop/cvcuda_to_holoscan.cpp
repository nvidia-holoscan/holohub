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

#include <cvcuda/OpReformat.hpp>
#include <nvcv/Tensor.hpp>
#include <vector>

#include "cvcuda_to_holoscan.hpp"
#include "cvcuda_utils.hpp"
#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

void CvCudaToHoloscan::setup(OperatorSpec& spec) {
  spec.input<nvcv::Tensor>("input");
  spec.output<holoscan::TensorMap>("output");
  spec.param(pool_, "pool", "Pool", "Pool to allocate the Holoscan tensor");
}

void CvCudaToHoloscan::compute(InputContext& op_input, OutputContext& op_output,
                               ExecutionContext& context) {
  auto cv_in_tensor = op_input.receive<nvcv::Tensor>("input").value();
  std::cout << "cv_in_tensor shape: " << cv_in_tensor.shape() << std::endl;

  validate_cvcuda_tensor(cv_in_tensor);

  HOLOSCAN_LOG_DEBUG("cv_in_tensor retrieved");

  HOLOSCAN_LOG_DEBUG("before create_out_message_with_tensor_like");
  const auto& [out_message, tensor_data_pointer] =
      create_out_message_with_tensor_like(context.context(), cv_in_tensor, pool_);
  HOLOSCAN_LOG_DEBUG("create_out_message_with_tensor_like success");

  //   nvcv::TensorDataStridedCuda::Buffer cv_out_buffer;
  //   auto in_strided_data = cv_in_tensor.exportData<nvcv::TensorDataStridedCuda>();
  //   cv_out_buffer.basePtr = static_cast<NVCVByte*>(*tensor_data_pointer);
  //   for (int i = 0; i < cv_in_tensor.rank(); i++) {
  //     cv_out_buffer.strides[i] = in_strided_data->stride(i);
  //   }

  //   nvcv::TensorDataStridedCuda out_data(cv_in_tensor.shape(), cv_in_tensor.dtype(),
  //   cv_out_buffer);

  //   nvcv::Tensor cv_out_tensor = nvcv::TensorWrapData(out_data);

  //   std::cout << "cv_out_tensor shape: " << cv_out_tensor.shape() << std::endl;

  //   cvcuda::Reformat reformatOp;
  //   reformatOp(0, cv_in_tensor, cv_out_tensor);
  //   HOLOSCAN_LOG_DEBUG("ReformatOp success");

  op_output.emit(out_message, "output");
}

}  // namespace holoscan::ops