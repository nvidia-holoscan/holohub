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
#include "gxf_utils.hpp"
#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

void CvCudaToHoloscan::setup(OperatorSpec& spec) {
  spec.input<nvcv::Tensor>("input");
  spec.output<holoscan::TensorMap>("output");
}

void CvCudaToHoloscan::compute(InputContext& op_input, OutputContext& op_output,
                               ExecutionContext& context) {
  auto cv_in_tensor = op_input.receive<nvcv::Tensor>("input").value();

  HOLOSCAN_LOG_DEBUG("cv_in_tensor retrieved");

  validate_cvcuda_tensor(cv_in_tensor);

  const auto& [out_message, tensor_data_pointer] =
      create_out_message_with_tensor(context.context(), cv_in_tensor);

  HOLOSCAN_LOG_DEBUG("create_out_message_with_tensor success");

  op_output.emit(out_message, "output");
}

}  // namespace holoscan::ops
