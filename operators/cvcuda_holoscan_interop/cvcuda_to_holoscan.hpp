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

#ifndef HOLOSCAN_OPERATORS_CVCUDA_TO_HOLOSCAN
#define HOLOSCAN_OPERATORS_CVCUDA_TO_HOLOSCAN

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

/**
 * @brief This operator converts a CVCUDA tensor to a Holoscan tensor. It only works for tensor with
 * less than or equal to 4 dimensions (i.e., rank in CVCUDA tensor). If the batch size or channel
 * size is 1, then those dimensions are dropped in the Holoscan tensor.
 *
 * Input is a CVCUDA tensor as `nvcv::Tensor`, and output is a Holoscan tensor in
 * `holoscan::TensorMap`.
 *
 */
class CvCudaToHoloscan : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CvCudaToHoloscan);
  CvCudaToHoloscan() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_CVCUDA_TO_HOLOSCAN */
