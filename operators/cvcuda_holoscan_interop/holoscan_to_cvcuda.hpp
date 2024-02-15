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

#ifndef HOLOSCAN_OPERATORS_HOLOSCAN_TO_CVCUDA
#define HOLOSCAN_OPERATORS_HOLOSCAN_TO_CVCUDA

#include <memory>

#include <holoscan/holoscan.hpp>
#include <nvcv/Tensor.hpp>

namespace holoscan::ops {

/**
 * @brief This operator converts a Holoscan tensor to a CVCUDA tensor. It currently works with
 * tensor dimensions of less than or equal to 4. The tensor also needs to be on the device memory.
 *
 * Input is a Holoscan tensor as `holoscan::Tensor`, and output is a CVCUDA tensor as
 * `nvcv::Tensor`.
 *
 */
class HoloscanToCvCuda : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HoloscanToCvCuda);
  HoloscanToCvCuda() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override;

  /**
   * @brief This function converts a Holoscan tensor to a CVCUDA tensor in NHWC format. Even if
   * there is no batch dimension, the output tensor will be created in NHWC format.
   *
   * @param in_tensor the input Holoscan tensor
   * @param holoscan_tensor_data the pointer to the Holoscan tensor data. This pointer has a custom
   * deleter so that the memory is deallocated when the pointer goes out of scope.
   * @return the output CVCUDA tensor
   */
  static nvcv::Tensor to_cvcuda_NHWC_tensor(std::shared_ptr<holoscan::Tensor> in_tensor,
                                            std::shared_ptr<void*>& holoscan_tensor_data);

 private:
  std::shared_ptr<void*> holoscan_tensor_data_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_HOLOSCAN_TO_CVCUDA */
