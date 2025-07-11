/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef OPERATORS_NPP_FILTER_NPP_FILTER
#define OPERATORS_NPP_FILTER_NPP_FILTER

#include <memory>

#include <holoscan/core/resources/gxf/allocator.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

struct NppStreamContext_;

namespace holoscan::ops {

class NppFilterOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(NppFilterOp);

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override;

 private:
  Parameter<std::string> filter_;
  Parameter<uint32_t> mask_size_;
  Parameter<std::shared_ptr<Allocator>> allocator_;

  std::shared_ptr<NppStreamContext_> npp_stream_ctx_;

  CudaStreamHandler cuda_stream_handler_;
};

}  // namespace holoscan::ops

#endif /* OPERATORS_NPP_FILTER_NPP_FILTER */
