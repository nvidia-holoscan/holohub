/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef APPLICATIONS_VOLUME_RENDERING_JSON_LOADER
#define APPLICATIONS_VOLUME_RENDERING_JSON_LOADER

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

class JsonLoaderOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(JsonLoaderOp);

  void setup(OperatorSpec& spec) override;
  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override;

 private:
  Parameter<std::vector<std::string>> file_names_;
};

}  // namespace holoscan::ops

#endif /* APPLICATIONS_VOLUME_RENDERING_JSON_LOADER */
