/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Wayland Technologies. All rights reserved.
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

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

extern "C" {
typedef struct CUstream_st* cudaStream_t;
}

#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

class __attribute__((visibility("default"))) PointCloudFilterOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(PointCloudFilterOp, holoscan::Operator)

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  void ensure_structured_output_entities(const holoscan::ExecutionContext& context,
                                         size_t point_count);

  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> structured_allocator_;
  holoscan::Parameter<std::shared_ptr<holoscan::CudaStreamPool>> cuda_stream_pool_;
  holoscan::Parameter<float> min_z_;
  holoscan::Parameter<float> max_z_;
  holoscan::Parameter<float> max_x_;
  holoscan::Parameter<float> max_y_;

  static constexpr size_t kEntityRingSize = 4;
  std::array<std::optional<holoscan::gxf::Entity>, kEntityRingSize> structured_output_entities_;
  size_t structured_output_entity_index_{0};
  size_t structured_output_point_count_{0};

  bool first_cloud_logged_{false};
};

}  // namespace holoscan::ops
