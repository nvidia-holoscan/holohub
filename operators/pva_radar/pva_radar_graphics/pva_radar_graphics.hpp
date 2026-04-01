/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PVA_RADAR_GRAPHICS_OP_HPP
#define PVA_RADAR_GRAPHICS_OP_HPP

#include "gxf/std/tensor.hpp"
#include "holoscan/holoscan.hpp"

#include <nvcv/Tensor.h>
#include <cstdint>
#include <memory>
#include <string>

namespace holoscan::ops {

/// Converts NVCV radar outputs (NCI image, peak count, DOA) into Holoscan/GXF types for
/// visualization: a gxf::Entity (RGBA image) and a TensorMap (xyz point cloud + origin lines).
class PVARadarGraphicsOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PVARadarGraphicsOp);

  PVARadarGraphicsOp();
  ~PVARadarGraphicsOp();

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<float> doa_scale_;

  // static tensors for the origin lines
  std::shared_ptr<Tensor> origin_x_tensor_;
  std::shared_ptr<Tensor> origin_y_tensor_;
  std::shared_ptr<Tensor> origin_z_tensor_;

  std::shared_ptr<Tensor> allocTensorSpace(nvidia::gxf::Shape shape,
                                           nvidia::gxf::PrimitiveType type);
};

}  // namespace holoscan::ops

#endif  // PVA_RADAR_GRAPHICS_OP_HPP
