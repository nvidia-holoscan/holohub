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

#ifndef PVA_RADAR_PIPELINE_OP_HPP
#define PVA_RADAR_PIPELINE_OP_HPP

#include "holoscan/holoscan.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace pva_radar {
class PVARadarPipeline;
}

namespace holoscan::ops {

class PVARadarPipelineOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PVARadarPipelineOp);

  PVARadarPipelineOp();
  ~PVARadarPipelineOp();

  PVARadarPipelineOp(PVARadarPipelineOp&&);
  PVARadarPipelineOp& operator=(PVARadarPipelineOp&&);

  void setup(OperatorSpec& spec) override;

  void initialize() override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  Parameter<int32_t> m_numSamples;
  Parameter<int32_t> m_numChirps;
  Parameter<int32_t> m_numRx;
  Parameter<int32_t> m_numTx;
  Parameter<int32_t> m_ddmRepeatFoldFactor;
  std::shared_ptr<pva_radar::PVARadarPipeline> pvaImpl_;
};

}  // namespace holoscan::ops

#endif  // PVA_RADAR_PIPELINE_OP_HPP
