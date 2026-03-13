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

#include "pva_radar.hpp"

#include <ErrorCheckMacros.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Tensor.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "impl/PVARadarPipeline.hpp"

namespace holoscan::ops {

PVARadarPipelineOp::PVARadarPipelineOp() = default;
PVARadarPipelineOp::~PVARadarPipelineOp() = default;

PVARadarPipelineOp::PVARadarPipelineOp(PVARadarPipelineOp&&) = default;
PVARadarPipelineOp& PVARadarPipelineOp::operator=(PVARadarPipelineOp&&) = default;

void PVARadarPipelineOp::setup(OperatorSpec& spec) {
  spec.param(m_numSamples, "numSamples", "Number of samples");
  spec.param(m_numChirps, "numChirps", "Number of chirps");
  spec.param(m_numRx, "numRx", "Number of receive antennas");
  spec.param(m_numTx, "numTx", "Number of transmit antennas");
  spec.param(m_ddmRepeatFoldFactor, "ddmRepeatFoldFactor", "Number of Doppler folds");

  spec.input<NVCVTensorHandle>("input");

  spec.output<NVCVTensorHandle>("output_nci");
  spec.output<NVCVTensorHandle>("output_peak_count");
  spec.output<NVCVTensorHandle>("output_doa");
}

void PVARadarPipelineOp::initialize() {
  holoscan::Operator::initialize();
  pvaImpl_ = std::make_shared<pva_radar::PVARadarPipeline>();
}

void PVARadarPipelineOp::start() {
  if (!pvaImpl_) {
    pvaImpl_.reset(new pva_radar::PVARadarPipeline());
  }
  if (!pvaImpl_->isInitialized()) {
    pvaImpl_->init(m_numSamples, m_numChirps, m_numRx, m_numTx, m_ddmRepeatFoldFactor);
  }
}

void PVARadarPipelineOp::compute(InputContext& op_input, OutputContext& op_output,
                                 ExecutionContext& context) {
  // receive the required input from upstream operator
  NVCVTensorHandle input_tensor = op_input.receive<NVCVTensorHandle>("input").value();

  pvaImpl_->process(input_tensor);

  op_output.emit(pvaImpl_->getNciOutputHandle(), "output_nci");
  op_output.emit(pvaImpl_->getPeakCountHandle(), "output_peak_count");
  op_output.emit(pvaImpl_->getDOAOutputHandle(), "output_doa");
}

void PVARadarPipelineOp::stop() {
  pvaImpl_.reset();
}

}  // namespace holoscan::ops
