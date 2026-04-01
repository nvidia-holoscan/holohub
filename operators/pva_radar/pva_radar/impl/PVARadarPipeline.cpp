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

#include "PVARadarPipeline.hpp"

#include <holoscan/logger/logger.hpp>

#include <ErrorCheckMacros.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <cmath>
#include <stdexcept>

namespace pva_radar {

PVARadarPipeline::PVARadarPipeline() = default;
PVARadarPipeline::~PVARadarPipeline() = default;

PVARadarPipeline::PVARadarPipeline(PVARadarPipeline&&) = default;
PVARadarPipeline& PVARadarPipeline::operator=(PVARadarPipeline&&) = default;

void PVARadarPipeline::init(int32_t numSamples, int32_t numChirps, int32_t numRx, int32_t numTx,
                            int32_t numDopplerFolds) {
  // Create tensor requirements and PVA operators
  RadarPipelineTensorRequirements tensorReqs(numSamples, numRx, numChirps, numDopplerFolds, numTx);
  radar_create_pva_workloads(m_pvaOperators, tensorReqs, numTx, numDopplerFolds);

  // Create input/output tensors
  m_pvaTensors =
      std::make_shared<RadarPipelineTensors>(m_pvaWorkloadParams.allocatorHandle, tensorReqs);

  m_initialized = true;
}

void PVARadarPipeline::process(NVCVTensorHandle rawADCIn) {
  if (!m_initialized) {
    throw std::runtime_error("PVA pipeline process called when not initialized");
  }
  m_pvaTensors->inRangeFFTTensorHandle = rawADCIn;
  radar_submit_pva_workloads(m_pvaOperators, *m_pvaTensors, m_pvaWorkloadParams);
}

NVCVTensorHandle PVARadarPipeline::getNciOutputHandle() const {
  return m_pvaTensors ? m_pvaTensors->outNciFinalTensorHandle : nullptr;
}

NVCVTensorHandle PVARadarPipeline::getPeakCountHandle() const {
  return m_pvaTensors ? m_pvaTensors->outPeakCountTensorHandle : nullptr;
}

NVCVTensorHandle PVARadarPipeline::getDOAOutputHandle() const {
  return m_pvaTensors ? m_pvaTensors->outDOATensorHandle : nullptr;
}

}  // namespace pva_radar
