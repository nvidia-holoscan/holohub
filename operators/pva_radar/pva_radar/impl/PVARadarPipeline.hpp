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

#ifndef PVA_RADAR_PIPELINE_HPP
#define PVA_RADAR_PIPELINE_HPP

#include <cupva_host.h>
#include <nvcv/Tensor.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "radar_pipeline_pva.hpp"
#include "radar_pipeline_tensors.hpp"

namespace pva_radar {

constexpr int32_t kMaxTargets = 1024;

// Implementation detail of PVARadarPipelineOp.
// This class encapsulates the pva-solutions radar pipeline
class PVARadarPipeline {
 public:
  PVARadarPipeline();
  ~PVARadarPipeline();

  PVARadarPipeline(PVARadarPipeline&&);
  PVARadarPipeline& operator=(PVARadarPipeline&&);
  // no copies allowed
  PVARadarPipeline& operator=(const PVARadarPipeline&) = delete;
  PVARadarPipeline(const PVARadarPipeline&) = delete;

  void init(int32_t numSamples, int32_t numChirps, int32_t numRx, int32_t numTx,
            int32_t numDopplerFolds);

  /// Runs PVA workloads. Output tensors are available via getters until next process().
  void process(NVCVTensorHandle rawADCIn);

  bool isInitialized() const { return m_initialized; }

  /// Get volatile handles to output tensors. They are not thread-safe so must not be accessed while
  /// process() is running. e.g., do not run operators that consume these tensors in parallel with
  /// the pva_radar operator or data corruption may be observed.
  NVCVTensorHandle getNciOutputHandle() const;
  NVCVTensorHandle getPeakCountHandle() const;
  NVCVTensorHandle getDOAOutputHandle() const;

 private:
  RadarPipelineOperators m_pvaOperators;
  std::shared_ptr<RadarPipelineTensors> m_pvaTensors;
  RadarPipelinePVAWorkloadParams m_pvaWorkloadParams;
  bool m_initialized = false;
};

}  // namespace pva_radar

#endif  // PVA_RADAR_PIPELINE_HPP
