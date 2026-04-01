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

/*
 * This file is based on pva-solutions/pipelines/radar/radar_pipeline_pva.hpp
 * Only minor modifications have been made to the original code. This is the original license text:
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RADAR_PIPELINE_PVA_HPP
#define RADAR_PIPELINE_PVA_HPP

#include "radar_pipeline_tensors.hpp"

#include <PvaOperator.h>
#include <cupva_host.h>
#include <cupva_host_scheduling.h>
#include <nvcv/alloc/Allocator.h>

#include <cstdint>

// ============================================================================
// RadarPipelineOperators - Manages PVA operator handles
// ============================================================================

class RadarPipelineOperators {
 public:
  RadarPipelineOperators() { initRadarPipelineOperators(); }
  ~RadarPipelineOperators() { cleanupRadarPipelineOperators(); }

  RadarPipelineOperators(RadarPipelineOperators&&) = default;
  RadarPipelineOperators& operator=(RadarPipelineOperators&&) = default;
  RadarPipelineOperators(const RadarPipelineOperators&) = delete;
  RadarPipelineOperators& operator=(const RadarPipelineOperators&) = delete;

  void initRadarPipelineOperators();
  void cleanupRadarPipelineOperators();

  NVCVOperatorHandle rangeFFTOperatorHandle;
  NVCVOperatorHandle dopplerFFTOperatorHandle;
  NVCVOperatorHandle nciOperatorHandle;
  NVCVOperatorHandle peakDetectionOperatorHandle;
  NVCVOperatorHandle doaOperatorHandle;
};

// ============================================================================
// RadarPipelinePVAWorkloadParams - Manages PVA workload parameters
// ============================================================================

class RadarPipelinePVAWorkloadParams {
 public:
  RadarPipelinePVAWorkloadParams() { initRadarPipelinePVAWorkloadParams(); }

  ~RadarPipelinePVAWorkloadParams() { cleanupRadarPipelinePVAWorkloadParams(); }

  RadarPipelinePVAWorkloadParams(RadarPipelinePVAWorkloadParams&&) = default;
  RadarPipelinePVAWorkloadParams& operator=(RadarPipelinePVAWorkloadParams&&) = default;
  RadarPipelinePVAWorkloadParams(const RadarPipelinePVAWorkloadParams&) = delete;
  RadarPipelinePVAWorkloadParams& operator=(const RadarPipelinePVAWorkloadParams&) = delete;

  void initRadarPipelinePVAWorkloadParams();
  void cleanupRadarPipelinePVAWorkloadParams();

  // Allocator
  NVCVAllocatorHandle allocatorHandle;

  // CUPVA objects
  cupvaSyncObj_t sync;
  cupvaFence_t fence;
  cupvaCmd_t rf;
  cupvaStream_t stream;
};

// ============================================================================
// PVA Workload Functions
// ============================================================================

// Function to create all PVA workloads at once
void radar_create_pva_workloads(RadarPipelineOperators& operators,
                                RadarPipelineTensorRequirements& tensorReqs, int32_t NofTx,
                                int32_t repeatFold);

// Function to submit all PVA workloads at once
void radar_submit_pva_workloads(RadarPipelineOperators& operators, RadarPipelineTensors& tensors,
                                RadarPipelinePVAWorkloadParams& pvaWorkloadParams);

#endif  // RADAR_PIPELINE_PVA_HPP
