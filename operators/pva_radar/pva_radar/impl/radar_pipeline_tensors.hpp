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

/**
 * This file is based on pva-solutions/pipelines/radar/radar_pipeline_tensors.hpp
 * It has been modified to remove the duplicate "ref" tensors. that file contained this original
 * license text:
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

#ifndef RADAR_PIPELINE_TENSORS_HPP
#define RADAR_PIPELINE_TENSORS_HPP

#include <PvaAllocator.h>
#include <PvaOperatorTypes.h>
#include <RadarOperatorTypes.h>
#include <nvcv/Fwd.h>
#include <nvcv/alloc/Allocator.h>

#include <vector>

// ============================================================================
// RadarPipelineTensorRequirements - Holds tensor requirements without allocation
// ============================================================================

class RadarPipelineTensorRequirements {
 public:
  RadarPipelineTensorRequirements(const int32_t Ns, const int32_t NofRx, const int32_t Nr,
                                  const int32_t repeatFold, const int32_t NofTx) {
    createTensorRequirements(Ns, NofRx, Nr, repeatFold, NofTx);
  }

  void createTensorRequirements(const int32_t Ns, const int32_t NofRx, const int32_t Nr,
                                const int32_t repeatFold, const int32_t NofTx);

  RadarPipelineTensorRequirements(const RadarPipelineTensorRequirements&) = delete;
  RadarPipelineTensorRequirements& operator=(const RadarPipelineTensorRequirements&) = delete;
  RadarPipelineTensorRequirements(RadarPipelineTensorRequirements&&) = delete;
  RadarPipelineTensorRequirements& operator=(RadarPipelineTensorRequirements&&) = delete;

  // Tensor requirements
  NVCVTensorRequirements inRangeFFTRequirements;
  NVCVTensorRequirements outRangeFFTRequirements;
  NVCVTensorRequirements winRangeFFTRequirements;
  NVCVTensorRequirements inDopplerFFTRequirements;
  NVCVTensorRequirements winDopplerFFTRequirements;
  NVCVTensorRequirements outDopplerFFTRequirements;
  NVCVTensorRequirements outNciRxTensorRequirements;
  NVCVTensorRequirements outNciFinalTensorRequirements;
  NVCVTensorRequirements outNoiseEstimateTensorRequirements;
  NVCVTensorRequirements inNciTensorRequirements;
  NVCVTensorRequirements outPeakCountTensorRequirements;
  NVCVTensorRequirements outPeakIndexTensorRequirements;
  NVCVTensorRequirements outPeakSnapTensorRequirements;
  NVCVTensorRequirements inDOACalibVectorTensorRequirements;
  NVCVTensorRequirements outDOATensorRequirements;
  std::vector<const NVCVTensorRequirements*> outNciTensorReqs;
  std::vector<const NVCVTensorRequirements*> inDOATensorReqs;

  PVARadarGP GP;
};

// ============================================================================
// RadarPipelineTensors - Manages all tensor allocations
// ============================================================================

class RadarPipelineTensors {
 public:
  RadarPipelineTensors(NVCVAllocatorHandle allocatorHandle,
                       RadarPipelineTensorRequirements const& tensorReqs) {
    initRadarPipelineTensors();
    createRadarPipelineTensors(allocatorHandle, tensorReqs);
  }

  ~RadarPipelineTensors() { cleanupRadarPipelineTensors(); }

  RadarPipelineTensors(RadarPipelineTensors&&) = delete;
  RadarPipelineTensors& operator=(RadarPipelineTensors&&) = delete;
  RadarPipelineTensors(const RadarPipelineTensors&) = delete;
  RadarPipelineTensors& operator=(const RadarPipelineTensors&) = delete;

  void initRadarPipelineTensors();
  void cleanupRadarPipelineTensors();
  void createRadarPipelineTensors(NVCVAllocatorHandle allocatorHandle,
                                  RadarPipelineTensorRequirements const& tensorReqs);

  // Range FFT tensors
  NVCVTensorHandle inRangeFFTTensorHandle;
  NVCVTensorHandle winRangeFFTTensorHandle;
  NVCVTensorHandle outRangeFFTTensorHandle;

  // Doppler FFT tensors
  NVCVTensorHandle inDopplerFFTTensorHandle;  // aliased to outRangeFFTTensorHandle
  NVCVTensorHandle winDopplerFFTTensorHandle;
  NVCVTensorHandle outDopplerFFTTensorHandle;

  // NCI tensors
  NVCVTensorHandle inNciTensorHandle;  // aliased to outDopplerFFTTensorHandle
  NVCVTensorHandle outNciRxTensorHandle;
  NVCVTensorHandle outNciFinalTensorHandle;
  NVCVTensorHandle outNoiseEstimateTensorHandle;

  // Peak Detection tensors
  // NOTE: input tensors are from an array of NCI output tensors
  NVCVTensorHandle outPeakCountTensorHandle;
  NVCVTensorHandle outPeakIndexTensorHandle;
  NVCVTensorHandle outPeakSnapTensorHandle;

  // DOA tensors
  // NOTE: input tensors are from an array of Peak Detection output tensors
  NVCVTensorHandle inDOACalibVectorTensorHandle;
  NVCVTensorHandle outDOATensorHandle;
};

#endif  // RADAR_PIPELINE_TENSORS_HPP
