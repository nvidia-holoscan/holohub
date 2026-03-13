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
 * This file is based on pva-solutions/pipelines/radar/radar_pipeline_tensors.cpp
 * It has been modified to remove the duplicate "ref" tensors, and refactored to reduce code
 * duplication in tensor requirements setup. This is the original license text:
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

#include "radar_pipeline_tensors.hpp"

#include "doa_target_processing_ref.h"
#include "doppler_fft_ref.h"
#include "range_fft_ref.h"

#include <ErrorCheckMacros.h>
#include <PvaAllocator.h>
#include <nvcv/Tensor.h>
#include <nvcv/DataType.hpp>
#include <string>

// ============================================================================
// RadarPipelineTensorRequirements Implementation
// ============================================================================

// Function to create tensor requirements without allocating actual tensors
void RadarPipelineTensorRequirements::createTensorRequirements(const int32_t Ns,
                                                               const int32_t NofRx,
                                                               const int32_t Nr,
                                                               const int32_t repeatFold,
                                                               const int32_t NofTx) {
  int32_t err = 0;

  // Calculate derived constants
  const int32_t NbNci = 224;

  // Define data types
  NVCVDataType inRangeFFTDataType = NVCV_DATA_TYPE_S32;
  NVCVDataType winRangeFFTDataType = NVCV_DATA_TYPE_S32;
  NVCVDataType outRangeFFTDataType = NVCV_DATA_TYPE_2S32;
  NVCVDataType inDopplerFFTDataType = NVCV_DATA_TYPE_2S32;
  NVCVDataType winDopplerFFTDataType = NVCV_DATA_TYPE_S32;
  NVCVDataType outDopplerFFTDataType = NVCV_DATA_TYPE_2S32;
  NVCVDataType nciInputDataType = NVCV_DATA_TYPE_2S32;
  NVCVDataType nciOutputDataType = NVCV_DATA_TYPE_U32;
  NVCVDataType peakIndexDataType = NVCV_DATA_TYPE_U32;
  NVCVDataType peakSnapDataType = NVCV_DATA_TYPE_2S32;
  NVCVDataType inDOACalibVectorTensorDataType = NVCV_DATA_TYPE_2S32;
  NVCVDataType outDOATensorDataType = NVCV_DATA_TYPE_F32;

  // Define tensor ranks
  const int32_t tensorRank = 3;
  const int32_t winRangeFFTTensorRank = 1;
  const int32_t winDopplerFFTTensorRank = 1;
  const int32_t doaTensorRank = 2;

  // Define shapes
  int64_t inRangeFFTShape[] = {Ns, NofRx, Nr};
  int64_t winRangeFFTShape[] = {Ns};
  int64_t outRangeFFTShape[] = {Nr, NofRx, NbNci};
  int64_t inDopplerFFTShape[] = {Nr, NofRx, NbNci};
  int64_t winDopplerFFTShape[] = {Nr};
  int64_t outDopplerFFTShape[] = {NbNci, NofRx, Nr};
  int64_t nciInputShape[] = {NbNci, NofRx, Nr};
  int64_t outNciRxTensorShape[] = {NbNci, Nr};
  int64_t outNciFinalTensorShape[] = {NbNci, Nr / repeatFold};
  int64_t outNoiseEstimateTensorShape[] = {NbNci};
  int64_t outPeakCountTensorShape[] = {1};
  int64_t outPeakIndexTensorShape[] = {PVA_RADAR_PEAKDET_NUM_PEAK_INDICES,
                                       PVA_RADAR_MAX_TARGET_COUNT};
  int64_t outPeakSnapTensorShape[] = {PVA_RADAR_MAX_TARGET_COUNT, 16};
  int64_t inDOACalibVectorTensorShape[] = {1, 16};
  int64_t outDOATensorShape[] = {PVA_RADAR_NUM_TARGET_DETECTION_PROPERTIES,
                                 PVA_RADAR_MAX_TARGET_COUNT};

  // Declare tensor layout variables
  NVCVTensorLayout tensorLayout;
  NVCVTensorLayout winRangeFFTLayout;
  NVCVTensorLayout winDopplerFFTLayout;
  NVCVTensorLayout outNciRxTensorLayout;
  NVCVTensorLayout outNciFinalTensorLayout;
  NVCVTensorLayout outNoiseEstimateTensorLayout;
  NVCVTensorLayout nciInputTensorLayout;
  NVCVTensorLayout outPeakCountTensorLayout;
  NVCVTensorLayout outPeakIndexTensorLayout;
  NVCVTensorLayout outPeakSnapTensorLayout;
  NVCVTensorLayout inDOACalibVectorTensorLayout;
  NVCVTensorLayout outDOATensorLayout;

  // Create tensor layouts
  NVCV_CHECK_ERROR(nvcvTensorLayoutMake("HCW", &tensorLayout));
  NVCV_CHECK_ERROR(nvcvTensorLayoutMake("W", &winRangeFFTLayout));
  NVCV_CHECK_ERROR(nvcvTensorLayoutMake("W", &winDopplerFFTLayout));
  NVCV_CHECK_ERROR(nvcvTensorLayoutMake("HW", &outNciRxTensorLayout));
  NVCV_CHECK_ERROR(nvcvTensorLayoutMake("HW", &outNciFinalTensorLayout));
  NVCV_CHECK_ERROR(nvcvTensorLayoutMake("W", &outNoiseEstimateTensorLayout));
  NVCV_CHECK_ERROR(nvcvTensorLayoutMake("HCW", &nciInputTensorLayout));
  NVCV_CHECK_ERROR(nvcvTensorLayoutMake("W", &outPeakCountTensorLayout));
  NVCV_CHECK_ERROR(nvcvTensorLayoutMake("HW", &outPeakIndexTensorLayout));
  NVCV_CHECK_ERROR(nvcvTensorLayoutMake("HW", &outPeakSnapTensorLayout));
  NVCV_CHECK_ERROR(nvcvTensorLayoutMake("HW", &inDOACalibVectorTensorLayout));
  NVCV_CHECK_ERROR(nvcvTensorLayoutMake("HW", &outDOATensorLayout));

  // Calculate tensor requirements (no actual allocation)
  NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(tensorRank,
                                                 inRangeFFTShape,
                                                 inRangeFFTDataType,
                                                 tensorLayout,
                                                 0,
                                                 0,
                                                 &inRangeFFTRequirements));
  NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(tensorRank,
                                                 outRangeFFTShape,
                                                 outRangeFFTDataType,
                                                 tensorLayout,
                                                 0,
                                                 0,
                                                 &outRangeFFTRequirements));
  NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(winRangeFFTTensorRank,
                                                 winRangeFFTShape,
                                                 winRangeFFTDataType,
                                                 winRangeFFTLayout,
                                                 0,
                                                 0,
                                                 &winRangeFFTRequirements));
  NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(winRangeFFTTensorRank,
                                                 winRangeFFTShape,
                                                 winRangeFFTDataType,
                                                 winRangeFFTLayout,
                                                 0,
                                                 0,
                                                 &winRangeFFTRequirements));
  NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(tensorRank,
                                                 inDopplerFFTShape,
                                                 inDopplerFFTDataType,
                                                 tensorLayout,
                                                 0,
                                                 0,
                                                 &inDopplerFFTRequirements));
  NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(winDopplerFFTTensorRank,
                                                 winDopplerFFTShape,
                                                 winDopplerFFTDataType,
                                                 winDopplerFFTLayout,
                                                 0,
                                                 0,
                                                 &winDopplerFFTRequirements));
  NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(tensorRank,
                                                 outDopplerFFTShape,
                                                 outDopplerFFTDataType,
                                                 tensorLayout,
                                                 0,
                                                 0,
                                                 &outDopplerFFTRequirements));
  NVCV_CHECK_ERROR(
      nvcvTensorCalcRequirementsPva(sizeof(outNciRxTensorShape) / sizeof(outNciRxTensorShape[0]),
                                    outNciRxTensorShape,
                                    nciOutputDataType,
                                    outNciRxTensorLayout,
                                    0,
                                    0,
                                    &outNciRxTensorRequirements));
  NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(
      sizeof(outNciFinalTensorShape) / sizeof(outNciFinalTensorShape[0]),
      outNciFinalTensorShape,
      nciOutputDataType,
      outNciFinalTensorLayout,
      0,
      0,
      &outNciFinalTensorRequirements));
  NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(
      sizeof(outNoiseEstimateTensorShape) / sizeof(outNoiseEstimateTensorShape[0]),
      outNoiseEstimateTensorShape,
      nciOutputDataType,
      outNoiseEstimateTensorLayout,
      0,
      0,
      &outNoiseEstimateTensorRequirements));
  NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(tensorRank,
                                                 nciInputShape,
                                                 nciInputDataType,
                                                 nciInputTensorLayout,
                                                 0,
                                                 0,
                                                 &inNciTensorRequirements));
  NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(
      sizeof(outPeakCountTensorShape) / sizeof(outPeakCountTensorShape[0]),
      outPeakCountTensorShape,
      NVCV_DATA_TYPE_U32,
      outPeakCountTensorLayout,
      0,
      0,
      &outPeakCountTensorRequirements));
  NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(
      sizeof(outPeakIndexTensorShape) / sizeof(outPeakIndexTensorShape[0]),
      outPeakIndexTensorShape,
      peakIndexDataType,
      outPeakIndexTensorLayout,
      0,
      0,
      &outPeakIndexTensorRequirements));
  NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(
      sizeof(outPeakSnapTensorShape) / sizeof(outPeakSnapTensorShape[0]),
      outPeakSnapTensorShape,
      peakSnapDataType,
      outPeakSnapTensorLayout,
      0,
      0,
      &outPeakSnapTensorRequirements));
  NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(doaTensorRank,
                                                 inDOACalibVectorTensorShape,
                                                 inDOACalibVectorTensorDataType,
                                                 inDOACalibVectorTensorLayout,
                                                 0,
                                                 0,
                                                 &inDOACalibVectorTensorRequirements));
  NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(doaTensorRank,
                                                 outDOATensorShape,
                                                 outDOATensorDataType,
                                                 outDOATensorLayout,
                                                 0,
                                                 0,
                                                 &outDOATensorRequirements));

  // Set up NCI output tensor requirements pointers
  outNciTensorReqs = {&outNciRxTensorRequirements,
                      &outNciFinalTensorRequirements,
                      &outNoiseEstimateTensorRequirements};

  // Set up DOA input tensor requirements pointers
  inDOATensorReqs = {&outPeakCountTensorRequirements,
                     &inDOACalibVectorTensorRequirements,
                     &outPeakIndexTensorRequirements,
                     &outPeakSnapTensorRequirements,
                     &outNciFinalTensorRequirements};

  // Initialize GP parameters used by DOA operator creation
  populateGP(&GP, 30, 20);
}

// ============================================================================
// RadarPipelineTensors Implementation
// ============================================================================

// Function to initialize all tensor handles to NULL
void RadarPipelineTensors::initRadarPipelineTensors() {
  // Initialize tensor handles to NULL
  inRangeFFTTensorHandle = NULL;
  winRangeFFTTensorHandle = NULL;
  outRangeFFTTensorHandle = NULL;
  inDopplerFFTTensorHandle = NULL;
  winDopplerFFTTensorHandle = NULL;
  outDopplerFFTTensorHandle = NULL;
  inNciTensorHandle = NULL;
  outNciRxTensorHandle = NULL;
  outNciFinalTensorHandle = NULL;
  outNoiseEstimateTensorHandle = NULL;
  outPeakCountTensorHandle = NULL;
  outPeakIndexTensorHandle = NULL;
  outPeakSnapTensorHandle = NULL;
  inDOACalibVectorTensorHandle = NULL;
  outDOATensorHandle = NULL;
}

// Function to cleanup all allocated resources
void RadarPipelineTensors::cleanupRadarPipelineTensors() {
  /// NOTE: we don't need to decref inRangeFFTTensorHandle because it is received from upstream
  /// operator as the input "rawADCIn" tensor and is owned by that operator.
  //   nvcvTensorDecRef(inRangeFFTTensorHandle, NULL);
  nvcvTensorDecRef(winRangeFFTTensorHandle, NULL);
  nvcvTensorDecRef(outRangeFFTTensorHandle, NULL);
  nvcvTensorDecRef(winDopplerFFTTensorHandle, NULL);
  nvcvTensorDecRef(outDopplerFFTTensorHandle, NULL);
  nvcvTensorDecRef(outNciRxTensorHandle, NULL);
  nvcvTensorDecRef(outNciFinalTensorHandle, NULL);
  nvcvTensorDecRef(outNoiseEstimateTensorHandle, NULL);
  nvcvTensorDecRef(outPeakCountTensorHandle, NULL);
  nvcvTensorDecRef(outPeakIndexTensorHandle, NULL);
  nvcvTensorDecRef(outPeakSnapTensorHandle, NULL);
  nvcvTensorDecRef(inDOACalibVectorTensorHandle, NULL);
  nvcvTensorDecRef(outDOATensorHandle, NULL);
}

// Function to create all tensors and calculate requirements
void RadarPipelineTensors::createRadarPipelineTensors(
    NVCVAllocatorHandle allocatorHandle, RadarPipelineTensorRequirements const& tensorReqs) {
  // Range FFT tensors
  /// NOTE: we don't need inRangeFFTTensorHandle because it is received from upstream operator as
  /// the input "rawADCIn" tensor.
  NVCV_CHECK_ERROR(nvcvTensorConstruct(
      &tensorReqs.outRangeFFTRequirements, allocatorHandle, &outRangeFFTTensorHandle));

  NVCV_CHECK_ERROR(nvcvTensorConstruct(
      &tensorReqs.winRangeFFTRequirements, allocatorHandle, &winRangeFFTTensorHandle));

  // Doppler FFT tensors
  NVCV_CHECK_ERROR(nvcvTensorConstruct(
      &tensorReqs.winDopplerFFTRequirements, allocatorHandle, &winDopplerFFTTensorHandle));
  NVCV_CHECK_ERROR(nvcvTensorConstruct(
      &tensorReqs.outDopplerFFTRequirements, allocatorHandle, &outDopplerFFTTensorHandle));

  // NCI tensors
  NVCV_CHECK_ERROR(nvcvTensorConstruct(
      &tensorReqs.outNciRxTensorRequirements, allocatorHandle, &outNciRxTensorHandle));

  NVCV_CHECK_ERROR(nvcvTensorConstruct(
      &tensorReqs.outNciFinalTensorRequirements, allocatorHandle, &outNciFinalTensorHandle));

  NVCV_CHECK_ERROR(nvcvTensorConstruct(&tensorReqs.outNoiseEstimateTensorRequirements,
                                       allocatorHandle,
                                       &outNoiseEstimateTensorHandle));

  // Peak Detection tensors
  NVCV_CHECK_ERROR(nvcvTensorConstruct(
      &tensorReqs.outPeakCountTensorRequirements, allocatorHandle, &outPeakCountTensorHandle));

  NVCV_CHECK_ERROR(nvcvTensorConstruct(
      &tensorReqs.outPeakIndexTensorRequirements, allocatorHandle, &outPeakIndexTensorHandle));

  NVCV_CHECK_ERROR(nvcvTensorConstruct(
      &tensorReqs.outPeakSnapTensorRequirements, allocatorHandle, &outPeakSnapTensorHandle));

  // DOA tensors
  NVCV_CHECK_ERROR(nvcvTensorConstruct(&tensorReqs.inDOACalibVectorTensorRequirements,
                                       allocatorHandle,
                                       &inDOACalibVectorTensorHandle));

  NVCV_CHECK_ERROR(nvcvTensorConstruct(
      &tensorReqs.outDOATensorRequirements, allocatorHandle, &outDOATensorHandle));

  // Set up tensor handle aliases for data passed between stages of the pipeline
  inDopplerFFTTensorHandle = outRangeFFTTensorHandle;
  inNciTensorHandle = outDopplerFFTTensorHandle;

  // Set up the FFT windows
  generateRangeFFTWindow(winRangeFFTTensorHandle, PVA_BATCH_FFT_WINDOW_HANNING);
  generateDopplerFFTWindow(winDopplerFFTTensorHandle, PVA_BATCH_FFT_WINDOW_HANNING);

  // Load hard-coded calibration vector to tensor
  int32_t calibVector[PVA_RADAR_NUM_TOTAL_ANTENNA_ELEMENTS * 2];
  populateCalibVector(calibVector, 30);
  loadCalibVectorToTensorFixedPoint(
      inDOACalibVectorTensorHandle, calibVector, sizeof(calibVector) / sizeof(calibVector[0]));
}
