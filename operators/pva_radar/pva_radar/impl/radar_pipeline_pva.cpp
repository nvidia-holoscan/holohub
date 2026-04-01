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
 * This file is based on pva-solutions/pipelines/radar/radar_pipeline_pva.cpp
 * It has been modified from the original to use CPP style error handling. This is the original
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

#include "radar_pipeline_pva.hpp"

#include <ErrorCheckMacros.h>
#include <OpDOA.h>
#include <OpDopplerFFT.h>
#include <OpNci.h>
#include <OpPeakDetection.h>
#include <OpRangeFFT.h>
#include <PvaAllocator.h>

#include <cstdio>
#include <iterator>  // For std::size
#include <string>

// ============================================================================
// RadarPipelineOperators Implementation
// ============================================================================

// Function to initialize all operator handles to NULL
void RadarPipelineOperators::initRadarPipelineOperators() {
  rangeFFTOperatorHandle = NULL;
  dopplerFFTOperatorHandle = NULL;
  nciOperatorHandle = NULL;
  peakDetectionOperatorHandle = NULL;
  doaOperatorHandle = NULL;
}

// Function to cleanup all operator handles
void RadarPipelineOperators::cleanupRadarPipelineOperators() {
  // Destroy PVA operators
  if (rangeFFTOperatorHandle != NULL) {
    nvcvOperatorDestroy(rangeFFTOperatorHandle);
    rangeFFTOperatorHandle = NULL;
  }

  if (dopplerFFTOperatorHandle != NULL) {
    nvcvOperatorDestroy(dopplerFFTOperatorHandle);
    dopplerFFTOperatorHandle = NULL;
  }

  if (nciOperatorHandle != NULL) {
    nvcvOperatorDestroy(nciOperatorHandle);
    nciOperatorHandle = NULL;
  }

  if (peakDetectionOperatorHandle != NULL) {
    nvcvOperatorDestroy(peakDetectionOperatorHandle);
    peakDetectionOperatorHandle = NULL;
  }

  if (doaOperatorHandle != NULL) {
    nvcvOperatorDestroy(doaOperatorHandle);
    doaOperatorHandle = NULL;
  }
}

// ============================================================================
// RadarPipelinePVAWorkloadParams Implementation
// ============================================================================

// Function to initialize RadarPipelinePVAWorkloadParams
void RadarPipelinePVAWorkloadParams::initRadarPipelinePVAWorkloadParams() {
  // Create allocator
  NVCV_CHECK_ERROR(nvcvAllocatorConstructPva(&allocatorHandle));

  // Initialize CUPVA objects
  CUPVA_CHECK_ERROR(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD));
  CUPVA_CHECK_ERROR(CupvaFenceInit(&fence, sync));
  CUPVA_CHECK_ERROR(CupvaCmdRequestFencesInit(&rf, &fence, 1));
  CUPVA_CHECK_ERROR(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY));
}

// Function to tear down RadarPipelinePVAWorkloadParams
void RadarPipelinePVAWorkloadParams::cleanupRadarPipelinePVAWorkloadParams() {
  nvcvAllocatorDecRef(allocatorHandle, NULL);
  allocatorHandle = NULL;

  // Destroy CUPVA objects
  if (stream != NULL) {
    CupvaStreamDestroy(stream);
    stream = NULL;
  }

  if (sync != NULL) {
    CupvaSyncObjDestroy(sync);
    sync = NULL;
  }
}

// ============================================================================
// PVA Workload Functions
// ============================================================================

// Function to create all PVA workloads at once
void radar_create_pva_workloads(RadarPipelineOperators& operators,
                                RadarPipelineTensorRequirements& tensorReqs, int32_t NofTx,
                                int32_t repeatFold) {
  PVARadarNCIParams nciParams = {};
  nciParams.repeatFold = repeatFold;
  nciParams.noiseEstimationEnabled = true;

  PVARangeFFTParams rangeParams = {};
  rangeParams.windowType = PVA_BATCH_FFT_WINDOW_HANNING;

  PVADopplerFFTParams dopplerParams = {};
  dopplerParams.windowType = PVA_BATCH_FFT_WINDOW_HANNING;
  dopplerParams.transposeOutput = 1;

  // Create PVA RangeFFT operator
  NVCV_CHECK_ERROR(pvaRangeFFTCreate(&operators.rangeFFTOperatorHandle,
                                     &tensorReqs.inRangeFFTRequirements,
                                     &tensorReqs.winRangeFFTRequirements,
                                     &tensorReqs.outRangeFFTRequirements,
                                     &rangeParams));

  // Create PVA DopplerFFT operator
  NVCV_CHECK_ERROR(pvaDopplerFFTCreate(&operators.dopplerFFTOperatorHandle,
                                       &tensorReqs.inDopplerFFTRequirements,
                                       &tensorReqs.winDopplerFFTRequirements,
                                       &tensorReqs.outDopplerFFTRequirements,
                                       &dopplerParams));

  // Create PVA NCI operator
  NVCV_CHECK_ERROR(pvaNciCreate(&operators.nciOperatorHandle,
                                &tensorReqs.inNciTensorRequirements,
                                tensorReqs.outNciTensorReqs.data(),
                                &nciParams));

  // Create PVA Peak Detection operator
  NVCV_CHECK_ERROR(pvaPeakDetectionCreate(&operators.peakDetectionOperatorHandle,
                                          &tensorReqs.inNciTensorRequirements,
                                          &tensorReqs.outPeakSnapTensorRequirements,
                                          NofTx,
                                          repeatFold));

  // Create PVA DOA operator
  NVCV_CHECK_ERROR(pvaDOACreate(&operators.doaOperatorHandle,
                                tensorReqs.inDOATensorReqs.data(),
                                &tensorReqs.outDOATensorRequirements,
                                tensorReqs.inDOATensorReqs.size(),
                                &tensorReqs.GP));
}

// Function to submit all PVA workloads at once
void radar_submit_pva_workloads(RadarPipelineOperators& operators, RadarPipelineTensors& tensors,
                                RadarPipelinePVAWorkloadParams& pvaWorkloadParams) {
  // Set up command arrays for submission
  cupvaCmd_t* cmds[1];
  cupvaCmdStatus_t status[1];
  cmds[0] = &pvaWorkloadParams.rf;
  status[0] = NULL;
  bool waitSuccess = false;

  NVCVTensorHandle inNciTensorHandles[] = {tensors.inNciTensorHandle};
  NVCVTensorHandle outNciTensorHandles[] = {tensors.outNciRxTensorHandle,
                                            tensors.outNciFinalTensorHandle,
                                            tensors.outNoiseEstimateTensorHandle};
  NVCVTensorHandle inPeakDetectionTensorHandles[] = {tensors.inNciTensorHandle,
                                                     tensors.outNciRxTensorHandle,
                                                     tensors.outNciFinalTensorHandle,
                                                     tensors.outNoiseEstimateTensorHandle};
  NVCVTensorHandle outPeakDetectionTensorHandles[] = {tensors.outPeakCountTensorHandle,
                                                      tensors.outPeakIndexTensorHandle,
                                                      tensors.outPeakSnapTensorHandle};

  NVCVTensorHandle inDOATensorHandles[] = {tensors.outPeakCountTensorHandle,
                                           tensors.inDOACalibVectorTensorHandle,
                                           tensors.outPeakIndexTensorHandle,
                                           tensors.outPeakSnapTensorHandle,
                                           tensors.outNciFinalTensorHandle};

  // Submit PVA RangeFFT operator
  NVCV_CHECK_ERROR(pvaRangeFFTSubmit(operators.rangeFFTOperatorHandle,
                                     pvaWorkloadParams.stream,
                                     tensors.inRangeFFTTensorHandle,
                                     tensors.winRangeFFTTensorHandle,
                                     tensors.outRangeFFTTensorHandle));

  // Submit PVA DopplerFFT operator
  NVCV_CHECK_ERROR(pvaDopplerFFTSubmit(operators.dopplerFFTOperatorHandle,
                                       pvaWorkloadParams.stream,
                                       tensors.inDopplerFFTTensorHandle,
                                       tensors.winDopplerFFTTensorHandle,
                                       tensors.outDopplerFFTTensorHandle));

  // Submit PVA NCI operator
  NVCV_CHECK_ERROR(pvaNciSubmit(operators.nciOperatorHandle,
                                pvaWorkloadParams.stream,
                                inNciTensorHandles,
                                outNciTensorHandles));

  // Submit PVA Peak Detection operator
  NVCV_CHECK_ERROR(pvaPeakDetectionSubmit(operators.peakDetectionOperatorHandle,
                                          pvaWorkloadParams.stream,
                                          inPeakDetectionTensorHandles,
                                          outPeakDetectionTensorHandles,
                                          std::size(inPeakDetectionTensorHandles),
                                          std::size(outPeakDetectionTensorHandles)));

  // Submit PVA DOA operator
  NVCV_CHECK_ERROR(pvaDOASubmit(operators.doaOperatorHandle,
                                pvaWorkloadParams.stream,
                                inDOATensorHandles,
                                tensors.outDOATensorHandle,
                                std::size(inDOATensorHandles)));

  // wait for stream completion
  CUPVA_CHECK_ERROR(
      CupvaStreamSubmit(pvaWorkloadParams.stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1));
  CUPVA_CHECK_ERROR(CupvaFenceWait(&pvaWorkloadParams.fence, -1, &waitSuccess));
  if (!waitSuccess) {
    throw std::runtime_error("CupvaFenceWait reported that the command timed out!");
  }
}
