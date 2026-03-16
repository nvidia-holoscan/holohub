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

#include "raw_radar_cube_source.hpp"

#include <ErrorCheckMacros.h>
#include <PvaAllocator.h>
#include <cupva_host.h>
#include <nvcv/Tensor.h>
#include <nvcv/TensorShape.h>

#include <cmath>
#include <stdexcept>

#include <radar_file_parser.h>

namespace holoscan::ops {

void RawRadarCubeSourceOp::setup(OperatorSpec& spec) {
  spec.output<NVCVTensorHandle>("output");
  spec.param(m_directory, "directory");
  spec.param(m_basename, "basename");
  spec.param(m_numChirps, "numChirps");
  spec.param(m_numRx, "numRx");
  spec.param(m_numSamples, "numSamples");
}

namespace {

void LoadFile(std::string file_name, NVCVTensorHandle rawADCTensor) {
  NVCVTensorData tensorData;
  NVCV_CHECK_ERROR(nvcvTensorExportData(rawADCTensor, &tensorData));

  // parse radar data using the function provided by pva-solutions example code
  // returns a std::pair where second element is 3D vector of raw ADC samples.
  auto [status, rawAdcVector3D] = rsps_RawFileRead<int32_t>(file_name);

  // sanity check that the tensor matches the input data shape
  if (tensorData.shape[0] != rawAdcVector3D.size() ||
      tensorData.shape[1] != rawAdcVector3D[0].size() ||
      tensorData.shape[2] != rawAdcVector3D[0][0].size()) {
    HOLOSCAN_LOG_ERROR("rawADCTensor shape does not match the radar cube data read from {}!",
                       file_name);
    HOLOSCAN_LOG_ERROR("rawADCTensor shape: {}Hx{}Cx{}W, expected shape: {}Hx{}Cx{}W",
                       tensorData.shape[0],
                       tensorData.shape[1],
                       tensorData.shape[2],
                       rawAdcVector3D.size(),
                       rawAdcVector3D[0].size(),
                       rawAdcVector3D[0][0].size());
    throw std::runtime_error("failed to load radar cube data.");
  }

  // fill tensor from the raw ADC data
  uint8_t* tensorBytePtr = NULL;
  CUPVA_CHECK_ERROR(
      CupvaMemGetHostPointer((void**)&tensorBytePtr, tensorData.buffer.strided.basePtr));

  for (int32_t h = 0; h < tensorData.shape[0]; h++) {
    for (int32_t c = 0; c < tensorData.shape[1]; c++) {
      for (int32_t w = 0; w < tensorData.shape[2]; w++) {
        size_t offset = h * tensorData.buffer.strided.strides[0] +
                        c * tensorData.buffer.strided.strides[1] +
                        w * tensorData.buffer.strided.strides[2];
        *reinterpret_cast<int32_t*>(tensorBytePtr + offset) = rawAdcVector3D[h][c][w];
      }
    }
  }
}

}  // namespace

void RawRadarCubeSourceOp::start() {
  // Read data file
  m_files = readFilesWithPrefix(m_directory, m_basename);
  HOLOSCAN_LOG_INFO("Found {} radar data files", m_files.size());
  if (m_files.empty()) {
    throw std::runtime_error("No radar data found!");
  }

  // allocate ADC tensor handle
  NVCVAllocatorHandle allocatorHandle;
  NVCVTensorRequirements rawAdcDataTensorRequirements;
  NVCVDataType rawAdcDataTensorDataType = NVCV_DATA_TYPE_S32;
  int64_t rawAdcDataTensorShape[] = {m_numChirps, m_numRx, m_numSamples};
  NVCVTensorLayout tensorLayoutHCW;

  NVCV_CHECK_ERROR(nvcvAllocatorConstructPva(&allocatorHandle));
  NVCV_CHECK_ERROR(nvcvTensorLayoutMake("HCW", &tensorLayoutHCW));
  NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(3,
                                                 rawAdcDataTensorShape,
                                                 rawAdcDataTensorDataType,
                                                 tensorLayoutHCW,
                                                 0,
                                                 0,
                                                 &rawAdcDataTensorRequirements));

  NVCV_CHECK_ERROR(
      nvcvTensorConstruct(&rawAdcDataTensorRequirements, allocatorHandle, &m_rawADCTensor));

  nvcvAllocatorDecRef(allocatorHandle, nullptr);

  // pre-load first file so it will be ready on first call to compute()
  try {
    LoadFile(m_files[0], m_rawADCTensor);
  } catch (...) {
    HOLOSCAN_LOG_ERROR("Failed to first radar cube data file");
    nvcvTensorDecRef(m_rawADCTensor, nullptr);
    m_rawADCTensor = nullptr;
    throw;
  }
}

void RawRadarCubeSourceOp::compute(InputContext&, OutputContext& output, ExecutionContext&) {
  // advance to next file in a loop over all files
  int32_t nextIndex = (m_fileIndex + 1) % m_files.size();

  // only load new file if not already loaded (single-file source wastes no cycles)
  if (m_fileIndex != nextIndex) {
    LoadFile(m_files[nextIndex], m_rawADCTensor);
    m_fileIndex = nextIndex;
  }

  // post the NVCVTensorHandle to the output port
  output.emit(m_rawADCTensor, "output");
}

void RawRadarCubeSourceOp::stop() {
  // done with ADC tensor handle
  if (m_rawADCTensor) {
    nvcvTensorDecRef(m_rawADCTensor, nullptr);
    m_rawADCTensor = nullptr;
  }
}

}  // namespace holoscan::ops
