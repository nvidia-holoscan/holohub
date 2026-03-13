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

#ifndef RAW_RADAR_CUBE_SOURCE_HPP
#define RAW_RADAR_CUBE_SOURCE_HPP

#include "holoscan/holoscan.hpp"

#include <nvcv/Tensor.h>
#include <cstdint>
#include <string>
#include <vector>

namespace holoscan::ops {

// Source operator to read RAW RADAR Cube data (ADC samples) from files. Each compute() call reads
// the next file in the alphanumeric sequence of <basename>*.bin files from the given <directory>
// and emits an NVCVTensorHandle containing the ADC data to the output port. The sequence loops
// indefinitely.
class RawRadarCubeSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RawRadarCubeSourceOp);

  void setup(OperatorSpec& spec) override;

  void start() override;

  void compute(InputContext&, OutputContext& output, ExecutionContext&) override;

  void stop() override;

 private:
  Parameter<std::string> m_directory;
  Parameter<std::string> m_basename;
  Parameter<int32_t> m_numChirps;
  Parameter<int32_t> m_numRx;
  Parameter<int32_t> m_numSamples;

  int32_t m_fileIndex = -1;
  std::vector<std::string> m_files;
  NVCVTensorHandle m_rawADCTensor = nullptr;
};

}  // namespace holoscan::ops

#endif  // RAW_RADAR_CUBE_SOURCE_HPP
