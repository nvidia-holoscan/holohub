/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef OPERATORS_UNDISTORT_RECTIFY
#define OPERATORS_UNDISTORT_RECTIFY

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

class UndistortRectifyOp : public Operator {
  static void stereoRecitfy();

 public:
  class RectificationMap {
   public:
    RectificationMap() {}
    RectificationMap(float* M, float* d, float* R, float* P, int width, int height);
    ~RectificationMap();
    void setParameters(float* M, float* d, float* R, float* P, int width, int height);
    float* mapx_ = NULL;
    float* mapy_ = NULL;
    int width_ = 0;
    int height_ = 0;

   private:
  };

  static void stereoRectify(float* M1, float* d1, float* M2, float* d2, float* R, float* t,
                            int width, int height, float* R1, float* R2, float* P1, float* P2,
                            float* Q, int* roi);

  HOLOSCAN_OPERATOR_FORWARD_ARGS(UndistortRectifyOp);
  UndistortRectifyOp() = default;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override;
  void setRectificationMap(std::shared_ptr<RectificationMap> rectification_map) {
    rectification_map_ = rectification_map;
  }

 private:
  std::shared_ptr<RectificationMap> rectification_map_;
  static std::vector<std::pair<float, float>> distortPoints(
      std::vector<std::pair<float, float>> pts_in, float* M, float* d);
  static std::vector<std::pair<float, float>> undistortPoints(
      std::vector<std::pair<float, float>> pts_in, float* M, float* d, float tol = 1e-3,
      int max_it = 1e2);
  static std::vector<std::pair<float, float>> originalToRectified(
      std::vector<std::pair<float, float>> pts_in, float* M, float* d, float* R, float f);
};
}  // namespace holoscan::ops
#endif
