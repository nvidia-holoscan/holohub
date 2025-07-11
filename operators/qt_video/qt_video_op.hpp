/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef OPERATORS_QT_VIDEO_QT_VIDEO_OP
#define OPERATORS_QT_VIDEO_QT_VIDEO_OP

#include <gxf/multimedia/video.hpp>

#include <holoscan/core/conditions/gxf/boolean.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

// forward declarations
class QtHoloscanVideo;
typedef struct CUevent_st* cudaEvent_t;

namespace holoscan::ops {

/**
 * @brief This operator inputs VideoBuffer or Tensor and displays it with the QtHoloscanVideo
 * QtQuick item.
 *
 * Parameters
 *
 * - **`QtHoloscanVideo`**: Instance of QtHoloscanVideo to be used
 *      - type: `QtHoloscanVideo`

 */
class QtVideoOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(QtVideoOp)

  QtVideoOp();

  void initialize() override;
  void start() override;
  void stop() override;
  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<QtHoloscanVideo*> qt_holoscan_video_;

  CudaStreamHandler cuda_stream_handler_;
  cudaEvent_t cuda_event_ = nullptr;
};

}  // namespace holoscan::ops

#endif /* OPERATORS_QT_VIDEO_QT_VIDEO_OP */
