/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_VISUALIZER_ICARDIO_HPP
#define HOLOSCAN_OPERATORS_VISUALIZER_ICARDIO_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"

#include "holoinfer.hpp"
#include "holoinfer_buffer.hpp"
#include "holoinfer_utils.hpp"

namespace HoloInfer = holoscan::inference;

namespace holoscan::ops {
/**
 * @brief Visualizer iCardio Operator class to generate data for visualization
 *
 * Class wraps a GXF Codelet(`nvidia::holoscan::multiai::VisualizerICardio`).
 */
class VisualizerICardioOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VisualizerICardioOp)

  VisualizerICardioOp() = default;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<std::vector<std::string>> in_tensor_names_;
  Parameter<std::vector<std::string>> out_tensor_names_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::vector<IOSpec*>> receivers_;
  Parameter<std::vector<IOSpec*>> transmitters_;
  Parameter<std::string> data_dir_;
  Parameter<bool> input_on_cuda_;

  // Internal state
  std::map<std::string, std::vector<int>> tensor_size_map_;

  const std::string module_{"Visualizer icardio Codelet"};
  const std::string pc_tensor_name_{"plax_chamber_processed"};

  const std::map<std::string, std::vector<int>> tensor_to_shape_ = {{"keypoints", {1, 5, 3}},
                                                                    {"keyarea_1", {1, 1, 4}},
                                                                    {"keyarea_2", {1, 1, 4}},
                                                                    {"keyarea_3", {1, 1, 4}},
                                                                    {"keyarea_4", {1, 1, 4}},
                                                                    {"keyarea_5", {1, 1, 4}},
                                                                    {"lines", {1, 5, 2}},
                                                                    {"logo", {320, 320, 4}}};
  const std::map<std::string, int> tensor_to_index_ = {
      {"keyarea_1", 1}, {"keyarea_2", 2}, {"keyarea_3", 3}, {"keyarea_4", 4}, {"keyarea_5", 5}};

  const std::string logo_file_ = "logo.txt";
  void* logo_image_ = nullptr;
  CudaStreamHandler cuda_stream_handler_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_VISUALIZER_ICARDIO_HPP */
