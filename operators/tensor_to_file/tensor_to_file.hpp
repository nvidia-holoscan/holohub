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

#ifndef NV_VIDEO_WRITER_NV_VIDEO_WRITER_HPP
#define NV_VIDEO_WRITER_NV_VIDEO_WRITER_HPP

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

namespace holoscan::ops {

/**
 * @brief Operator to stream tensor data to a file
 *
 * This operator streams input tensor data to a file.
 * The file is opened at init. Tensor binary data are appended in the
 * same order in which messages are received.
 */
class TensorToFileOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TensorToFileOp)

  TensorToFileOp() = default;
  ~TensorToFileOp();

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  Parameter<std::string> tensor_name_;
  Parameter<std::string> output_file_;
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<bool> verbose_;
  Parameter<size_t> buffer_size_;

  // File output stream for writing encoded video data
  std::ofstream output_stream_;
  std::filesystem::path file_path_;

  // Statistics
  size_t frame_count_ = 0;
  size_t total_bytes_written_ = 0;

  // Performance tracking
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point last_log_time_;
  size_t frames_since_last_log_ = 0;

  // Constants
  static constexpr size_t DEFAULT_BUFFER_SIZE = 1024 * 1024;  // 1MB
  static constexpr size_t LOG_INTERVAL_FRAMES = 100;  // Log every 100 frames in verbose mode

  void validateOutputFile(const std::string& filepath);
  void logPerformanceStats();
};

}  // namespace holoscan::ops

#endif /* NV_VIDEO_WRITER_NV_VIDEO_WRITER_HPP */
