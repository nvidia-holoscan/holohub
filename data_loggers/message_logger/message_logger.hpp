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

#ifndef DATA_LOGGERS_MESSAGE_LOGGER_MESSAGE_LOGGER_HPP
#define DATA_LOGGERS_MESSAGE_LOGGER_MESSAGE_LOGGER_HPP

#include <any>
#include <cstdint>
#include <memory>  // For std::shared_ptr in parameters
#include <string>
#include <vector>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/resources/data_logger.hpp"

namespace holoscan {
namespace data_loggers {

/**
 * @brief Class for logging a message indicating when an emit or receive call is made.
 */
class MessageLogger : public DataLoggerResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(MessageLogger, DataLoggerResource)
  MessageLogger() = default;

  void setup(ComponentSpec& spec) override;
  void initialize() override;
  bool log_data(std::any data, const std::string& unique_id, int64_t acquisition_timestamp = -1,
                std::shared_ptr<MetadataDictionary> metadata = nullptr,
                IOSpec::IOType io_type = IOSpec::IOType::kOutput) override;
  bool log_tensor_data(const std::shared_ptr<Tensor>& tensor, const std::string& unique_id,
                       int64_t acquisition_timestamp = -1,
                       const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                       IOSpec::IOType io_type = IOSpec::IOType::kOutput) override;
  bool log_tensormap_data(const TensorMap& tensor_map, const std::string& unique_id,
                          int64_t acquisition_timestamp = -1,
                          const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                          IOSpec::IOType io_type = IOSpec::IOType::kOutput) override;
};

}  // namespace data_loggers
}  // namespace holoscan

#endif /* DATA_LOGGERS_MESSAGE_LOGGER_MESSAGE_LOGGER_HPP */
