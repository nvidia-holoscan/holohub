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

#include "message_logger.hpp"

#include <any>
#include <cstdint>
#include <memory>
#include <string>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/domain/tensor_map.hpp"
#include "holoscan/core/resources/data_logger.hpp"

namespace holoscan {
namespace data_loggers {

void MessageLogger::setup(ComponentSpec& spec) {
  // add any additional parameters for the custom logger here

  // setup the parameters present on the base DataLoggerResource
  DataLoggerResource::setup(spec);
}

void MessageLogger::initialize() {
  // add any additional initialize code here

  // call parent initialize after adding missing serializer arg above
  DataLoggerResource::initialize();
}

bool MessageLogger::log_data(
    [[maybe_unused]] std::any data,
    [[maybe_unused]] const std::string& unique_id,
    [[maybe_unused]] int64_t acquisition_timestamp,
    [[maybe_unused]] std::shared_ptr<MetadataDictionary> metadata,
    [[maybe_unused]] IOSpec::IOType io_type) {
  auto current_timestamp = get_timestamp();
  HOLOSCAN_LOG_INFO("[timestamp: {}] log_data called for {} port with unique id {}",
                    current_timestamp,
                    io_type == IOSpec::IOType::kInput ? "input" : "output",
                    unique_id);
  return true;
}

bool MessageLogger::log_tensor_data(
    [[maybe_unused]] const std::shared_ptr<Tensor>& tensor,
    [[maybe_unused]] const std::string& unique_id,
    [[maybe_unused]] int64_t acquisition_timestamp,
    [[maybe_unused]] const std::shared_ptr<MetadataDictionary>& metadata,
    [[maybe_unused]] IOSpec::IOType io_type) {
  auto current_timestamp = get_timestamp();
  HOLOSCAN_LOG_INFO("[timestamp: {}] log_tensor_data called for {} port with unique id {}",
                    current_timestamp,
                    io_type == IOSpec::IOType::kInput ? "input" : "output",
                    unique_id);
  return true;
}

bool MessageLogger::log_tensormap_data(
    [[maybe_unused]] const TensorMap& tensor_map,
    [[maybe_unused]] const std::string& unique_id,
    [[maybe_unused]] int64_t acquisition_timestamp,
    [[maybe_unused]] const std::shared_ptr<MetadataDictionary>& metadata,
    [[maybe_unused]] IOSpec::IOType io_type) {
  auto current_timestamp = get_timestamp();
  HOLOSCAN_LOG_INFO("[timestamp: {}] log_tensormap_data called for {} port with unique id {}",
                    current_timestamp,
                    io_type == IOSpec::IOType::kInput ? "input" : "output",
                    unique_id);
  return true;
}

}  // namespace data_loggers
}  // namespace holoscan
