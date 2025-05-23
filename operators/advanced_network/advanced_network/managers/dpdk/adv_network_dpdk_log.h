/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <unordered_map>
#include <string>
#include <tuple>
#include <stdexcept>
#include "advanced_network/types.h"

namespace holoscan::advanced_network {

class DpdkLogLevel {
 public:
  enum Level {
    OFF = 0,
    EMERGENCY = 1,
    ALERT = 2,
    CRITICAL = 3,
    ERROR = 4,
    WARN = 5,
    NOTICE = 6,
    INFO = 7,
    DEBUG = 8,
  };

  static std::string to_description_string(Level level) {
    auto it = level_to_cmd_map.find(level);
    if (it != level_to_cmd_map.end()) { return std::get<0>(it->second); }
    throw std::logic_error(
        "Unrecognized log level, available options "
        "debug/info/notice/warn/error/critical/alert/emergency/off");
  }

  static std::string to_cmd_string(Level level) {
    auto it = level_to_cmd_map.find(level);
    if (it != level_to_cmd_map.end()) { return std::get<1>(it->second); }
    throw std::logic_error(
        "Unrecognized log level, available options "
        "debug/info/notice/warn/error/critical/alert/emergency/off");
  }

  static Level from_ano_log_level(LogLevel::Level ano_level) {
    auto it = ano_to_dpdk_log_level_map.find(ano_level);
    if (it != ano_to_dpdk_log_level_map.end()) { return it->second; }
    return OFF;
  }

 private:
  /**
   * A map of log level to a tuple of the description and command strings.
   */
  static const std::unordered_map<Level, std::tuple<std::string, std::string>> level_to_cmd_map;
  static const std::unordered_map<LogLevel::Level, Level> ano_to_dpdk_log_level_map;
};

/**
 * @class DpdkLogLevelCommandBuilder
 * @brief Concrete class for building DPDK log level commands.
 *
 * This class implements the ManagerLogLevelCommandBuilder interface to provide
 * specific command flag strings for managing DPDK log levels.
 */
class DpdkLogLevelCommandBuilder : public ManagerLogLevelCommandBuilder {
 public:
  /**
   * @brief Constructor for DpdkLogLevelCommandBuilder.
   *
   * @param ano_level The log level from AnoLogLevel to be converted to DPDK log level.
   */
  explicit DpdkLogLevelCommandBuilder(LogLevel::Level ano_level)
      : level_(DpdkLogLevel::from_ano_log_level(ano_level)) {}

  /**
   * @brief Get the command flag strings for DPDK log levels.
   *
   * This function returns the specific command flag strings required to set
   * the DPDK log levels.
   *
   * @return A vector of command flag strings.
   */
  std::vector<std::string> get_cmd_flags_strings() const override {
    return {"--log-level=" + DpdkLogLevel::to_cmd_string(DpdkLogLevel::Level::OFF),
            "--log-level=pmd.net.mlx5:" + DpdkLogLevel::to_cmd_string(level_)};
  }

 private:
  DpdkLogLevel::Level level_;  ///< The DPDK log level.
};

}  // namespace holoscan::advanced_network
