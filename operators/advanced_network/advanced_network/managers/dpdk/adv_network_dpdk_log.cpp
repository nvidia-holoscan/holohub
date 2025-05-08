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

/**
 * A map of log level to a tuple of the description and command strings.
 */

#include "adv_network_dpdk_log.h"

namespace holoscan::advanced_network {

const std::unordered_map<DpdkLogLevel::Level, std::tuple<std::string, std::string>>
    DpdkLogLevel::level_to_cmd_map = {{OFF, {"Disabled", "9"}},
                                      {EMERGENCY, {"Emergency", "emergency"}},
                                      {ALERT, {"Alert", "alert"}},
                                      {CRITICAL, {"Critical", "critical"}},
                                      {ERROR, {"Error", "error"}},
                                      {WARN, {"Warning", "warning"}},
                                      {NOTICE, {"Notice", "notice"}},
                                      {INFO, {"Info", "info"}},
                                      {DEBUG, {"Debug", "debug"}}};

const std::unordered_map<LogLevel::Level, DpdkLogLevel::Level>
    DpdkLogLevel::ano_to_dpdk_log_level_map = {
        {LogLevel::TRACE, DEBUG},
        {LogLevel::DEBUG, DEBUG},
        {LogLevel::INFO, INFO},
        {LogLevel::WARN, WARN},
        {LogLevel::ERROR, ERROR},
        {LogLevel::CRITICAL, CRITICAL},
        {LogLevel::OFF, OFF},
};

}  // namespace holoscan::advanced_network
