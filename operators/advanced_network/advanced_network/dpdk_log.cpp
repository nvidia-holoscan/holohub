/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "advanced_network/dpdk_log.h"

namespace holoscan::advanced_network {

const std::unordered_map<uint32_t, std::tuple<std::string, std::string>>
    DpdkLogLevel::level_to_cmd_map = {{OFF, {"Disabled", "9"}},
                                      {RTE_LOG_EMERG, {"Emergency", "emergency"}},
                                      {RTE_LOG_ALERT, {"Alert", "alert"}},
                                      {RTE_LOG_CRIT, {"Critical", "critical"}},
                                      {RTE_LOG_ERR, {"Error", "error"}},
                                      {RTE_LOG_WARNING, {"Warning", "warning"}},
                                      {RTE_LOG_NOTICE, {"Notice", "notice"}},
                                      {RTE_LOG_INFO, {"Info", "info"}},
                                      {RTE_LOG_DEBUG, {"Debug", "debug"}}};

const std::unordered_map<LogLevel::Level, uint32_t>
    DpdkLogLevel::ano_to_dpdk_log_level_map = {
        {LogLevel::TRACE, RTE_LOG_DEBUG},
        {LogLevel::DEBUG, RTE_LOG_DEBUG},
        {LogLevel::INFO, RTE_LOG_INFO},
        {LogLevel::WARN, RTE_LOG_WARNING},
        {LogLevel::ERROR, RTE_LOG_ERR},
        {LogLevel::CRITICAL, RTE_LOG_CRIT},
        {LogLevel::OFF, OFF},
};

}  // namespace holoscan::advanced_network
