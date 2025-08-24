/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RIVERMAX_ANO_APPLICATIONS_UTILS_H_
#define RIVERMAX_ANO_APPLICATIONS_UTILS_H_

#include <vector>
#include <string>
#include <unordered_map>

namespace holoscan::advanced_network {

struct StreamNetworkSettings
{
  std::string local_ip;
  std::vector<std::string> local_ips;
  std::string source_ip;
  std::vector<std::string> source_ips;
  std::string destination_ip;
  std::vector<std::string> destination_ips;
  uint16_t source_port;
  std::vector<uint16_t> destination_ports;
  uint16_t destination_port;
  uint32_t stream_id;
};

struct ThreadSettings
{
  std::vector<StreamNetworkSettings> stream_network_settings; 
  int thread_id;
};

}  // namespace holoscan::advanced_network

#endif  // RIVERMAX_ANO_APPLICATIONS_UTILS_H_
