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

#ifndef STATS_PRINTER_H_
#define STATS_PRINTER_H_

#include <unordered_map>
#include <sstream>
#include <vector>

#include "adv_network_mgr.h"
#include "rmax_ipo_receiver_service.h"

namespace holoscan::ops {

class IpoRxStatsPrinter {
 public:
  static constexpr double GIGABYTE = 1073741824.0;
  static constexpr double MEGABYTE = 1048576.0;

  static void print_stream_stats(std::stringstream& ss, uint32_t stream_index,
                                 IPORXStatistics stream_stats,
                                 std::vector<IPOPathStatistics> stream_path_stats) {
    ss << "[stream_index " << std::setw(3) << stream_index << "]"
       << " Got " << std::setw(7) << stream_stats.rx_counter << " packets | ";

    if (stream_stats.received_bytes >= GIGABYTE) {
      ss << std::fixed << std::setprecision(2) << (stream_stats.received_bytes / GIGABYTE)
         << " GB |";
    } else if (stream_stats.received_bytes >= MEGABYTE) {
      ss << std::fixed << std::setprecision(2) << (stream_stats.received_bytes / MEGABYTE)
         << " MB |";
    } else {
      ss << stream_stats.received_bytes << " bytes |";
    }

    ss << " dropped: ";
    for (uint32_t s_index = 0; s_index < stream_path_stats.size(); ++s_index) {
      if (s_index > 0) { ss << ", "; }
      ss << stream_path_stats[s_index].rx_dropped + stream_stats.rx_dropped;
    }
    ss << " |"
       << " consumed: " << stream_stats.consumed_packets << " |"
       << " unconsumed: " << stream_stats.unconsumed_packets << " |"
       << " lost: " << stream_stats.rx_dropped << " |"
       << " exceed MD: " << stream_stats.rx_exceed_md << " |"
       << " bad RTP hdr: " << stream_stats.rx_corrupt_rtp_header << " | ";

    for (uint32_t s_index = 0; s_index < stream_path_stats.size(); ++s_index) {
      if (stream_stats.rx_counter > 0) {
        uint32_t number = static_cast<uint32_t>(
            floor(100 * static_cast<double>(stream_path_stats[s_index].rx_count) /
                  static_cast<double>(stream_stats.rx_counter)));
        ss << " " << std::setw(3) << number << "%";
      } else {
        ss << "   0%";
      }
    }
    ss << "\n";
  }

  /**
   * @brief Prints the statistics of the Rmax manager.
   */
  static void print_total_stats(
      std::stringstream& ss,
      std::unordered_map<uint32_t,
                         std::unique_ptr<ral::services::rmax_ipo_receiver::RmaxIPOReceiverService>>&
          rx_services) {
    uint32_t stream_id = 0;

    ss << "RIVERMAX ANO Statistics\n";
    ss << "====================\n";
    ss << "Total Statistics\n";
    ss << "----------------\n";
    for (const auto& entry : rx_services) {
      uint32_t key = entry.first;
      auto& rx_service = entry.second;
      auto [stream_stats, path_stats] = rx_service->get_streams_statistics();
      for (uint32_t i = 0; i < stream_stats.size(); ++i) {
        print_stream_stats(ss, stream_id++, stream_stats[i], path_stats[i]);
      }
    }
  }
};

};  // namespace holoscan::ops

#endif  // STATS_PRINTER_H
