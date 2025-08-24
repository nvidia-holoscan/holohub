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

#ifndef RIVERMAX_ANO_APPLICATIONS_RTP_RECEIVER_H_
#define RIVERMAX_ANO_APPLICATIONS_RTP_RECEIVER_H_

#include <chrono>

#include "utils.h"
#include "rdk/apps/rmax_receiver_base.h"
#include "rdk/io_node/receivers/rtp_receiver_io_node.h"
#include "rdk/rivermax_dev_kit.h"

namespace holoscan::advanced_network {

using namespace rivermax::dev_kit::apps;
using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

struct ANORTPReceiverSettings : AppSettings
{
public:
  static constexpr uint32_t DEFAULT_NUM_OF_PACKETS_IN_CHUNK = 262144;
  void init_default_values() override;

  bool is_extended_sequence_number;
  std::vector<ThreadSettings> thread_settings;
};

/**
* @brief: Validator for Rivermax RTP Receiver settings.
*/
class ANORTPReceiverSettingsValidator : public ISettingsValidator<ANORTPReceiverSettings>
{
public:
  ReturnStatus validate(const std::shared_ptr<ANORTPReceiverSettings>& settings) const override;
};


class ANORTPReceiverApp : public RmaxReceiverBaseApp
{
private:
    /* Settings builder pointer */
    std::shared_ptr<ISettingsBuilder<ANORTPReceiverSettings>> m_settings_builder;
    /* Application settings pointer */
    std::shared_ptr<ANORTPReceiverSettings> m_rtp_receiver_settings;
    /* Network recv flows */
    std::vector<std::vector<ReceiveFlow>> m_threads_streams;
    /* list of devices IPs */
    std::vector<std::string> m_devices_ips;
    /* map external stream ID to internal thread ID and stream ID */
    std::unordered_map<size_t, std::pair<size_t, size_t>> m_stream_id_map;
public:
  /**
   * @brief: ANORTPReceiverApp class constructor.
   *
   * @param [in] settings_builder: Settings builder pointer.
   */
  ANORTPReceiverApp(std::shared_ptr<ISettingsBuilder<ANORTPReceiverSettings>> settings_builder);
  /**
   * @brief: ANORTPReceiverApp class destructor.
   */
  virtual ~ANORTPReceiverApp() = default;
  /**
   * @brief: Get RTP streams total statistics.
   *
   * This function hides the base class function and provides default template arguments.
   *
   * @return: A vector of @ref RTPRXStatistics.
   */
  std::vector<RXStatistics> get_streams_total_statistics() const {
    return RmaxReceiverBaseApp::get_streams_total_statistics<RXStatistics, AppRTPReceiveStream>();
  }

  /**
   * @brief: Find internal thread and stream index for a given external stream index.
   * 
   * @param [in] external_stream_index: The external stream index.
   * @param [out] thread_index: The internal thread index.
   * @param [out] internal_stream_index: The internal stream index.
   *
   * @return: Status of the operation. 
  */
  ReturnStatus find_internal_stream_index(size_t external_stream_index, size_t& thread_index, size_t& internal_stream_index) override;

private:
  ReturnStatus initialize_app_settings() final;
  ReturnStatus initialize_connection_parameters() final;
  void configure_network_flows() final;
  void initialize_receive_io_nodes() final;
  void run_receiver_threads() final;
  void distribute_work_for_threads();
};

}  // namespace holoscan::advanced_network

#endif  // RIVERMAX_ANO_APPLICATIONS_RTP_RECEIVER_H_
