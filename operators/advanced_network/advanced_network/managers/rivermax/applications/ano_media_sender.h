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


#ifndef RIVERMAX_ANO_APPLICATIONS_MEDIA_SENDER_H_
#define RIVERMAX_ANO_APPLICATIONS_MEDIA_SENDER_H_

#include "utils.h"
#include "rdk/apps/rmax_base_app.h"
#include "rdk/apps/rmax_base_memory_strategy.h"
#include "rdk/services/utils/defs.h"
#include "rdk/services/utils/clock.h"
#include "rdk/services/utils/enum_utils.h"
#include "rdk/services/sdp/sdp_defs.h"
#include "rdk/services/media/media.h"

namespace holoscan::advanced_network {

using namespace rivermax::dev_kit::apps;
using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

/**
 * @brief: Configuration settings for Rivermax ANO Media Sender.
 */
struct ANOMediaSenderSettings : AppSettings
{
public:
  static constexpr uint32_t DEFAULT_NUM_OF_PACKETS_IN_CHUNK_FHD = 16;
  static constexpr uint32_t DEFAULT_NUM_OF_PACKETS_IN_CHUNK_UHD = 32;
  void init_default_values() override;

  std::vector<ThreadSettings> thread_settings;
};

/**
 * @brief: Validator for Rivermax ANO Media Sender settings.
 */
class ANOMediaSenderSettingsValidator : public ISettingsValidator<ANOMediaSenderSettings>
{
public:
     ReturnStatus validate(const std::shared_ptr<ANOMediaSenderSettings>& settings) const override;
};

/**
 * @brief: ANO Media Sender application.
 *
 * This is an example of usage application for Rivermax media TX API.
 */
class ANOMediaSenderApp : public RmaxBaseApp
{
private:
  /* Settings builder pointer */
  std::shared_ptr<ISettingsBuilder<ANOMediaSenderSettings>> m_settings_builder;
  /* Application settings pointer */
  std::shared_ptr<ANOMediaSenderSettings> m_media_sender_settings;
  /* Sender objects container */
  std::vector<std::shared_ptr<MediaSenderIONode>> m_senders;
  /* Network recv flows */
  std::vector<std::vector<TwoTupleFlow>> m_threads_streams;
  /* NIC device interface */
  rmx_device_iface m_device_interface;
  /* Number of paths per stream */
  size_t m_num_paths_per_stream = 1;
  /* Network send flows */
  std::vector<TwoTupleFlow> m_flows;
  /* map external stream ID to internal thread ID and stream ID */
  std::unordered_map<size_t, std::pair<size_t, size_t>> m_stream_id_map;
public:
  /**
   * @brief: ANOMediaSenderApp class constructor.
   *
   * @param [in] settings_builder: Settings builder pointer.
   */
  ANOMediaSenderApp(std::shared_ptr<ISettingsBuilder<ANOMediaSenderSettings>> settings_builder);
  virtual ~ANOMediaSenderApp() = default;
  ReturnStatus run() override;
  ReturnStatus initialize() override;
  /**
   * @brief: Sets the frame provider for the specified stream index.
   *
   * @param [in] stream_index: Stream index.
   * @param [in] frame_provider: Framer provider pointer.
   * @param [in] media_type: Media type.
   * @param [in] contains_payload: Flag indicating whether the frame provider contains payload.
   *
   * @return: Status of the operation.
   */
  ReturnStatus set_frame_provider(size_t stream_index, std::shared_ptr<IFrameProvider> frame_provider,
    MediaType media_type = MediaType::Video, bool contains_payload = true);

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

  /**
   * @brief: Returns application settings.
   *
   * If the application was initialized successfully, this method will set the
   * provided reference to the application settings and return success.
   * Otherwise, it will return a failure status.
   *
   * @param [out] settings: A reference to store the application settings.
   *
   * @return: Status of the operation.
   */
  ReturnStatus get_app_settings(const AppSettings*& settings) const override;
private:
  ReturnStatus initialize_app_settings() final;
  ReturnStatus post_load_settings() final;
  ReturnStatus initialize_memory_strategy() override;
  ReturnStatus set_rivermax_clock() override;
  ReturnStatus initialize_connection_parameters() final;
  /**
   * @brief: Initializes network send flows.
   *
   * This method is responsible to initialize the send flows will be
   * used in the application. Those flows will be distributed in
   * @ref ANOMediaSenderApp::distribute_work_for_threads
   * to the streams will be used in the application.
   * The application supports unicast and multicast UDPv4 send flows.
   */
  void configure_network_flows();
  /**
   * @brief: Distributes work for threads.
   *
   * This method is responsible to distribute work to threads, by
   * distributing number of streams per sender thread uniformly.
   * In future development, this can be extended to different
   * streams per thread distribution policies.
   */
  void distribute_work_for_threads();
  /**
   * @brief: Initializes sender threads.
   *
   * This method is responsible to initialize @ref MediaSenderIONode objects to work.
   * It will initiate objects with the relevant parameters.
   * The objects initialized in this method, will be the contexts to the std::thread objects
   * will run in @ref RmaxBaseApp::run_threads method.
   */
  void initialize_sender_threads();
  /**
   * @brief: Returns current time in nanoseconds.
   *
   * This method uses @ref get_rivermax_ptp_time_ns to return the current PTP time.
   * @note: PTP4l must be running on the system for time to be valid.
   *
   * @return: Current time in nanoseconds.
   */
  static uint64_t get_time_ns(void* context = nullptr);
};

} // namespace holoscan::advanced_network

#endif // RIVERMAX_ANO_APPLICATIONS_MEDIA_SENDER_H_
