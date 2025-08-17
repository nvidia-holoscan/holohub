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

#include "ano_media_sender.h"

namespace holoscan::advanced_network {

void ANOMediaSenderSettings::init_default_values()
{
  AppSettings::init_default_values();
  media.frames_fields_in_mem_block = 1;
  media.resolution = { FHD_WIDTH, FHD_HEIGHT };
  num_of_packets_in_chunk = ANOMediaSenderSettings::DEFAULT_NUM_OF_PACKETS_IN_CHUNK_FHD;
}

ReturnStatus ANOMediaSenderSettingsValidator::validate(const std::shared_ptr<ANOMediaSenderSettings>& settings) const
{
  if (settings->thread_settings.empty()) {
    std::cerr << "Must be at least one thread" << std::endl;
    return ReturnStatus::failure;
  }
  ReturnStatus rc;
  for (const auto& thread : settings->thread_settings) {
    if (thread.stream_network_settings.empty()) {
      std::cerr << "Must be at least one stream" << std::endl;
      return ReturnStatus::failure;
    }
    for (const auto& stream : thread.stream_network_settings) {
      rc = ValidatorUtils::validate_ip4_address(stream.local_ip);
      if (rc != ReturnStatus::success) {
        return rc;
      }
      rc = ValidatorUtils::validate_ip4_address(stream.destination_ip);
      if (rc != ReturnStatus::success) {
        return rc;
      }
      rc = ValidatorUtils::validate_ip4_port(stream.destination_port);
      if (rc != ReturnStatus::success) {
        return rc;
      }
    }
  }

  if (settings->register_memory && !settings->app_memory_alloc) {
    std::cerr << "Register memory option is supported only with application memory allocation" << std::endl;
    return ReturnStatus::failure;
  }
  return ReturnStatus::success;
}

ANOMediaSenderApp::ANOMediaSenderApp(std::shared_ptr<ISettingsBuilder<ANOMediaSenderSettings>> settings_builder) :
  RmaxBaseApp(),
  m_settings_builder(std::move(settings_builder)),
  m_device_interface{}
{
}

ReturnStatus ANOMediaSenderApp::post_load_settings()
{
  uint32_t default_packets_in_chunk;

  if (m_app_settings->media.resolution == Resolution(UHD_WIDTH, UHD_HEIGHT) ||
    m_app_settings->media.resolution == Resolution(UHD_HEIGHT, UHD_WIDTH)) {
    default_packets_in_chunk = ANOMediaSenderSettings::DEFAULT_NUM_OF_PACKETS_IN_CHUNK_UHD;
  } else {
    default_packets_in_chunk = ANOMediaSenderSettings::DEFAULT_NUM_OF_PACKETS_IN_CHUNK_FHD;
  }

  if (m_app_settings->num_of_packets_in_chunk != default_packets_in_chunk) {
    m_app_settings->num_of_packets_in_chunk_specified = true;
  }
  auto rc = initialize_media_settings(*m_app_settings);
  if (rc != ReturnStatus::success) {
    std::cerr << "Failed to initialize media settings" << std::endl;
  }
  return rc;
}

ReturnStatus ANOMediaSenderApp::initialize_app_settings()
{
  if (m_settings_builder == nullptr) {
    std::cerr << "Settings builder is not initialized" << std::endl;
    return ReturnStatus::failure;
  }
  m_media_sender_settings = std::make_shared<ANOMediaSenderSettings>();
  ReturnStatus rc = m_settings_builder->build(m_media_sender_settings);
  if (rc == ReturnStatus::success) {
    m_app_settings = m_media_sender_settings;
    std::cout << "Successfully initialized ANO Media Sender application settings" << std::endl;
    return ReturnStatus::success;
  }
  if (rc != ReturnStatus::success_cli_help) {
    std::cerr << "Failed to build settings" << std::endl;
  }
  m_obj_init_status = rc;
  return rc;
}

ReturnStatus ANOMediaSenderApp::initialize()
{
  ReturnStatus rc  = RmaxBaseApp::initialize();

  if (rc != ReturnStatus::obj_init_success) {
    return m_obj_init_status;
  }

  try {
    distribute_work_for_threads();
    configure_network_flows();
    initialize_sender_threads();
    rc = configure_memory_layout();
    if (rc == ReturnStatus::failure) {
      std::cerr << "Failed to configure memory layout" << std::endl;
      return rc;
    }
  }
  catch (const std::exception & error) {
    std::cerr << error.what() << std::endl;
    return ReturnStatus::failure;
  }

  m_obj_init_status = ReturnStatus::obj_init_success;
  return m_obj_init_status;
}

ReturnStatus ANOMediaSenderApp::initialize_connection_parameters()
{
  in_addr device_address;
  if (inet_pton(AF_INET, m_app_settings->local_ip.c_str(), &device_address) != 1) {
    std::cerr << "Failed to parse address of device " << m_app_settings->local_ip << std::endl;
    return ReturnStatus::failure;
  }
  rmx_status status = rmx_retrieve_device_iface_ipv4(&m_device_interface, &device_address);
  if (status != RMX_OK) {
    std::cerr << "Failed to get device: " << m_app_settings->local_ip << " with status: " << status << std::endl;
    return ReturnStatus::failure;
  }

  return ReturnStatus::success;
}

ReturnStatus ANOMediaSenderApp::initialize_memory_strategy()
{
  std::vector<rmx_device_iface> device_interfaces = {m_device_interface};
  auto base_memory_strategy = std::make_unique<RmaxBaseMemoryStrategy>(
      *m_header_allocator, *m_payload_allocator,
      *m_memory_utils,
      device_interfaces,
      m_num_paths_per_stream,
      m_app_settings->app_memory_alloc,
      m_app_settings->register_memory);

  for (const auto& sender : m_senders) {
    base_memory_strategy->add_memory_subcomponent(sender);
  }

  m_memory_strategy.reset(base_memory_strategy.release());

  return ReturnStatus::success;
}

ReturnStatus ANOMediaSenderApp::run()
{
  if (m_obj_init_status != ReturnStatus::obj_init_success) {
    return m_obj_init_status;
  }

  ReturnStatus rc = run_stats_reader();
  if (rc == ReturnStatus::failure) {
    return ReturnStatus::failure;
  }

  try {
    run_threads(m_senders);
  }
  catch (const std::exception & error) {
    std::cerr << error.what() << std::endl;
    return ReturnStatus::failure;
  }

  return ReturnStatus::success;
}

ReturnStatus ANOMediaSenderApp::set_rivermax_clock()
{
  ReturnStatus rc = set_rivermax_ptp_clock(&m_device_interface);
  if(rc == ReturnStatus::success) {
    uint64_t ptp_time = 0;
    rc = get_rivermax_ptp_time_ns(ptp_time);
  }
  return rc;
}

void ANOMediaSenderApp::distribute_work_for_threads()
{
  m_streams_per_thread.reserve(m_media_sender_settings->thread_settings.size());
  m_media_sender_settings->num_of_total_streams = 0;
  m_media_sender_settings->num_of_threads = m_media_sender_settings->thread_settings.size();
  int thread_idx = 0;
  for (const auto& thread : m_media_sender_settings->thread_settings) {
    m_streams_per_thread[thread_idx] = thread.stream_network_settings.size();
    m_media_sender_settings->num_of_total_streams += thread.stream_network_settings.size();
    thread_idx++;
  }
}

void ANOMediaSenderApp::configure_network_flows()
{
  uint16_t source_port = 0;
  for (const auto& thread : m_media_sender_settings->thread_settings) {
    std::vector<TwoTupleFlow> streams;
    for (const auto& stream : thread.stream_network_settings) {
      streams.push_back(TwoTupleFlow(stream.stream_id, stream.destination_ip, stream.destination_port));
    }
    m_threads_streams.push_back(streams);
  }
}

void ANOMediaSenderApp::initialize_sender_threads()
{
  // All streams with the same local IP and source port
  m_app_settings->local_ip = m_media_sender_settings->thread_settings[0].stream_network_settings[0].local_ip;
  m_app_settings->source_port = 0;
  // These are not really used
  m_app_settings->destination_ip = "";
  m_app_settings->destination_port = 0;

  for (int thread_idx = 0; thread_idx < m_media_sender_settings->num_of_threads; ++thread_idx) {
    auto network_address = FourTupleFlow(
            thread_idx,
            m_app_settings->local_ip,
            m_app_settings->source_port,
            m_app_settings->destination_ip,
            m_app_settings->destination_port
          );
    m_senders.push_back(std::unique_ptr<MediaSenderIONode>(new MediaSenderIONode(
            network_address,
            m_app_settings,
            thread_idx,
            m_streams_per_thread[thread_idx],
            m_media_sender_settings->app_threads_cores[thread_idx],
            *m_memory_utils,
            ANOMediaSenderApp::get_time_ns)));
    m_senders[thread_idx]->initialize_send_flows(m_threads_streams[thread_idx]);
    m_senders[thread_idx]->initialize_streams();
  }
}

ReturnStatus ANOMediaSenderApp::set_frame_provider(size_t stream_index,
    std::shared_ptr<IFrameProvider> frame_provider, MediaType media_type, bool contains_payload)
{
  size_t sender_thread_index = 0;
  size_t sender_stream_index = 0;

  auto rc = find_internal_stream_index(stream_index, sender_thread_index, sender_stream_index);
  if (rc != ReturnStatus::success) {
    std::cerr << "Error setting frame provider, invalid stream index " << stream_index << std::endl;
    return rc;
  }

  rc = m_senders[sender_thread_index]->set_frame_provider(
      sender_stream_index, std::move(frame_provider), media_type, contains_payload);

  if (rc != ReturnStatus::success) {
    std::cerr << "Error setting frame provider for stream "
              << sender_stream_index << " on sender " << sender_thread_index << std::endl;
  }

  return rc;
}

uint64_t ANOMediaSenderApp::get_time_ns(void* context)
{
  NOT_IN_USE(context);
  uint64_t ptp_time = 0;
  ReturnStatus rc = get_rivermax_ptp_time_ns(ptp_time);
  if (rc != ReturnStatus::success) {
    std::cerr << "Failed to get PTP time" << std::endl;
    return 0;
  }
  return ptp_time;
}

} // namespace holoscan::advanced_network
