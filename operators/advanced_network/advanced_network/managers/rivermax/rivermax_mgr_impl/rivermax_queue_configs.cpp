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

#include <holoscan/logger/logger.hpp>

#include "rt_threads.h"
#include "rdk/services/services.h"

#include "rivermax_config_manager.h"
#include "rivermax_queue_configs.h"

namespace holoscan::advanced_network {

using namespace rivermax::dev_kit::services;

RivermaxCommonRxQueueConfig::RivermaxCommonRxQueueConfig(const RivermaxCommonRxQueueConfig& other)
    : BaseQueueConfig(other),
      max_packet_size(other.max_packet_size),
      max_chunk_size(other.max_chunk_size),
      packets_buffers_size(other.packets_buffers_size),
      gpu_direct(other.gpu_direct),
      gpu_device_id(other.gpu_device_id),
      lock_gpu_clocks(other.lock_gpu_clocks),
      split_boundary(other.split_boundary),
      print_parameters(other.print_parameters),
      sleep_between_operations_us(other.sleep_between_operations_us),
      allocator_type(other.allocator_type),
      ext_seq_num(other.ext_seq_num),
      memory_registration(other.memory_registration),
      send_packet_ext_info(other.send_packet_ext_info),
      stats_report_interval_ms(other.stats_report_interval_ms),
      cpu_cores(other.cpu_cores),
      master_core(other.master_core),
      thread_settings(other.thread_settings) {}

RivermaxCommonRxQueueConfig& RivermaxCommonRxQueueConfig::operator=(
    const RivermaxCommonRxQueueConfig& other) {
  if (this == &other) { return *this; }
  BaseQueueConfig::operator=(other);
  max_packet_size = other.max_packet_size;
  max_chunk_size = other.max_chunk_size;
  packets_buffers_size = other.packets_buffers_size;
  gpu_direct = other.gpu_direct;
  gpu_device_id = other.gpu_device_id;
  lock_gpu_clocks = other.lock_gpu_clocks;
  split_boundary = other.split_boundary;
  print_parameters = other.print_parameters;
  sleep_between_operations_us = other.sleep_between_operations_us;
  allocator_type = other.allocator_type;
  ext_seq_num = other.ext_seq_num;
  memory_registration = other.memory_registration;
  send_packet_ext_info = other.send_packet_ext_info;
  stats_report_interval_ms = other.stats_report_interval_ms;
  cpu_cores = other.cpu_cores;
  master_core = other.master_core;
  thread_settings = other.thread_settings;
  return *this;
}

RivermaxIPOReceiverQueueConfig::RivermaxIPOReceiverQueueConfig(
    const RivermaxIPOReceiverQueueConfig& other)
    : RivermaxCommonRxQueueConfig(other),
      max_path_differential_us(other.max_path_differential_us) {}

RivermaxIPOReceiverQueueConfig& RivermaxIPOReceiverQueueConfig::operator=(
    const RivermaxIPOReceiverQueueConfig& other) {
  if (this == &other) { return *this; }
  RivermaxCommonRxQueueConfig::operator=(other);
  max_path_differential_us = other.max_path_differential_us;
  return *this;
}

RivermaxRTPReceiverQueueConfig::RivermaxRTPReceiverQueueConfig(
    const RivermaxRTPReceiverQueueConfig& other)
    : RivermaxCommonRxQueueConfig(other) {}

RivermaxRTPReceiverQueueConfig& RivermaxRTPReceiverQueueConfig::operator=(
    const RivermaxRTPReceiverQueueConfig& other) {
  if (this == &other) { return *this; }
  RivermaxCommonRxQueueConfig::operator=(other);
  return *this;
}

RivermaxCommonTxQueueConfig::RivermaxCommonTxQueueConfig(const RivermaxCommonTxQueueConfig& other)
    : gpu_direct(other.gpu_direct),
      gpu_device_id(other.gpu_device_id),
      lock_gpu_clocks(other.lock_gpu_clocks),
      split_boundary(other.split_boundary),
      local_ip(other.local_ip),
      print_parameters(other.print_parameters),
      sleep_between_operations(other.sleep_between_operations),
      allocator_type(other.allocator_type),
      memory_allocation(other.memory_allocation),
      memory_registration(other.memory_registration),
      send_packet_ext_info(other.send_packet_ext_info),
      num_of_packets_in_chunk(other.num_of_packets_in_chunk),
      stats_report_interval_ms(other.stats_report_interval_ms),
      cpu_cores(other.cpu_cores),
      master_core(other.master_core),
      dummy_sender(other.dummy_sender),
      thread_settings(other.thread_settings) {}

RivermaxCommonTxQueueConfig& RivermaxCommonTxQueueConfig::operator=(
    const RivermaxCommonTxQueueConfig& other) {
  if (this == &other) { return *this; }
  gpu_direct = other.gpu_direct;
  gpu_device_id = other.gpu_device_id;
  lock_gpu_clocks = other.lock_gpu_clocks;
  split_boundary = other.split_boundary;
  local_ip = other.local_ip;
  print_parameters = other.print_parameters;
  sleep_between_operations = other.sleep_between_operations;
  allocator_type = other.allocator_type;
  memory_allocation = other.memory_allocation;
  memory_registration = other.memory_registration;
  send_packet_ext_info = other.send_packet_ext_info;
  num_of_packets_in_chunk = other.num_of_packets_in_chunk;
  stats_report_interval_ms = other.stats_report_interval_ms;
  cpu_cores = other.cpu_cores;
  master_core = other.master_core;
  dummy_sender = other.dummy_sender;
  thread_settings = other.thread_settings;
  return *this;
}

RivermaxMediaSenderQueueConfig::RivermaxMediaSenderQueueConfig(
    const RivermaxMediaSenderQueueConfig& other)
    : RivermaxCommonTxQueueConfig(other),
      video_format(other.video_format),
      bit_depth(other.bit_depth),
      frame_width(other.frame_width),
      frame_height(other.frame_height),
      frame_rate(other.frame_rate),
      use_internal_memory_pool(other.use_internal_memory_pool),
      memory_pool_location((other.memory_pool_location)) {}

RivermaxMediaSenderQueueConfig& RivermaxMediaSenderQueueConfig::operator=(
    const RivermaxMediaSenderQueueConfig& other) {
  if (this == &other) { return *this; }
  RivermaxCommonTxQueueConfig::operator=(other);
  video_format = other.video_format;
  bit_depth = other.bit_depth;
  frame_width = other.frame_width;
  frame_height = other.frame_height;
  frame_rate = other.frame_rate;
  use_internal_memory_pool = other.use_internal_memory_pool;
  memory_pool_location = other.memory_pool_location;
  return *this;
}

void RivermaxRTPReceiverQueueConfig::dump_parameters() const {
  if (this->print_parameters) {
    HOLOSCAN_LOG_INFO("Rivermax RX Queue Config:");
    HOLOSCAN_LOG_INFO("\tNetwork settings:");
    for (const auto& thread : thread_settings) {
      HOLOSCAN_LOG_INFO("\t\tthread_id: {}", thread.thread_id);
      for (const auto& stream : thread.stream_network_settings) {
        HOLOSCAN_LOG_INFO("\t\t\tstream_id: {}", stream.stream_id);
        HOLOSCAN_LOG_INFO("\t\t\t\tlocal_ip: {}", stream.local_ip);
        HOLOSCAN_LOG_INFO("\t\t\t\tsource_ip: {}", stream.source_ip);
        HOLOSCAN_LOG_INFO("\t\t\t\tdestination_ip: {}", stream.destination_ip);
        HOLOSCAN_LOG_INFO("\t\t\t\tdestination_port: {}", stream.destination_port);
      }
    }
    HOLOSCAN_LOG_INFO("\tGPU settings:");
    HOLOSCAN_LOG_INFO("\t\tGPU ID: {}", gpu_device_id);
    HOLOSCAN_LOG_INFO("\t\tGPU Direct: {}", gpu_direct);
    HOLOSCAN_LOG_INFO("\t\tlock_gpu_clocks: {}", lock_gpu_clocks);
    HOLOSCAN_LOG_INFO("\tMemory config settings:");
    HOLOSCAN_LOG_INFO("\t\tallocator_type: {}", allocator_type);
    HOLOSCAN_LOG_INFO("\t\tmemory_registration: {}", memory_registration);
    HOLOSCAN_LOG_INFO("\tPacket settings:");
    HOLOSCAN_LOG_INFO("\t\tbatch_size/max_chunk_size: {}", max_chunk_size);
    HOLOSCAN_LOG_INFO("\t\tsplit_boundary/header_size: {}", split_boundary);
    HOLOSCAN_LOG_INFO("\t\tmax_packet_size: {}", max_packet_size);
    HOLOSCAN_LOG_INFO("\t\tpackets_buffers_size: {}", packets_buffers_size);
    HOLOSCAN_LOG_INFO("\tRMAX RTP settings:");
    HOLOSCAN_LOG_INFO("\t\text_seq_num: {}", ext_seq_num);
    HOLOSCAN_LOG_INFO("\t\tsleep_between_operations_us: {}", sleep_between_operations_us);
    HOLOSCAN_LOG_INFO("\t\tsend_packet_ext_info: {}", send_packet_ext_info);
    HOLOSCAN_LOG_INFO("\t\tstats_report_interval_ms: {}", stats_report_interval_ms);
  }
}

void RivermaxIPOReceiverQueueConfig::dump_parameters() const {
  if (this->print_parameters) {
    HOLOSCAN_LOG_INFO("Rivermax RX Queue Config:");
    HOLOSCAN_LOG_INFO("\tNetwork settings:");
    for (const auto& thread : thread_settings) {
      HOLOSCAN_LOG_INFO("\t\tthread_id: {}", thread.thread_id);
      for (const auto& stream : thread.stream_network_settings) {
        HOLOSCAN_LOG_INFO("\t\t\tstream_id: {}", stream.stream_id);
        HOLOSCAN_LOG_INFO("\t\t\t\tlocal_ips: {}", stream.local_ips);
        HOLOSCAN_LOG_INFO("\t\t\t\tsource_ips: {}", stream.source_ips);
        HOLOSCAN_LOG_INFO("\t\t\t\tdestination_ips: {}", stream.destination_ips);
        HOLOSCAN_LOG_INFO("\t\t\t\tdestination_ports: {}", stream.destination_ports);
      }
    }
    HOLOSCAN_LOG_INFO("\tGPU settings:");
    HOLOSCAN_LOG_INFO("\t\tGPU ID: {}", gpu_device_id);
    HOLOSCAN_LOG_INFO("\t\tGPU Direct: {}", gpu_direct);
    HOLOSCAN_LOG_INFO("\t\tlock_gpu_clocks: {}", lock_gpu_clocks);
    HOLOSCAN_LOG_INFO("\tMemory config settings:");
    HOLOSCAN_LOG_INFO("\t\tallocator_type: {}", allocator_type);
    HOLOSCAN_LOG_INFO("\t\tmemory_registration: {}", memory_registration);
    HOLOSCAN_LOG_INFO("\tPacket settings:");
    HOLOSCAN_LOG_INFO("\t\tbatch_size/max_chunk_size: {}", max_chunk_size);
    HOLOSCAN_LOG_INFO("\t\tsplit_boundary/header_size: {}", split_boundary);
    HOLOSCAN_LOG_INFO("\t\tmax_packet_size: {}", max_packet_size);
    HOLOSCAN_LOG_INFO("\t\tpackets_buffers_size: {}", packets_buffers_size);
    HOLOSCAN_LOG_INFO("\tRMAX IPO settings:");
    HOLOSCAN_LOG_INFO("\t\text_seq_num: {}", ext_seq_num);
    HOLOSCAN_LOG_INFO("\t\tsleep_between_operations_us: {}", sleep_between_operations_us);
    HOLOSCAN_LOG_INFO("\t\tmax_path_differential_us: {}", max_path_differential_us);
    HOLOSCAN_LOG_INFO("\t\tsend_packet_ext_info: {}", send_packet_ext_info);
    HOLOSCAN_LOG_INFO("\t\tstats_report_interval_ms: {}", stats_report_interval_ms);
  }
}

void RivermaxMediaSenderQueueConfig::dump_parameters() const {
  if (this->print_parameters) {
    HOLOSCAN_LOG_INFO("Rivermax TX Queue Config:");
    HOLOSCAN_LOG_INFO("\tNetwork settings:");
    for (const auto& thread : thread_settings) {
      HOLOSCAN_LOG_INFO("\t  thread_id: {}", thread.thread_id);
      for (const auto& stream : thread.stream_network_settings) {
        HOLOSCAN_LOG_INFO("\t    stream_id: {}", stream.stream_id);
        HOLOSCAN_LOG_INFO("\t      local_ip: {}", stream.local_ip);
        HOLOSCAN_LOG_INFO("\t      destination_ip: {}", stream.destination_ip);
        HOLOSCAN_LOG_INFO("\t      destination_port: {}", stream.destination_port);
      }
    }
    HOLOSCAN_LOG_INFO("\tGPU settings:");
    HOLOSCAN_LOG_INFO("\t  GPU ID: {}", gpu_device_id);
    HOLOSCAN_LOG_INFO("\t  GPU Direct: {}", gpu_direct);
    HOLOSCAN_LOG_INFO("\t  lock_gpu_clocks: {}", lock_gpu_clocks);
    HOLOSCAN_LOG_INFO("\tMemory config settings:");
    HOLOSCAN_LOG_INFO("\t  allocator_type: {}", allocator_type);
    HOLOSCAN_LOG_INFO("\t  memory_registration: {}", memory_registration);
    HOLOSCAN_LOG_INFO("\t  memory_allocation: {}", memory_allocation);
    HOLOSCAN_LOG_INFO("\t  use_internal_memory_pool: {}", use_internal_memory_pool);
    if (use_internal_memory_pool) {
      HOLOSCAN_LOG_INFO("\t  memory_pool_location: {}", (int)memory_pool_location);
    }
    HOLOSCAN_LOG_INFO("\tPacket settings:");
    HOLOSCAN_LOG_INFO("\t  split_boundary: {}", split_boundary);
    HOLOSCAN_LOG_INFO("\t  num_of_packets_in_chunk: {}", num_of_packets_in_chunk);
    HOLOSCAN_LOG_INFO("\tSender settings:");
    HOLOSCAN_LOG_INFO("\t  dummy_sender: {}", dummy_sender);
    HOLOSCAN_LOG_INFO("\t  send_packet_ext_info: {}", send_packet_ext_info);
    HOLOSCAN_LOG_INFO("\t  sleep_between_operations: {}", sleep_between_operations);
    HOLOSCAN_LOG_INFO("\t  stats_report_interval_ms: {}", stats_report_interval_ms);
    HOLOSCAN_LOG_INFO("\tVideo settings:");
    HOLOSCAN_LOG_INFO("\t  video_format: {}", video_format);
    HOLOSCAN_LOG_INFO("\t  bit_depth: {}", bit_depth);
    HOLOSCAN_LOG_INFO("\t  frame_width: {}", frame_width);
    HOLOSCAN_LOG_INFO("\t  frame_height: {}", frame_height);
    HOLOSCAN_LOG_INFO("\t  frame_rate: {}", frame_rate);
  }
}

ReturnStatus RivermaxCommonRxQueueValidator::validate(
    const std::shared_ptr<RivermaxCommonRxQueueConfig>& settings) const {
  ReturnStatus rc = ValidatorUtils::validate_core(settings->master_core);
  if (rc != ReturnStatus::success) { return rc; }
  int cpu_cores = ConfigManagerUtilities::validate_cores(settings->cpu_cores);
  if (cpu_cores < 0 || cpu_cores != settings->thread_settings.size()) {
    HOLOSCAN_LOG_ERROR("Number of CPU cores must match number of threads");
    return ReturnStatus::failure;
  }

  if (settings->split_boundary == 0 && settings->memory_registration) {
    HOLOSCAN_LOG_ERROR("Memory registration is supported only in header-data split mode");
    return ReturnStatus::failure;
  }
  if (settings->split_boundary == 0 && settings->gpu_direct) {
    HOLOSCAN_LOG_ERROR("GPU Direct is supported only in header-data split mode");
    return ReturnStatus::failure;
  }

  if(settings->sleep_between_operations_us < 0) {
    HOLOSCAN_LOG_ERROR("Sleep between operations must be non-negative");
    return ReturnStatus::failure;
  }

  return ReturnStatus::success;
}

ReturnStatus RivermaxIPOReceiverQueueValidator::validate(
    const std::shared_ptr<RivermaxIPOReceiverQueueConfig>& settings) const {
  auto validator = std::make_shared<RivermaxCommonRxQueueValidator>();
  ReturnStatus rc = validator->validate(settings);
  if (rc != ReturnStatus::success) { return rc; }

  int thread_id = 0;
  int stream_id = 0;
  for (const auto& thread : settings->thread_settings) {
    if (thread.thread_id != thread_id) {
      HOLOSCAN_LOG_ERROR("Thread ID must be sequential starting from 0");
      return ReturnStatus::failure;
    }
    if (thread.stream_network_settings.empty()) {
      HOLOSCAN_LOG_ERROR("No stream network settings provided for thread ID {}", thread.thread_id);
      return ReturnStatus::failure;
    }
    thread_id++;
    for (const auto& stream : thread.stream_network_settings) {
      // if (stream.stream_id != stream_id) {
      //   HOLOSCAN_LOG_ERROR("Stream ID must be sequential starting from 0");
      //   return ReturnStatus::failure;
      // }
      stream_id++;
      if (stream.source_ips.empty()) {
        HOLOSCAN_LOG_ERROR(
            "Source IPs are not set for stream in thread ID {}", thread.thread_id);
        return ReturnStatus::failure;
      }
      rc = ValidatorUtils::validate_ip4_address(stream.source_ips);
      if (rc != ReturnStatus::success) { return rc; }
      if (stream.destination_ips.size() != stream.source_ips.size()) {
        HOLOSCAN_LOG_ERROR(
            "Must be the same number of destination multicast IPs as number of source IPs "
            "for stream in thread ID {}",
            thread.thread_id);
        return ReturnStatus::failure;
      }
      rc = ValidatorUtils::validate_ip4_address(stream.destination_ips);
      if (rc != ReturnStatus::success) { return rc; }
      if (stream.local_ips.size() != stream.source_ips.size()) {
        HOLOSCAN_LOG_ERROR(
            "Must be the same number of NIC addresses as number of source IPs for stream in "
            "thread ID {}",
            thread.thread_id);
        return ReturnStatus::failure;
      }
      rc = ValidatorUtils::validate_ip4_address(stream.local_ips);
      if (rc != ReturnStatus::success) { return rc; }
      if (stream.destination_ports.size() != stream.source_ips.size()) {
        HOLOSCAN_LOG_ERROR(
            "Must be the same number of destination ports as number of source IPs for stream in "
            "thread ID {}",
            thread.thread_id);
        return ReturnStatus::failure;
      }
      rc = ValidatorUtils::validate_ip4_port(stream.destination_ports);
      if (rc != ReturnStatus::success) { return rc; }
    }
  }

  return ReturnStatus::success;
}

ReturnStatus RivermaxRTPReceiverQueueValidator::validate(
    const std::shared_ptr<RivermaxRTPReceiverQueueConfig>& settings) const {
  auto validator = std::make_shared<RivermaxCommonRxQueueValidator>();
  ReturnStatus rc = validator->validate(settings);
  if (rc != ReturnStatus::success) { return rc; }

  int thread_id = 0;
  int stream_id = 0;
  for (const auto& thread : settings->thread_settings) {
    if (thread.thread_id != thread_id) {
      HOLOSCAN_LOG_ERROR("Thread ID must be sequential starting from 0");
      return ReturnStatus::failure;
    }
    thread_id++;
    if (thread.stream_network_settings.empty()) {
      HOLOSCAN_LOG_ERROR("No stream network settings provided for thread ID {}", thread.thread_id);
      return ReturnStatus::failure;
    }
    for (const auto& stream : thread.stream_network_settings) {
      if (stream.stream_id != stream_id) {
        HOLOSCAN_LOG_ERROR("Stream ID must be sequential starting from 0");
        return ReturnStatus::failure;
      }
      stream_id++;
      rc = ValidatorUtils::validate_ip4_address(stream.source_ip);
      if (rc != ReturnStatus::success) { return rc; }
      rc = ValidatorUtils::validate_ip4_address(stream.destination_ip);
      if (rc != ReturnStatus::success) { return rc; }
      rc = ValidatorUtils::validate_ip4_address(stream.local_ip);
      if (rc != ReturnStatus::success) { return rc; }
      rc = ValidatorUtils::validate_ip4_port(stream.destination_port);
      if (rc != ReturnStatus::success) { return rc; }
    }
  }
  return ReturnStatus::success;
}

ReturnStatus RivermaxCommonTxQueueValidator::validate(
    const std::shared_ptr<RivermaxCommonTxQueueConfig>& settings) const {
  ReturnStatus rc = ValidatorUtils::validate_core(settings->master_core);
  if (rc != ReturnStatus::success) { return rc; }
  int thread_id = 0;
  int stream_id = 0;
  for (const auto& thread : settings->thread_settings) {
    thread_id++;
    if (thread.stream_network_settings.empty()) {
      HOLOSCAN_LOG_ERROR("No stream network settings provided for thread ID {}", thread.thread_id);
      return ReturnStatus::failure;
    }
    for (const auto& stream : thread.stream_network_settings) {
      // if (stream.stream_id != stream_id) {
      //   HOLOSCAN_LOG_ERROR("Stream ID must be sequential starting from 0");
      //   return ReturnStatus::failure;
      // }
      // stream_id++;
      rc = ValidatorUtils::validate_ip4_address(stream.destination_ip);
      if (rc != ReturnStatus::success) { return rc; }
      rc = ValidatorUtils::validate_ip4_address(stream.local_ip);
      if (rc != ReturnStatus::success) { return rc; }
      rc = ValidatorUtils::validate_ip4_port(stream.destination_port);
      if (rc != ReturnStatus::success) { return rc; }
    }
  }
  if (!settings->memory_allocation && settings->memory_registration) {
    HOLOSCAN_LOG_ERROR(
        "Register memory option is supported only with application memory allocation");
    return ReturnStatus::failure;
  }
  if (settings->gpu_direct && settings->split_boundary == 0) {
    HOLOSCAN_LOG_ERROR("GPU Direct is supported only in header-data split mode");
    return ReturnStatus::failure;
  }

  return ReturnStatus::success;
}

ReturnStatus RivermaxMediaSenderQueueValidator::validate(
    const std::shared_ptr<RivermaxMediaSenderQueueConfig>& settings) const {
  auto validator = std::make_shared<RivermaxCommonTxQueueValidator>();
  ReturnStatus rc = validator->validate(settings);
  if (rc != ReturnStatus::success) { return rc; }

  return ReturnStatus::success;
}

ReturnStatus RivermaxQueueToIPOReceiverSettingsBuilder::convert_settings(
    const std::shared_ptr<RivermaxIPOReceiverQueueConfig>& source_settings,
    std::shared_ptr<ANOIPOReceiverSettings>& target_settings) {
  target_settings->thread_settings = source_settings->thread_settings;
  target_settings->num_of_threads = target_settings->thread_settings.size();

  if (source_settings->gpu_direct) {
    target_settings->gpu_id = source_settings->gpu_device_id;
  } else {
    target_settings->gpu_id = INVALID_GPU_ID;
  }

  ConfigManagerUtilities::set_allocator_type(*target_settings, source_settings->allocator_type);
  if (source_settings->master_core < 0) {
    target_settings->internal_thread_core = CPU_NONE;
    target_settings->lock_gpu_clocks = source_settings->lock_gpu_clocks;
  } else {
    target_settings->internal_thread_core = source_settings->master_core;
  }
  bool res = ConfigManagerUtilities::parse_and_set_cores(target_settings->app_threads_cores,
                                                         source_settings->cpu_cores);
  if (!res) {
    HOLOSCAN_LOG_ERROR("Failed to parse CPU cores");
    return ReturnStatus::failure;
  }

  target_settings->print_parameters = source_settings->print_parameters;
  target_settings->sleep_between_operations_us = source_settings->sleep_between_operations_us;
  target_settings->packet_payload_size = source_settings->max_packet_size;
  target_settings->packet_app_header_size = source_settings->split_boundary;
  (target_settings->packet_app_header_size == 0) ? target_settings->header_data_split = false :
    target_settings->header_data_split = true;

  target_settings->num_of_packets_in_chunk =
      std::pow(2, std::ceil(std::log2(source_settings->packets_buffers_size)));
  target_settings->is_extended_sequence_number = source_settings->ext_seq_num;
  target_settings->max_path_differential_us = source_settings->max_path_differential_us;
  if (target_settings->max_path_differential_us >= USECS_IN_SECOND) {
    HOLOSCAN_LOG_ERROR("Max path differential must be less than 1 second");
    target_settings->max_path_differential_us = USECS_IN_SECOND;
  }

  target_settings->stats_report_interval_ms = source_settings->stats_report_interval_ms;
  target_settings->register_memory = source_settings->memory_registration;
  target_settings->max_packets_in_rx_chunk = source_settings->max_chunk_size;

  send_packet_ext_info_ = source_settings->send_packet_ext_info;
  settings_built_ = true;
  built_settings_ = *target_settings;

  return ReturnStatus::success;
}

ReturnStatus RivermaxQueueToRTPReceiverSettingsBuilder::convert_settings(
    const std::shared_ptr<RivermaxRTPReceiverQueueConfig>& source_settings,
    std::shared_ptr<ANORTPReceiverSettings>& target_settings) {
  target_settings->thread_settings = source_settings->thread_settings;
  target_settings->num_of_threads = target_settings->thread_settings.size();

  if (source_settings->gpu_direct) {
    target_settings->gpu_id = source_settings->gpu_device_id;
  } else {
    target_settings->gpu_id = INVALID_GPU_ID;
  }

  ConfigManagerUtilities::set_allocator_type(*target_settings, source_settings->allocator_type);
  if (source_settings->master_core < 0) {
    target_settings->internal_thread_core = CPU_NONE;
    target_settings->lock_gpu_clocks = source_settings->lock_gpu_clocks;
  } else {
    target_settings->internal_thread_core = source_settings->master_core;
  }
  bool res = ConfigManagerUtilities::parse_and_set_cores(target_settings->app_threads_cores,
                                                         source_settings->cpu_cores);
  if (!res) {
    HOLOSCAN_LOG_ERROR("Failed to parse CPU cores");
    return ReturnStatus::failure;
  }

  target_settings->print_parameters = source_settings->print_parameters;
  target_settings->sleep_between_operations_us = source_settings->sleep_between_operations_us;
  target_settings->packet_payload_size = source_settings->max_packet_size;
  target_settings->packet_app_header_size = source_settings->split_boundary;
  (target_settings->packet_app_header_size == 0) ? target_settings->header_data_split = false :
    target_settings->header_data_split = true;

  target_settings->num_of_packets_in_chunk =
      std::pow(2, std::ceil(std::log2(source_settings->packets_buffers_size)));
  target_settings->is_extended_sequence_number = source_settings->ext_seq_num;

  target_settings->stats_report_interval_ms = source_settings->stats_report_interval_ms;
  target_settings->register_memory = source_settings->memory_registration;
  max_chunk_size_ = source_settings->max_chunk_size;
  send_packet_ext_info_ = source_settings->send_packet_ext_info;

  settings_built_ = true;
  built_settings_ = *target_settings;

  return ReturnStatus::success;
}

ReturnStatus RivermaxQueueToMediaSenderSettingsBuilder::convert_settings(
    const std::shared_ptr<RivermaxMediaSenderQueueConfig>& source_settings,
    std::shared_ptr<ANOMediaSenderSettings>& target_settings) {
  target_settings->thread_settings = source_settings->thread_settings;
  target_settings->num_of_threads = target_settings->thread_settings.size();
  // Same local IP for all streams of the first thread
  // Next - This will be taken from the adv_network settings `address`
  target_settings->local_ip = source_settings->thread_settings[0].stream_network_settings[0].local_ip;

  if (source_settings->gpu_direct) {
    target_settings->gpu_id = source_settings->gpu_device_id;
    target_settings->lock_gpu_clocks = source_settings->lock_gpu_clocks;
  } else {
    target_settings->gpu_id = INVALID_GPU_ID;
  }

  ConfigManagerUtilities::set_allocator_type(*target_settings, source_settings->allocator_type);
  if (source_settings->master_core < 0) {
    target_settings->internal_thread_core = CPU_NONE;
  } else {
    target_settings->internal_thread_core = source_settings->master_core;
  }
  bool res = ConfigManagerUtilities::parse_and_set_cores(target_settings->app_threads_cores,
                                                         source_settings->cpu_cores);
  if (!res) {
    HOLOSCAN_LOG_ERROR("Failed to parse CPU cores");
    return ReturnStatus::failure;
  }

  target_settings->print_parameters = source_settings->print_parameters;
  target_settings->sleep_between_operations = source_settings->sleep_between_operations;
  target_settings->packet_app_header_size = source_settings->split_boundary;
  (target_settings->packet_app_header_size == 0) ? target_settings->header_data_split = false :
    target_settings->header_data_split = true;

  target_settings->stats_report_interval_ms = source_settings->stats_report_interval_ms;
  target_settings->register_memory = source_settings->memory_registration;
  target_settings->app_memory_alloc = source_settings->memory_allocation;
  target_settings->num_of_packets_in_chunk = source_settings->num_of_packets_in_chunk;

  target_settings->media.resolution.height = source_settings->frame_height;
  target_settings->media.resolution.width = source_settings->frame_width;
  target_settings->media.frame_rate = FrameRate(source_settings->frame_rate);
  target_settings->media.bit_depth =
      ConfigManagerUtilities::convert_bit_depth(source_settings->bit_depth);
  target_settings->media.sampling_type =
      ConfigManagerUtilities::convert_video_sampling(source_settings->video_format);

  dummy_sender_ = source_settings->dummy_sender;
  settings_built_ = true;
  built_settings_ = *target_settings;

  return ReturnStatus::success;
}

std::string queue_config_type_to_string(QueueConfigType type) {
  static const std::map<QueueConfigType, std::string> queueTypeMap = {
      {QueueConfigType::IPOReceiver, "IPOReceiver"},
      {QueueConfigType::RTPReceiver, "RTPReceiver"},
      {QueueConfigType::MediaFrameSender, "MediaFrameSender"},
      {QueueConfigType::GenericPacketSender, "GenericPacketSender"}};

  auto it = queueTypeMap.find(type);
  if (it != queueTypeMap.end()) {
    return it->second;
  } else {
    return "Unknown";
  }
}

}  // namespace holoscan::advanced_network
