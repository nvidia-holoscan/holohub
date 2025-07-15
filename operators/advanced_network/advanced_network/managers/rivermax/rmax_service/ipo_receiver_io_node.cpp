/*
 * Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <ostream>
#include <thread>
#include <vector>

#include <rivermax_api.h>

#include "rt_threads.h"

#ifdef RMAX_APPS_LIB_FLAT_STRUCTURE
#include "ipo_receiver_io_node.h"
#include "ipo_chunk_consumer_base.h"
#else
#include "receivers/ipo_receiver_io_node.h"
#include "receivers/ipo_chunk_consumer_base.h"
#endif

#include "services/utils/cpu.h"

#define BREAK_ON_FAILURE(rc) \
  if (unlikely(rc == ReturnStatus::failure)) { break; }
#define BREAK_ON_NO_SUCCESS(rc) \
  if (unlikely(rc != ReturnStatus::success)) { break; }

using namespace ral::io_node;
using namespace ral::lib::core;
using namespace ral::lib::services;

namespace {
static constexpr uint8_t RTP_HEADER_CSRC_GRANULARITY_BYTES = 4;
static constexpr uint32_t SEQUENCE_NUMBER_MASK_16BIT = 0xFFFF;
static constexpr uint32_t SEQUENCE_NUMBER_MASK_32BIT = 0xFFFFFFFF;
}  // namespace

AppIPOReceiveStream::AppIPOReceiveStream(size_t id, const ipo_stream_settings_t& settings,
                                         bool extended_sequence_number,
                                         const std::vector<IPOReceivePath>& paths)
    : IPOReceiveStream(id, settings, paths, extended_sequence_number),
      m_is_extended_sequence_number(extended_sequence_number),
      m_sequence_number_mask((extended_sequence_number) ? SEQUENCE_NUMBER_MASK_32BIT
                                                        : SEQUENCE_NUMBER_MASK_16BIT) {
  m_path_stats.resize(paths.size());
  m_path_packets.resize(settings.num_of_packets_in_chunk, std::vector<uint8_t>(paths.size(), 0));
  m_path_stats_totals.resize(paths.size());
}

ReturnStatus AppIPOReceiveStream::get_next_chunk(IPOReceiveChunk* chunk) {
  ReturnStatus status = IPOReceiveStream::get_next_chunk(chunk);
  if (status == ReturnStatus::success) {
    m_statistic.rx_counter += chunk->get_completion_chunk_size();
    m_statistic_totals.rx_counter += chunk->get_completion_chunk_size();

    const auto packet_info_array = chunk->get_completion_info_ptr();
    for (uint32_t stride_index = 0; stride_index < chunk->get_completion_chunk_size();
         ++stride_index) {
      auto sub_block_size_0 = packet_info_array[stride_index].get_packet_sub_block_size(0);
      m_statistic.received_bytes += sub_block_size_0;
      m_statistic_totals.received_bytes += sub_block_size_0;
      if (m_header_data_split) {
        auto sub_block_size_1 = packet_info_array[stride_index].get_packet_sub_block_size(1);
        m_statistic.received_bytes += sub_block_size_1;
        m_statistic_totals.received_bytes += sub_block_size_1;
      }
    }
  }
  return status;
}

void AppIPOReceiveStream::print_statistics(
    std::ostream& out, const std::chrono::high_resolution_clock::duration& duration) const {
  std::stringstream ss;
  double duration_sec =
      static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(duration).count()) /
      1.e6;

  ss << "[stream_index " << std::setw(3) << get_id() << "]"
     << " Got " << std::setw(7) << m_statistic.rx_counter << " packets during " << std::fixed
     << std::setprecision(1) << std::setw(4) << duration_sec << " sec | ";

  ss << std::fixed << std::setprecision(2);
  double bitrate_Mbps = m_statistic.get_bitrate_Mbps();

  if (bitrate_Mbps > 1000.) {
    ss << std::setw(4) << (bitrate_Mbps / 1000.) / duration_sec << " Gbps |";
  } else {
    ss << std::setw(4) << bitrate_Mbps / duration_sec << " Mbps |";
  }

  ss << " dropped: ";
  for (uint32_t s_index = 0; s_index < m_paths.size(); ++s_index) {
    if (s_index > 0) { ss << ", "; }
    ss << m_path_stats[s_index].rx_dropped + m_statistic.rx_dropped;
  }
  ss << " |"
     << " consumed: " << m_statistic.consumed_packets << " |"
     << " unconsumed: " << m_statistic.unconsumed_packets << " |"
     << " lost: " << m_statistic.rx_dropped << " |"
     << " exceed MD: " << m_statistic.rx_exceed_md << " |"
     << " bad RTP hdr: " << m_statistic.rx_corrupt_rtp_header << " | ";

  for (uint32_t s_index = 0; s_index < m_paths.size(); ++s_index) {
    ss << " | " << m_paths[s_index].flow.get_destination_ip() << ":"
       << m_paths[s_index].flow.get_destination_port();
    if (m_statistic.rx_counter) {
      uint32_t number =
          static_cast<uint32_t>(floor(100 * static_cast<double>(m_path_stats[s_index].rx_count) /
                                      static_cast<double>(m_statistic.rx_counter)));
      ss << " " << std::setw(3) << number << "%";
    } else {
      ss << "   0%";
    }
  }

  if (m_statistic.rx_dropped) {
    ss << std::endl << "ERROR !!! Lost Packets - count: " << m_statistic.rx_dropped;
  }
  if (m_statistic.rx_corrupt_rtp_header) {
    ss << std::endl << "ERROR !!! Corrupted Packets - count: " << m_statistic.rx_corrupt_rtp_header;
  }

  out << ss.str() << std::endl;
}

void AppIPOReceiveStream::reset_statistics() {
  for (auto& stat : m_path_stats) { stat.reset(); }
  m_statistic.reset();
}

void AppIPOReceiveStream::reset_statistics_totals() {
  for (auto& stat : m_path_stats_totals) { stat.reset(); }
  m_statistic_totals.reset();
}

std::pair<IPORXStatistics, std::vector<IPOPathStatistics>> AppIPOReceiveStream::get_statistics()
    const {
  return {m_statistic_totals, m_path_stats_totals};
}

void AppIPOReceiveStream::handle_corrupted_packet(size_t index,
                                                  const ReceivePacketInfo& packet_info) {
  IPOReceiveStream::handle_corrupted_packet(index, packet_info);

  ++m_statistic.rx_corrupt_rtp_header;
  ++m_statistic_totals.rx_corrupt_rtp_header;
}

void AppIPOReceiveStream::handle_late_packet(size_t index, uint32_t sequence_number,
                                             const ReceivePacketInfo& packet_info) {
  IPOReceiveStream::handle_late_packet(index, sequence_number, packet_info);

  ++m_statistic.rx_exceed_md;
  ++m_statistic_totals.rx_exceed_md;
}

void AppIPOReceiveStream::handle_packet(size_t index, uint32_t sequence_number,
                                        const ReceivePacketInfo& packet_info) {
  IPOReceiveStream::handle_packet(index, sequence_number, packet_info);

  auto& by_paths = m_path_packets.at(sequence_number % get_sequence_number_wrap_around());

  for (size_t i = 0; i < by_paths.size(); ++i) { by_paths[i] = (i == index) ? 1 : 0; }
}

void AppIPOReceiveStream::handle_redundant_packet(size_t index, uint32_t sequence_number,
                                                  const ReceivePacketInfo& packet_info) {
  IPOReceiveStream::handle_redundant_packet(index, sequence_number, packet_info);

  auto& by_paths = m_path_packets.at(sequence_number % get_sequence_number_wrap_around());

  by_paths[index] = 1;
}

void AppIPOReceiveStream::complete_packet(uint32_t sequence_number) {
  IPOReceiveStream::complete_packet(sequence_number);

  auto& by_paths = m_path_packets.at(sequence_number % get_sequence_number_wrap_around());

  for (size_t i = 0; i < by_paths.size(); ++i) {
    m_path_stats[i].rx_count += by_paths[i];
    m_path_stats[i].rx_dropped += 1 - by_paths[i];
    m_path_stats_totals[i].rx_count += by_paths[i];
    m_path_stats_totals[i].rx_dropped += 1 - by_paths[i];
  }

  // count dropped packets by sequence number
  if (m_initialized) {
    uint32_t expected = m_last_sequence_number + 1;
    uint32_t num_dropped = (sequence_number - expected) & get_sequence_number_mask();
    m_statistic.rx_dropped += num_dropped;
    m_statistic_totals.rx_dropped += num_dropped;
  }
  m_initialized = true;
  m_last_sequence_number = sequence_number;
}

void AppIPOReceiveStream::handle_sender_restart() {
  std::cout << "Sender restart detected" << std::endl;
  m_initialized = false;
}

bool AppIPOReceiveStream::get_sequence_number(const byte_t* header, size_t length,
                                              uint32_t& sequence_number) const {
  if (length < 4 || (header[0] & 0xC0) != 0x80) { return false; }

  sequence_number = header[3] | header[2] << 8;
  if (m_is_extended_sequence_number) {
    uint8_t cc = 0x0F & header[0];
    uint8_t offset = cc * RTP_HEADER_CSRC_GRANULARITY_BYTES;
    if (length < offset + 14) { return false; }
    sequence_number |= (header[offset + 12] << 24) | (header[offset + 13] << 16);
  }
  return true;
}

void AppIPOReceiveStream::update_consumed_packets_stats(size_t packets_consumed,
                                                        size_t packets_unconsumed) {
  m_statistic.consumed_packets += packets_consumed;
  m_statistic.unconsumed_packets += packets_unconsumed;
  m_statistic_totals.consumed_packets += packets_consumed;
  m_statistic_totals.unconsumed_packets += packets_unconsumed;
}

IPOReceiverIONode::IPOReceiverIONode(const AppSettings& app_settings,
                                     uint64_t max_path_differential_us,
                                     bool extended_sequence_number, size_t max_chunk_size,
                                     const std::vector<std::string>& devices, size_t index,
                                     int cpu_core_affinity, IIPOChunkConsumer* chunk_consumer)
    : m_app_settings(app_settings),
      m_is_extended_sequence_number(extended_sequence_number),
      m_devices(devices),
      m_index(index),
      m_print_parameters(app_settings.print_parameters),
      m_cpu_core_affinity(cpu_core_affinity),
      m_sleep_between_operations(
          std::chrono::microseconds(app_settings.sleep_between_operations_us)),
      m_chunk_consumer(chunk_consumer) {
  m_stream_settings.stream_options.insert(RMX_INPUT_STREAM_CREATE_INFO_PER_PACKET);
  m_stream_settings.packet_payload_size = m_app_settings.packet_payload_size;
  m_stream_settings.packet_app_header_size = m_app_settings.packet_app_header_size;
  m_stream_settings.num_of_packets_in_chunk = m_app_settings.num_of_packets_in_chunk;
  // m_stream_settings.max_chunk_size = DEFAULT_MAX_CHUNK_SIZE;
  m_stream_settings.max_chunk_size = max_chunk_size;
  m_stream_settings.max_path_differential_us = max_path_differential_us;
}

std::ostream& IPOReceiverIONode::print(std::ostream& out) const {
  out << "+#############################################\n"
      << "| Sender index: " << m_index << "\n"
      << "| Thread ID: 0x" << std::hex << std::this_thread::get_id() << std::dec << "\n"
      << "| CPU core affinity: " << m_cpu_core_affinity << "\n"
      << "| Number of streams in this thread: " << m_streams.size() << "\n"
      << "+#############################################\n";
  for (const auto& stream : m_streams) { stream->print(out); }
  return out;
}

std::pair<std::vector<IPORXStatistics>, std::vector<std::vector<IPOPathStatistics>>>
IPOReceiverIONode::get_streams_statistics() const {
  std::vector<IPORXStatistics> streams_stats;
  std::vector<std::vector<IPOPathStatistics>> streams_path_stats;

  // Populate streams_stats and streams_path_stats
  for (const auto& stream : m_streams) {
    auto [stream_stats, stream_path_stats] = stream->get_statistics();
    streams_stats.push_back(stream_stats);
    streams_path_stats.push_back(stream_path_stats);
  }

  return {streams_stats, streams_path_stats};
}

void IPOReceiverIONode::initialize_streams(size_t start_id,
                                           const std::vector<std::vector<FourTupleFlow>>& flows) {
  size_t id = start_id;
  m_streams.reserve(flows.size());
  for (const auto& flow_list : flows) {
    std::vector<IPOReceivePath> paths;

    paths.reserve(flow_list.size());
    assert(flow_list.size() == m_devices.size());
    for (size_t i = 0; i < flow_list.size(); ++i) {
      paths.emplace_back(m_devices[i], flow_list[i]);
    }
    m_streams.emplace_back(new AppIPOReceiveStream(
        id, m_stream_settings, m_is_extended_sequence_number, std::move(paths)));
    ++id;
  }
}

void IPOReceiverIONode::print_parameters() {
  if (!m_print_parameters) { return; }

  std::stringstream receiver_parameters;
  print(receiver_parameters);
  std::cout << receiver_parameters.str() << std::endl;
}

void IPOReceiverIONode::print_statistics_settings(bool print_stats, uint32_t print_interval_ms) {
  m_print_stats = print_stats;
  m_print_interval_ms = print_interval_ms;
}

ReturnStatus IPOReceiverIONode::sync_streams() {
  ReturnStatus rc = ReturnStatus::success;
  for (auto& stream : m_streams) {
    rc = stream->sync_paths();
    BREAK_ON_FAILURE(rc);
  }
  return rc;
}

ReturnStatus IPOReceiverIONode::wait_first_packet() {
  IPOReceiveChunk chunk(m_stream_settings.packet_app_header_size != 0);
  ReturnStatus rc = ReturnStatus::success;
  bool initialized = false;
  while (likely(!initialized && rc != ReturnStatus::failure &&
                SignalHandler::get_received_signal() < 0)) {
    for (auto& stream : m_streams) {
      rc = stream->get_next_chunk(&chunk);
      BREAK_ON_FAILURE(rc);
      if (chunk.get_completion_chunk_size() > 0) {
        initialized = true;
        if (m_chunk_consumer) {
          auto [status, packets_consumed, packets_unconsumed] =
              m_chunk_consumer->consume_chunk_packets(chunk, *(stream.get()));
          stream->update_consumed_packets_stats(packets_consumed, packets_unconsumed);
        }
        break;
      }
    }
  }
  return rc;
}

void IPOReceiverIONode::operator()() {
  set_cpu_resources();
  ReturnStatus rc = create_streams();
  if (rc == ReturnStatus::failure) {
    std::cerr << "Failed to create receiver (" << m_index << ") streams" << std::endl;
    return;
  }
  rc = attach_flows();
  if (rc == ReturnStatus::failure) {
    std::cerr << "Failed to attach flows to receiver (" << m_index << ") streams" << std::endl;
    return;
  }
  print_parameters();

  IPOReceiveChunk chunk(m_stream_settings.packet_app_header_size != 0);
  rc = ReturnStatus::success;

  rc = sync_streams();
  if (rc == ReturnStatus::failure) { std::cerr << "Error during initial sync" << std::endl; }
  start();
  rc = wait_first_packet();
  if (rc == ReturnStatus::failure) {
    std::cerr << "Error during waiting for a first packet" << std::endl;
  }
  // main receive loop
  auto start_time = std::chrono::high_resolution_clock::now();

  while (likely(rc != ReturnStatus::failure && SignalHandler::get_received_signal() < 0)) {
    for (auto& stream : m_streams) {
      rc = stream->get_next_chunk(&chunk);
      if (unlikely(rc == ReturnStatus::failure)) {
        std::cerr << "Error getting next chunk of packets" << std::endl;
        break;
      }
      if (m_chunk_consumer) {
        auto [status, packets_consumed, packets_unconsumed] =
            m_chunk_consumer->consume_chunk_packets(chunk, *(stream.get()));
        stream->update_consumed_packets_stats(packets_consumed, packets_unconsumed);
        BREAK_ON_NO_SUCCESS(status);
      }
    }

    if (m_print_stats) {
      auto now = std::chrono::high_resolution_clock::now();
      auto duration = now - start_time;
      if (duration >= std::chrono::milliseconds(m_print_interval_ms)) {
        for (auto& stream : m_streams) {
          stream->print_statistics(std::cout, duration);
          stream->reset_statistics();
        }
        start_time = now;
      }
    }
    if (m_sleep_between_operations.count() > 0) {
      std::this_thread::sleep_for(m_sleep_between_operations);
    }
  }

  rc = destroy_streams();
  if (rc == ReturnStatus::failure) {
    std::cerr << "Failed to destroy sender (" << m_index << ") streams" << std::endl;
    return;
  }
}

ReturnStatus IPOReceiverIONode::create_streams() {
  ReturnStatus rc;

  for (auto& stream : m_streams) {
    rc = stream->create_stream();
    if (rc == ReturnStatus::failure) {
      std::cerr << "Failed to create stream (" << stream->get_id() << ")" << std::endl;
      return rc;
    }
  }

  return ReturnStatus::success;
}

ReturnStatus IPOReceiverIONode::attach_flows() {
  ReturnStatus rc;

  for (auto& stream : m_streams) {
    rc = stream->attach_flow();
    if (rc == ReturnStatus::failure) {
      std::cerr << "Failed to attach flow to stream (" << stream->get_id() << ")" << std::endl;
      return rc;
    }
  }

  return ReturnStatus::success;
}

void IPOReceiverIONode::start() {
  for (auto& stream : m_streams) { stream->start(); }
}

ReturnStatus IPOReceiverIONode::detach_flows() {
  ReturnStatus rc;

  for (auto& stream : m_streams) {
    rc = stream->detach_flow();
    if (rc == ReturnStatus::failure) {
      std::cerr << "Failed to detach flow from stream (" << stream->get_id() << ")" << std::endl;
      return rc;
    }
  }

  return ReturnStatus::success;
}

ReturnStatus IPOReceiverIONode::destroy_streams() {
  ReturnStatus rc;

  for (auto& stream : m_streams) {
    rc = stream->destroy_stream();
    if (rc == ReturnStatus::failure) {
      std::cerr << "Failed to destroy stream (" << stream->get_id() << ")" << std::endl;
      return rc;
    }
  }

  return ReturnStatus::success;
}

void IPOReceiverIONode::set_cpu_resources() {
  set_current_thread_affinity(m_cpu_core_affinity);
  rt_set_thread_priority(RMAX_THREAD_PRIORITY_TIME_CRITICAL - 1);
}
