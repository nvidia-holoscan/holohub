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

#ifndef RMAX_APPS_LIB_IO_NODE_RECEIVERS_IPO_RECEIVER_IO_NODE_H_

#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>
#include <chrono>

#include <rivermax_api.h>

#include "api/rmax_apps_lib_api.h"
#include "ipo_chunk_consumer_base.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

namespace ral {
namespace io_node {

/**
 * @brief: Receiving statistics struct.
 *
 * This struct will hold run time statistics of a stream.
 */
struct IPORXStatistics {
  size_t rx_counter = 0;
  size_t rx_dropped = 0;
  size_t rx_corrupt_rtp_header = 0;
  size_t rx_exceed_md = 0;
  size_t received_bytes = 0;
  size_t consumed_packets = 0;
  size_t unconsumed_packets = 0;

  /**
   * @brief: Resets values to zero.
   */
  void reset() {
    rx_counter = 0;
    rx_dropped = 0;
    rx_corrupt_rtp_header = 0;
    rx_exceed_md = 0;
    received_bytes = 0;
    consumed_packets = 0;
    unconsumed_packets = 0;
  }

  /**
   * @return: Bitrate in Mbits per second.
   */
  double get_bitrate_Mbps() const { return ((received_bytes * 8) / 1.e6); }
};

/**
 * @brief: Receives path statistics.
 */
struct IPOPathStatistics {
  uint32_t rx_count = 0;
  uint32_t rx_dropped = 0;

  /**
   * @brief: Resets values to zero.
   */
  void reset() {
    rx_count = 0;
    rx_dropped = 0;
  }
};

/**
 * @brief: Application IPO receive stream specialized to parse RTP streams.
 *
 * This class implements and extends @ref ral::lib::core::IPOReceiveStream operations.
 */
class AppIPOReceiveStream : public IPOReceiveStream {
 private:
  const bool m_is_extended_sequence_number;
  const uint32_t m_sequence_number_mask;

  IPORXStatistics m_statistic;
  std::vector<IPOPathStatistics> m_path_stats;
  IPORXStatistics m_statistic_totals;
  std::vector<IPOPathStatistics> m_path_stats_totals;
  std::vector<std::vector<uint8_t>> m_path_packets;
  bool m_initialized = false;
  uint32_t m_last_sequence_number = 0;

 public:
  /**
   * @brief: Constructs Inline Packet Ordering stream wrapper.
   *
   * @param [in] id: Stream identifier.
   * @param [in] settings: Stream settings.
   * @param [in] extended_sequence_number: Parse extended sequence number.
   * @param [in] paths: List of redundant data receive paths.
   */
  AppIPOReceiveStream(size_t id, const ipo_stream_settings_t& settings,
                      bool extended_sequence_number, const std::vector<IPOReceivePath>& paths);
  virtual ~AppIPOReceiveStream() = default;

  /**
   * @brief: Prints stream statistics.
   *
   * @param [out] out: Output stream to print statistics to.
   * @param [in] duration: Statistics interval duration.
   */
  void print_statistics(std::ostream& out,
                        const std::chrono::high_resolution_clock::duration& duration) const;
  /**
   * @brief: Resets statistics.
   */
  void reset_statistics();
  /**
   * @brief: Resets totals statistics.
   */
  void reset_statistics_totals();
  /**
   * @brief: Receives next chunk from input stream.
   *
   * @param [out] chunk: Pointer to the returned chunk structure.
   *
   * @return: Status of the operation:
   *          @ref ral::lib::services::ReturnStatus::success - In case of success.
   *          @ref ral::lib::services::ReturnStatus::signal_received - If operation was interrupted
   * by an OS signal.
   *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status
   * will be logged.
   */
  ReturnStatus get_next_chunk(IPOReceiveChunk* chunk) final;
  /**
   * @brief: Returns stream statistics.
   *
   * @return: Stream statistics.
   */
  std::pair<IPORXStatistics, std::vector<IPOPathStatistics>> get_statistics() const;

  /**
   * @brief Updates the statistics for consumed and unconsumed packets.
   *
   * This function updates the internal statistics for the number of packets that have been
   * consumed and unconsumed. It is used to keep track of packet processing within the system.
   *
   * @param packets_consumed The number of packets that have been successfully consumed.
   * @param packets_unconsumed The number of packets that have not been consumed.
   */
  void update_consumed_packets_stats(size_t packets_consumed, size_t packets_unconsumed);

 private:
  /**
   * @brief: Handles packet with corrupt RTP header.
   *
   * @param [in] index: Redundant stream index (0-based).
   * @param [in] packet_info: Detailed packet information.
   */
  void handle_corrupted_packet(size_t index, const ReceivePacketInfo& packet_info) final;
  /**
   * @brief: Handles packet that arrived too late.
   *
   * @param [in] index: Redundant stream index (0-based).
   * @param [in] sequence_number: RTP sequence number.
   * @param [in] packet_info: Detailed packet information.
   */
  virtual void handle_late_packet(size_t index, uint32_t sequence_number,
                                  const ReceivePacketInfo& packet_info);
  /**
   * @brief: Handles received packet.
   *
   * This function is called only for the first packet, for redundant packets
   * copies received from another streams @ref handle_redundant_packet will
   * be called.
   *
   * @param [in] index: Redundant stream index (0-based).
   * @param [in] sequence_number: Sequence number.
   * @param [in] packet_info: Detailed packet information.
   */
  void handle_packet(size_t index, uint32_t sequence_number,
                     const ReceivePacketInfo& packet_info) final;
  /**
   * @brief: Handles received redundant packet.
   *
   * This function is called only for redundant packet(s), for the first
   * received packet @ref handle_packet will be called.
   *
   * @param [in] index: Redundant stream index (0-based).
   * @param [in] sequence_number: Sequence number.
   * @param [in] packet_info: Detailed packet information.
   */
  void handle_redundant_packet(size_t index, uint32_t sequence_number,
                               const ReceivePacketInfo& packet_info) final;
  /**
   * @brief: Handles packet before returning it to caller.
   *
   * This function is called when packet is transferred from cache buffer to
   * the caller.
   *
   * @param [in] sequence_number: Sequence number.
   */
  void complete_packet(uint32_t sequence_number) final;
  /**
   * @brief: Handles sender restart.
   *
   * This function is called once receiver detects that the sender restarted streaming.
   */
  virtual void handle_sender_restart();

 protected:
  /**
   * @brief: Extracts sequence number from RTP packet and (if needed) payload header.
   *
   * If @ref m_is_extended_sequence_number is set then parse the 16 high
   * order bits of the extended 32-bit sequence number from the start of RTP
   * payload.
   *
   * @param [in] header: Pointer to start of RTP header.
   * @param [in] length: Header length.
   * @param [out] sequence_number: Sequence number.
   *
   * @return: true if packet header is valid.
   */
  bool get_sequence_number(const byte_t* header, size_t length,
                           uint32_t& sequence_number) const final;
  /**
   * @brief: Gets sequence number mask.
   *
   * @return: Sequence number mask.
   */
  uint32_t get_sequence_number_mask() const final { return m_sequence_number_mask; }
};

/**
 * @brief: IPOReceiverIONode class.
 *
 * This class implements the required operations in order to be a IPO receiver.
 * The sender class will be the context that will be run under a std::thread by
 * overriding the operator ().  Each receiver will be able to run multiple
 * streams.
 */
class IPOReceiverIONode {
 private:
  static constexpr size_t DEFAULT_MAX_CHUNK_SIZE = 1024;
  const AppSettings m_app_settings;
  const bool m_is_extended_sequence_number;
  const std::vector<std::string> m_devices;
  const size_t m_index;
  const bool m_print_parameters;
  const int m_cpu_core_affinity;
  const std::chrono::microseconds m_sleep_between_operations;

  ipo_stream_settings_t m_stream_settings;
  std::vector<std::unique_ptr<AppIPOReceiveStream>> m_streams;

  IIPOChunkConsumer* m_chunk_consumer = nullptr;
  bool m_print_stats = true;
  uint32_t m_print_interval_ms = 1000;

 public:
  /**
   * @brief: IPOReceiverNode constructor.
   *
   * @param [in] app_settings: Application settings.
   * @param [in] max_path_differential_us: Maximum Path Differential value.
   * @param [in] extended_sequence_number: Parse extended sequence number.
   * @param [in] devices: List of NICs to receive data.
   * @param [in] index: Receiver index.
   * @param [in] cpu_core_affinity: CPU core affinity the sender will run on.
   */
  IPOReceiverIONode(const AppSettings& app_settings, uint64_t max_path_differential_us,
                    bool extended_sequence_number, size_t max_chunk_size,
                    const std::vector<std::string>& devices, size_t index, int cpu_core_affinity,
                    IIPOChunkConsumer* chunk_consumer = nullptr);

  virtual ~IPOReceiverIONode() = default;
  /**
   * @brief: Returns receiver's streams container.
   *
   * @return: Receiver's streams container.
   */
  std::vector<std::unique_ptr<AppIPOReceiveStream>>& get_streams() { return m_streams; }
  /**
   * @brief: Prints receiver's parameters to a output stream.
   *
   * The method prints the parameters of the receiver to be shown to the user
   * to a output stream.
   *
   * @param [out] out: Output stream parameter print to.
   *
   * @return: Output stream.
   */
  std::ostream& print(std::ostream& out) const;
  /**
   * @brief: Overrides operator << for @ref ral::io_node::IPOReceiverIONode reference.
   */
  friend std::ostream& operator<<(std::ostream& out, const IPOReceiverIONode& receiver) {
    receiver.print(out);
    return out;
  }
  /**
   * @brief: Overrides operator << for @ref ral::io_node::IPOReceiverIONode pointer.
   */
  friend std::ostream& operator<<(std::ostream& out, IPOReceiverIONode* receiver) {
    receiver->print(out);
    return out;
  }
  /**
   * @brief: Initializes receive streams.
   *
   * @param [in] start_id: Starting identifier for streams list.
   * @param [in] flows: Vector of vectors of flows to be received by streams.
   *                    Each item in outer level vector must contain vector
   *                    of the same number of items as in devices list passed
   *                    into constructor. Each item in the inner vector will
   *                    be mapped to a corresponding device.
   */
  void initialize_streams(size_t start_id, const std::vector<std::vector<FourTupleFlow>>& flows);
  /**
   * @brief: Prints receiver's parameters.
   *
   * @note: The information will be printed if the receiver was initialized with
   *         @ref app_settings->print_parameters parameter of set to true.
   */
  void print_parameters();
  /**
   * @brief: Gets receiver index.
   *
   * @return: Receiver index.
   */
  size_t get_index() const { return m_index; }
  /**
   * @brief: Receiver's worker.
   *
   * This method is the worker method of the std::thread will run with this
   * object as it's context. The user of @ref ral::io_node::IPOReceiverIONode
   * class can initialize the object in advance and run std::thread when
   * needed.
   */
  void operator()();

  /**
   * @brief: Prints statistics settings.
   *
   * @param [in] print_stats: Print statistics flag.
   * @param [in] print_interval_ms: Print interval in milliseconds.
   */
  void print_statistics_settings(bool print_stats, uint32_t print_interval_ms);

  /**
   * @brief: Returns streams statistics.
   *
   * @return: Pair of vectors of statistics. First vector contains stream
   *          statistics, second vector contains path statistics.
   */
  std::pair<std::vector<IPORXStatistics>, std::vector<std::vector<IPOPathStatistics>>>
  get_streams_statistics() const;

 private:
  /**
   * @brief: Creates receiver's streams.
   *
   * This method is responsible to go over receiver's stream objects and
   * create the appropriate Rivermax streams.
   *
   * @return: Status of the operation.
   */
  ReturnStatus create_streams();
  /**
   * @brief: Attaches flows to receiver's streams.
   *
   * @return: Status of the operation.
   */
  ReturnStatus attach_flows();
  /**
   * @brief: Sync all streams.
   *
   * Flushes input buffers of all streams.
   */
  ReturnStatus sync_streams();
  /**
   * @brief: Start all streams.
   */
  void start();
  /**
   * @brief: Wait for a first input packet.
   *
   * @return: Status code.
   */
  ReturnStatus wait_first_packet();
  /**
   * @brief: Detaches flows from receiver's streams.
   *
   * @return: Status of the operation.
   */
  ReturnStatus detach_flows();
  /**
   * @brief: Destroys receiver's streams.
   *
   * This method is responsible to go over receiver's stream objects and
   * destroy the appropriate Rivermax stream.
   *
   * @return: Status of the operation.
   */
  ReturnStatus destroy_streams();
  /**
   * @brief: Sets CPU related resources.
   *
   * This method is responsible to set sender's priority and CPU core affinity.
   */
  void set_cpu_resources();
};

}  // namespace io_node
}  // namespace ral

#define RMAX_APPS_LIB_IO_NODE_RECEIVERS_IPO_RECEIVER_IO_NODE_H_
#endif /* RMAX_APPS_LIB_IO_NODE_RECEIVERS_IPO_RECEIVER_IO_NODE_H_ */
