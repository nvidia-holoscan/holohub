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

#include "advanced_network/common.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <yaml-cpp/yaml.h>

namespace py = pybind11;
using pybind11::literals::operator""_a;

namespace holoscan::advanced_network {

PYBIND11_MODULE(_advanced_network_common, m) {
  m.doc() = "Advanced Network utility functions";

  // Bind enums
  py::enum_<Status>(m, "Status")
      .value("SUCCESS", Status::SUCCESS)
      .value("NULL_PTR", Status::NULL_PTR)
      .value("NO_FREE_BURST_BUFFERS", Status::NO_FREE_BURST_BUFFERS)
      .value("NO_FREE_PACKET_BUFFERS", Status::NO_FREE_PACKET_BUFFERS)
      .value("NOT_READY", Status::NOT_READY)
      .value("INVALID_PARAMETER", Status::INVALID_PARAMETER)
      .value("NO_SPACE_AVAILABLE", Status::NO_SPACE_AVAILABLE)
      .value("NOT_SUPPORTED", Status::NOT_SUPPORTED)
      .value("INTERNAL_ERROR", Status::INTERNAL_ERROR);

  py::enum_<ManagerType>(m, "ManagerType")
      .value("UNKNOWN", ManagerType::UNKNOWN)
      .value("DEFAULT", ManagerType::DEFAULT)
      .value("DPDK", ManagerType::DPDK)
      .value("DOCA", ManagerType::DOCA)
      .value("RIVERMAX", ManagerType::RIVERMAX);

  py::enum_<Direction>(m, "Direction")
      .value("RX", Direction::RX)
      .value("TX", Direction::TX)
      .value("TX_RX", Direction::TX_RX);

  py::enum_<BufferLocation>(m, "BufferLocation")
      .value("CPU", BufferLocation::CPU)
      .value("GPU", BufferLocation::GPU)
      .value("CPU_GPU_SPLIT", BufferLocation::CPU_GPU_SPLIT);

  py::enum_<MemoryKind>(m, "MemoryKind")
      .value("HOST", MemoryKind::HOST)
      .value("HOST_PINNED", MemoryKind::HOST_PINNED)
      .value("HUGE", MemoryKind::HUGE)
      .value("DEVICE", MemoryKind::DEVICE)
      .value("INVALID", MemoryKind::INVALID);

  // Bind classes with their members
  py::class_<BurstHeaderParams>(m, "BurstHeaderParams")
      .def(py::init<>())
      .def_readwrite("num_pkts", &BurstHeaderParams::num_pkts)
      .def_readwrite("port_id", &BurstHeaderParams::port_id)
      .def_readwrite("q_id", &BurstHeaderParams::q_id)
      .def_readwrite("num_segs", &BurstHeaderParams::num_segs)
      .def_readwrite("nbytes", &BurstHeaderParams::nbytes)
      .def_readwrite("first_pkt_addr", &BurstHeaderParams::first_pkt_addr)
      .def_readwrite("max_pkt", &BurstHeaderParams::max_pkt)
      .def_readwrite("max_pkt_size", &BurstHeaderParams::max_pkt_size)
      .def_readwrite("gpu_pkt0_idx", &BurstHeaderParams::gpu_pkt0_idx)
      .def_readwrite("gpu_pkt0_addr", &BurstHeaderParams::gpu_pkt0_addr)
      .def_readwrite("burst_flags", &BurstHeaderParams::burst_flags);

  py::class_<BurstHeader>(m, "BurstHeader")
      .def(py::init<>())
      .def_readwrite("hdr", &BurstHeader::hdr);

  py::class_<BurstParams>(m, "BurstParams")
      .def(py::init<>())
      .def_readwrite("hdr", &BurstParams::hdr);

  // Network initialization and configuration
  // Note: The original adv_net_init(NetworkConfig&) is not bound since NetworkConfig
  // is not bound to Python. Instead, we provide a Python-friendly version that handles
  // Holoscan config objects and converts them internally.
  m.def(
      "adv_net_init",
      [](py::object config_obj) -> Status {
        try {
          // The config_obj is likely a Holoscan Arg or similar object
          // We need to extract the underlying YAML data

          // Try to get the YAML representation
          std::string yaml_str;

          // Check if it has a 'value' attribute (Holoscan Arg objects)
          if (py::hasattr(config_obj, "value")) {
            auto value = config_obj.attr("value");
            yaml_str = py::str(value);
          } else if (py::hasattr(config_obj, "as_dict")) {
            auto dict_obj = config_obj.attr("as_dict")();
            // Convert dict to YAML string
            yaml_str = py::str(dict_obj);
          } else {
            yaml_str = py::str(config_obj);
          }

          HOLOSCAN_LOG_DEBUG("Attempting to parse YAML config: {}", yaml_str);

          // Parse YAML string to YAML::Node
          YAML::Node yaml_node = YAML::Load(yaml_str);

          // Use the existing YAML::convert specialization to convert to NetworkConfig
          NetworkConfig config = yaml_node.as<NetworkConfig>();

          // Call the actual initialization function
          auto status = adv_net_init(config);
          if (status == Status::SUCCESS) {
            HOLOSCAN_LOG_INFO("Successfully initialized Advanced Network from Python config");
          }
          return status;
        } catch (const YAML::Exception& e) {
          HOLOSCAN_LOG_ERROR("YAML parsing error in Python config: {}", e.what());
          return Status::INVALID_PARAMETER;
        } catch (const std::exception& e) {
          HOLOSCAN_LOG_ERROR("Failed to initialize advanced network from Python config: {}",
                             e.what());
          return Status::INTERNAL_ERROR;
        }
      },
      "Initialize the advanced network backend from Holoscan config object",
      "config"_a);

  m.def("get_manager_type",
        static_cast<ManagerType (*)()>(&get_manager_type),
        "Get the current manager type");

  // Burst creation and management
  m.def("create_tx_burst_params",
        &create_tx_burst_params,
        py::return_value_policy::reference,
        "Create a shared pointer burst params structure");

  // Packet information functions
  m.def("get_segment_packet_length",
        py::overload_cast<BurstParams*, int, int>(&get_segment_packet_length),
        "Get length of one segment of the packet",
        "burst"_a,
        "seg"_a,
        "idx"_a);
  m.def("get_packet_length",
        py::overload_cast<BurstParams*, int>(&get_packet_length),
        "Get length of the packet",
        "burst"_a,
        "idx"_a);
  m.def("get_packet_flow_id",
        py::overload_cast<BurstParams*, int>(&get_packet_flow_id),
        "Get flow ID of a packet",
        "burst"_a,
        "idx"_a);

  // Packet pointer functions
  m.def("get_segment_packet_ptr",
        py::overload_cast<BurstParams*, int, int>(&get_segment_packet_ptr),
        "Get packet pointer for one segment",
        "burst"_a,
        "seg"_a,
        "idx"_a);
  m.def("get_packet_ptr",
        py::overload_cast<BurstParams*, int>(&get_packet_ptr),
        "Get packet pointer",
        "burst"_a,
        "idx"_a);

  // TX burst functions
  m.def("tx_burst_available",
        py::overload_cast<BurstParams*>(&is_tx_burst_available),
        "Return true if a TX burst is available for use",
        "burst"_a);
  m.def("get_tx_packet_burst",
        py::overload_cast<BurstParams*>(&get_tx_packet_burst),
        "Get TX packet burst",
        "burst"_a);
  m.def("send_tx_burst",
        py::overload_cast<BurstParams*>(&send_tx_burst),
        "Send a TX burst",
        "burst"_a);

  // RX burst functions - all overloads
  m.def(
      "get_rx_burst",
      [](int port, int q) {
        BurstParams* burst_ptr = nullptr;
        Status status = get_rx_burst(&burst_ptr, port, q);
        return py::make_tuple(status, py::cast(burst_ptr, py::return_value_policy::reference));
      },
      "Get RX burst for specific port and queue",
      "port"_a,
      "q"_a);

  m.def(
      "get_rx_burst",
      [](int port) {
        BurstParams* burst_ptr = nullptr;
        Status status = get_rx_burst(&burst_ptr, port);
        return py::make_tuple(status, py::cast(burst_ptr, py::return_value_policy::reference));
      },
      "Get RX burst from any queue on specific port",
      "port"_a);

  m.def(
      "get_rx_burst",
      []() {
        BurstParams* burst_ptr = nullptr;
        Status status = get_rx_burst(&burst_ptr);
        return py::make_tuple(status, py::cast(burst_ptr, py::return_value_policy::reference));
      },
      "Get RX burst from any queue on any port");

  // Header setting functions
  m.def("set_eth_header",
        py::overload_cast<BurstParams*, int, char*>(&set_eth_header),
        "Set Ethernet header in packet",
        "burst"_a,
        "idx"_a,
        "dst_addr"_a);
  m.def("set_ipv4_header",
        py::overload_cast<BurstParams*, int, int, uint8_t, unsigned int, unsigned int>(
            &set_ipv4_header),
        "Set IPv4 header in packet",
        "burst"_a,
        "idx"_a,
        "ip_len"_a,
        "proto"_a,
        "src_host"_a,
        "dst_host"_a);
  m.def("set_udp_header",
        py::overload_cast<BurstParams*, int, int, uint16_t, uint16_t>(&set_udp_header),
        "Set UDP header in packet",
        "burst"_a,
        "idx"_a,
        "udp_len"_a,
        "src_port"_a,
        "dst_port"_a);
  m.def("set_udp_payload",
        py::overload_cast<BurstParams*, int, void*, int>(&set_udp_payload),
        "Set UDP payload in packet",
        "burst"_a,
        "idx"_a,
        "data"_a,
        "len"_a);

  // Burst metadata functions
  m.def("get_num_packets",
        py::overload_cast<BurstParams*>(&get_num_packets),
        "Get number of packets in a burst",
        "burst"_a);
  m.def(
      "get_q_id", py::overload_cast<BurstParams*>(&get_q_id), "Get queue ID of a burst", "burst"_a);
  m.def("set_num_packets",
        py::overload_cast<BurstParams*, int64_t>(&set_num_packets),
        "Set number of packets in a burst",
        "burst"_a,
        "num"_a);
  m.def("set_header",
        py::overload_cast<BurstParams*, uint16_t, uint16_t, int64_t, int>(&set_header),
        "Set parameters of burst header",
        "burst"_a,
        "port"_a,
        "q"_a,
        "num"_a,
        "segs"_a);

  // Memory management functions
  m.def("free_all_segment_packets",
        py::overload_cast<BurstParams*, int>(&free_all_segment_packets),
        "Free all packets in a burst for one segment",
        "burst"_a,
        "seg"_a);
  m.def("free_all_packets_and_burst_rx",
        py::overload_cast<BurstParams*>(&free_all_packets_and_burst_rx),
        "Free all packets and burst structure for RX",
        "burst"_a);
  m.def("free_all_packets_and_burst_tx",
        py::overload_cast<BurstParams*>(&free_all_packets_and_burst_tx),
        "Free all packets and burst structure for TX",
        "burst"_a);
  m.def("free_segment_packets_and_burst",
        py::overload_cast<BurstParams*, int>(&free_segment_packets_and_burst),
        "Free all packets and burst structure for one packet segment",
        "burst"_a,
        "seg"_a);
  m.def(
      "free_tx_burst", py::overload_cast<BurstParams*>(&free_tx_burst), "Free TX burst", "burst"_a);
  m.def(
      "free_rx_burst", py::overload_cast<BurstParams*>(&free_rx_burst), "Free RX burst", "burst"_a);

  // Network interface functions
  m.def("get_mac_addr",
        py::overload_cast<int, char*>(&get_mac_addr),
        "Get MAC address of an interface",
        "port"_a,
        "mac"_a);
  m.def("get_port_id",
        py::overload_cast<const std::string&>(&get_port_id),
        "Get port number from interface name",
        "key"_a);

  // Utility functions
  m.def("shutdown", &shutdown, "Shut down the advanced_network manager");
  m.def("print_stats", &print_stats, "Print statistics from the advanced_network manager");

  // Bind utility functions for string/enum conversion
  m.def("manager_type_from_string",
        &manager_type_from_string,
        "Convert string to manager type",
        "str"_a);
  m.def("manager_type_to_string",
        &manager_type_to_string,
        "Convert manager type to string",
        "type"_a);
}
};  // namespace holoscan::advanced_network
