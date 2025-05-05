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

namespace py = pybind11;

namespace holoscan::advanced_network {

PYBIND11_MODULE(_advanced_network_common, m) {
  m.doc() = "Advanced Network utility functions";

  py::enum_<Status>(m, "Status")
      .value("SUCCESS", Status::SUCCESS)
      .value("NULL_PTR", Status::NULL_PTR)
      .value("NO_FREE_BURST_BUFFERS", Status::NO_FREE_BURST_BUFFERS)
      .value("NO_FREE_PACKET_BUFFERS", Status::NO_FREE_PACKET_BUFFERS);

  m.def("create_tx_burst_params",
        &create_tx_burst_params,
        py::return_value_policy::reference,
        "Create a shared pointer burst params structure");
  m.def("get_segment_packet_length",
        py::overload_cast<BurstParams*, int, int>(&get_segment_packet_length),
        "Get length of one segments of the packet");
  m.def("get_packet_length",
        py::overload_cast<BurstParams*, int>(&get_packet_length),
        "Get length of the packet");
  m.def("free_all_segment_packets",
        py::overload_cast<BurstParams*, int>(&free_all_segment_packets),
        "Free all packets in a burst for one segment");
  m.def("free_all_packets_and_burst_rx",
        py::overload_cast<BurstParams*>(&free_all_packets_and_burst_rx),
        "Free all packets and burst structure for RX");
  m.def("free_all_packets_and_burst_tx",
        py::overload_cast<BurstParams*>(&free_all_packets_and_burst_tx),
        "Free all packets and burst structure for TX");
  m.def("free_segment_packets_and_burst",
        py::overload_cast<BurstParams*, int>(&free_segment_packets_and_burst),
        "Free all packets and burst structure for one packet segment");
  m.def("tx_burst_available",
        py::overload_cast<BurstParams*>(&is_tx_burst_available),
        "Return true if a TX burst is available for use");
  m.def("get_tx_packet_burst",
        py::overload_cast<BurstParams*>(&get_tx_packet_burst),
        "Get TX packet burst");
  m.def("shutdown", (&shutdown), "Shut down the advanced_network manager");
  m.def("print_stats", (&print_stats), "Print statistics from the advanced_network manager");
  // m.def("set_cpu_udp_payload",
  //     [](BurstParams *burst, int idx, long int data, int len) {
  //             return set_cpu_udp_payload(burst, idx,
  //                     reinterpret_cast<void*>(data), len); },
  //             "Set UDP header parameters and copy payload");
  // m.def("set_cpu_udp_payload",
  //     [](std::shared_ptr<BurstParams> burst, int idx, long int data, int len) {
  //         return set_cpu_udp_payload(burst, idx,
  //              reinterpret_cast<void*>(data), len); },
  //         "Set UDP header parameters and copy payload");

  m.def("get_num_packets",
        py::overload_cast<BurstParams*>(&get_num_packets),
        "Get number of packets in a burst");
  m.def("get_q_id",
        py::overload_cast<BurstParams*>(&get_q_id),
        "Get queue ID of a burst");
  m.def("set_num_packets",
        py::overload_cast<BurstParams*, int64_t>(&set_num_packets),
        "Set number of packets in a burst");
  m.def("set_header",
        py::overload_cast<BurstParams*, uint16_t, uint16_t, int64_t, int>(&set_header),
        "Set parameters of burst header");
  m.def("free_tx_burst",
        py::overload_cast<BurstParams*>(&free_tx_burst),
        "Free TX burst");
  m.def("free_rx_burst",
        py::overload_cast<BurstParams*>(&free_rx_burst),
        "Free RX burst");
  m.def("get_segment_packet_ptr",
        py::overload_cast<BurstParams*, int, int>(&get_segment_packet_ptr),
        "Get packet pointer for one segment");
  m.def("get_packet_ptr",
        py::overload_cast<BurstParams*, int>(&get_packet_ptr),
        "Get packet pointer");
  m.def("get_rx_burst", [](int port, int q) {
      BurstParams* burst_ptr = nullptr;
      Status status = get_rx_burst(&burst_ptr, port, q);
      return py::make_tuple(status, py::cast(burst_ptr,
            py::return_value_policy::take_ownership));
      }, py::arg("port"), py::arg("q"));

  // py::class_<BurstHeaderParams>(m, "BurstHeaderParams").def(py::init<>())
  //     .def_readwrite("num_pkts",  &BurstHeaderParams::num_pkts)
  //     .def_readwrite("port_id",   &BurstHeaderParams::port_id)
  //     .def_readwrite("q_id",      &BurstHeaderParams::q_id);

  // py::class_<BurstHeader>(m, "BurstHeader").def(py::init<>())
  //     .def_readwrite("hdr",  &BurstHeader::hdr);

  // py::class_<BurstParams>(m, "BurstParams").def(py::init<>())
  //     .def_readwrite("hdr", &BurstParams::hdr)
  //     .def_readwrite("cpu_pkts", &BurstParams::cpu_pkts)
  //     .def_readwrite("gpu_pkts", &BurstParams::gpu_pkts);

  py::class_<BurstHeaderParams>(m, "BurstHeaderParams").def(py::init<>());
  py::class_<BurstHeader>(m, "BurstHeader").def(py::init<>());
  py::class_<BurstParams>(m, "BurstParams").def(py::init<>());
  //  py::class_<BurstParams, std::shared_ptr<BurstParams>>
  //    (m, "BurstParams").def(py::init<>());
}
};  // namespace holoscan::advanced_network
