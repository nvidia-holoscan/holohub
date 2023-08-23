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

#include "../adv_network_common.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace holoscan::ops {


PYBIND11_MODULE(_advanced_network_common, m) {
    m.doc() = "Advanced networking operator utility functions";

    m.def("adv_net_free_pkt", &adv_net_free_pkt, "Free a single packet");
    m.def("adv_net_get_cpu_packet_len", py::overload_cast<AdvNetBurstParams *, int>(&adv_net_get_cpu_packet_len), "Get length of the CPU portion of the packet");
    m.def("adv_net_get_cpu_packet_len", py::overload_cast<std::shared_ptr<AdvNetBurstParams> &, int>(&adv_net_get_cpu_packet_len), "Get length of the CPU portion of the packet");
    m.def("adv_net_get_gpu_packet_len", py::overload_cast<AdvNetBurstParams *, int>(&adv_net_get_gpu_packet_len), "Get length of the GPU portion of the packet");
    m.def("adv_net_get_gpu_packet_len", py::overload_cast<std::shared_ptr<AdvNetBurstParams> &, int>(&adv_net_get_gpu_packet_len), "Get length of the GPU portion of the packet");
    m.def("adv_net_free_all_burst_pkts", py::overload_cast<AdvNetBurstParams *>(&adv_net_free_all_burst_pkts), "Free all packets in a burst");
    m.def("adv_net_free_all_burst_pkts", py::overload_cast<std::shared_ptr<AdvNetBurstParams> &>(&adv_net_free_all_burst_pkts), "Free all packets in a burst");
    m.def("adv_net_free_all_burst_pkts_and_burst", py::overload_cast<AdvNetBurstParams *>(&adv_net_free_all_burst_pkts_and_burst), "Free all packets and burst structure");
    m.def("adv_net_free_all_burst_pkts_and_burst", py::overload_cast<std::shared_ptr<AdvNetBurstParams> &>(&adv_net_free_all_burst_pkts_and_burst), "Free all packets and burst structure");
    m.def("adv_net_free_cpu_pkts_and_burst", py::overload_cast<AdvNetBurstParams *>(&adv_net_free_cpu_pkts_and_burst), "Free CPU packets and burst structure");
    m.def("adv_net_free_cpu_pkts_and_burst", py::overload_cast<std::shared_ptr<AdvNetBurstParams> &>(&adv_net_free_cpu_pkts_and_burst), "Free CPU packets and burst structure");
    m.def("adv_net_tx_burst_available", &adv_net_tx_burst_available, "Return true if a TX burst is available for use");
    m.def("adv_net_get_tx_pkt_burst", py::overload_cast<AdvNetBurstParams *>(&adv_net_free_cpu_pkts_and_burst), "Get TX packet burst");
    m.def("adv_net_get_tx_pkt_burst", py::overload_cast<std::shared_ptr<AdvNetBurstParams> &>(&adv_net_free_cpu_pkts_and_burst), "Get TX packet burst");
    m.def("adv_net_set_cpu_udp_payload", py::overload_cast<AdvNetBurstParams *, int, void *, int>(&adv_net_set_cpu_udp_payload), "Set UDP header parameters and copy payload");
    m.def("adv_net_set_cpu_udp_payload", py::overload_cast<std::shared_ptr<AdvNetBurstParams> &, int, void *, int>(&adv_net_set_cpu_udp_payload), "Set UDP header parameters and copy payload");
    m.def("adv_net_get_num_pkts", py::overload_cast<AdvNetBurstParams *>(&adv_net_get_num_pkts), "Get number of packets in a burst");
    m.def("adv_net_get_num_pkts", py::overload_cast<std::shared_ptr<AdvNetBurstParams> &>(&adv_net_get_num_pkts), "Get number of packets in a burst");
    m.def("adv_net_set_num_pkts", py::overload_cast<AdvNetBurstParams *, int64_t>(&adv_net_set_num_pkts), "Set number of packets in a burst");
    m.def("adv_net_set_num_pkts", py::overload_cast<std::shared_ptr<AdvNetBurstParams> &, int64_t>(&adv_net_set_num_pkts), "Set number of packets in a burst");
    m.def("adv_net_set_hdr", py::overload_cast<AdvNetBurstParams *, uint16_t, uint16_t, int64_t>(&adv_net_set_hdr), "Set parameters of burst header");
    m.def("adv_net_set_hdr", py::overload_cast<std::shared_ptr<AdvNetBurstParams> &, uint16_t, uint16_t, int64_t>(&adv_net_set_hdr), "Set parameters of burst header");
    m.def("adv_net_free_tx_burst", py::overload_cast<AdvNetBurstParams *>(&adv_net_free_tx_burst), "Free TX burst");
    m.def("adv_net_free_tx_burst", py::overload_cast<std::shared_ptr<AdvNetBurstParams> &>(&adv_net_free_tx_burst), "Free TX burst");
    m.def("adv_net_free_rx_burst", py::overload_cast<AdvNetBurstParams *>(&adv_net_free_rx_burst), "Free RX burst");
    m.def("adv_net_free_rx_burst", py::overload_cast<std::shared_ptr<AdvNetBurstParams> &>(&adv_net_free_rx_burst), "Free RX burst");
    m.def("adv_net_get_cpu_pkt_ptr", py::overload_cast<AdvNetBurstParams *, int>(&adv_net_get_cpu_pkt_ptr), "Get CPU packet pointer");
    m.def("adv_net_get_cpu_pkt_ptr", py::overload_cast<std::shared_ptr<AdvNetBurstParams> &, int>(&adv_net_get_cpu_pkt_ptr), "Get CPU packet pointer");
    m.def("adv_net_get_gpu_pkt_ptr", py::overload_cast<AdvNetBurstParams *, int>(&adv_net_get_gpu_pkt_ptr), "Get GPU packet pointer");
    m.def("adv_net_get_gpu_pkt_ptr", py::overload_cast<std::shared_ptr<AdvNetBurstParams> &, int>(&adv_net_get_gpu_pkt_ptr), "Get GPU packet pointer");
    m.def("adv_net_get_port_from_ifname", &adv_net_get_port_from_ifname, "Get port number from interface name");
    m.def("adv_net_free_pkts", [](void *pkts, int len) { adv_net_free_pkts(reinterpret_cast<void**>(pkts), len); }, "Frees a list of packets");
}

};  // namespace holoscan::ops
