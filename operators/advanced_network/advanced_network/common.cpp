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

#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "advanced_network/manager.h"
#include "advanced_network/common.h"
#include "holoscan/holoscan.hpp"
#if ANO_MGR_DPDK || ANO_MGR_GPUNETIO
#include <rte_mbuf.h>
#include <rte_memcpy.h>
#include <rte_ethdev.h>
#endif
#if ANO_MGR_RIVERMAX
#include "adv_network_rmax_mgr.h"
#endif

#define ASSERT_ANO_MGR_INITIALIZED() \
  assert(g_ano_mgr != nullptr && "ANO Manager is not initialized")
namespace holoscan::ops {

/**
 * @brief Structure for passing packets to/from advanced network operator
 *
 * AdvNetBurstParams is populated by the RX advanced network operator before arriving at the user's
 * operator, and the user populates it prior to sending to the TX advanced network operator. The
 * structure describes metadata about a packet batch and its packet pointers.
 *
 */
// Declare a static global variable for the manager
static ANOMgr* g_ano_mgr = nullptr;

const std::unordered_map<AnoLogLevel::Level, std::string> AnoLogLevel::level_to_string_map = {
    {TRACE, "trace"},
    {DEBUG, "debug"},
    {INFO, "info"},
    {WARN, "warn"},
    {ERROR, "error"},
    {CRITICAL, "critical"},
    {OFF, "off"},
};

const std::unordered_map<std::string, AnoLogLevel::Level> AnoLogLevel::string_to_level_map = {
    {"trace", TRACE},
    {"debug", DEBUG},
    {"info", INFO},
    {"warn", WARN},
    {"error", ERROR},
    {"critical", CRITICAL},
    {"off", OFF},
};

AdvNetBurstParams* adv_net_create_burst_params() {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->create_burst_params();
}

void adv_net_initialize_manager(ANOMgr* manager) {
  g_ano_mgr = manager;
}

ANOMgr* adv_net_get_active_manager() {
  return g_ano_mgr;
}

AnoMgrType adv_net_get_manager_type() {
  return AnoMgrFactory::get_manager_type();
}

template <typename Config>
AnoMgrType adv_net_get_manager_type(const Config& config) {
  return AnoMgrFactory::get_manager_type(config);
}

template AnoMgrType adv_net_get_manager_type<Config>(const Config&);

void adv_net_free_pkt(AdvNetBurstParams* burst, int pkt) {
  ASSERT_ANO_MGR_INITIALIZED();
  g_ano_mgr->free_pkt(burst, pkt);
}

void adv_net_free_pkt_seg(AdvNetBurstParams* burst, int seg, int pkt) {
  ASSERT_ANO_MGR_INITIALIZED();
  g_ano_mgr->free_pkt_seg(burst, seg, pkt);
}

uint16_t adv_net_get_pkt_len(AdvNetBurstParams* burst, int idx) {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->get_pkt_len(burst, idx);
}


uint16_t adv_net_get_pkt_flow_id(AdvNetBurstParams* burst, int idx) {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->get_pkt_flow_id(burst, idx);
}


uint64_t adv_net_get_burst_tot_byte(AdvNetBurstParams* burst) {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->get_burst_tot_byte(burst);
}

uint16_t adv_net_get_seg_pkt_len(AdvNetBurstParams* burst, int seg, int idx) {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->get_seg_pkt_len(burst, seg, idx);
}

void adv_net_free_all_seg_pkts(AdvNetBurstParams* burst, int seg) {
  ASSERT_ANO_MGR_INITIALIZED();
  g_ano_mgr->free_all_seg_pkts(burst, seg);
}

void adv_net_free_all_burst_pkts(AdvNetBurstParams* burst) {
  ASSERT_ANO_MGR_INITIALIZED();
  g_ano_mgr->free_all_pkts(burst);
}

void adv_net_free_all_pkts_and_burst(AdvNetBurstParams* burst) {
  adv_net_free_all_burst_pkts(burst);
  ASSERT_ANO_MGR_INITIALIZED();
  g_ano_mgr->free_rx_burst(burst);
}

void adv_net_free_seg_pkts_and_burst(AdvNetBurstParams* burst, int seg) {
  ASSERT_ANO_MGR_INITIALIZED();
  g_ano_mgr->free_all_seg_pkts(burst, seg);
  g_ano_mgr->free_rx_burst(burst);
}

void adv_net_format_eth_addr(char* dst, std::string addr) {
  std::istringstream iss(addr);
  std::string byteString;

  uint8_t byte_cnt = 0;
  while (std::getline(iss, byteString, ':')) {
    if (byteString.length() == 2) {
      uint16_t byte = std::stoi(byteString, nullptr, 16);
      dst[byte_cnt++] = static_cast<char>(byte);
    } else {
      HOLOSCAN_LOG_ERROR("Invalid MAC address format: {}", addr);
      dst[0] = 0x00;
    }
  }
}

AdvNetStatus adv_net_get_mac(int port, char* mac) {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->get_mac(port, mac);
}

bool adv_net_tx_burst_available(AdvNetBurstParams* burst) {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->tx_burst_available(burst);
}

int adv_net_address_to_port(const std::string& addr) {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->address_to_port(addr);
}

AdvNetStatus adv_net_get_tx_pkt_burst(AdvNetBurstParams* burst) {
  ASSERT_ANO_MGR_INITIALIZED();
  if (!g_ano_mgr->tx_burst_available(burst)) return AdvNetStatus::NO_FREE_BURST_BUFFERS;
  return g_ano_mgr->get_tx_pkt_burst(burst);
}

AdvNetStatus adv_net_set_eth_hdr(AdvNetBurstParams* burst, int idx, char* dst_addr) {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->set_eth_hdr(burst, idx, dst_addr);
}

AdvNetStatus adv_net_set_ipv4_hdr(AdvNetBurstParams* burst, int idx, int ip_len, uint8_t proto,
                                  unsigned int src_host, unsigned int dst_host) {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->set_ipv4_hdr(burst, idx, ip_len, proto, src_host, dst_host);
}

AdvNetStatus adv_net_set_udp_hdr(AdvNetBurstParams* burst, int idx, int udp_len, uint16_t src_port,
                                 uint16_t dst_port) {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->set_udp_hdr(burst, idx, udp_len, src_port, dst_port);
}

AdvNetStatus adv_net_set_udp_payload(AdvNetBurstParams* burst, int idx, void* data, int len) {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->set_udp_payload(burst, idx, data, len);
}

AdvNetStatus adv_net_set_pkt_lens(AdvNetBurstParams* burst, int idx,
                                  const std::initializer_list<int>& lens) {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->set_pkt_lens(burst, idx, lens);
}

AdvNetStatus adv_net_set_pkt_tx_time(AdvNetBurstParams* burst, int idx, uint64_t time) {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->set_pkt_tx_time(burst, idx, time);
}

int64_t adv_net_get_num_pkts(AdvNetBurstParams* burst) {
  return burst->hdr.hdr.num_pkts;
}

int64_t adv_net_get_q_id(AdvNetBurstParams* burst) {
  assert(burst != nullptr && "burst is null");
  return burst->hdr.hdr.q_id;
}

void adv_net_set_num_pkts(AdvNetBurstParams* burst, int64_t num) {
  assert(burst != nullptr && "burst is null");
  burst->hdr.hdr.num_pkts = num;
}

void adv_net_set_hdr(AdvNetBurstParams* burst, uint16_t port, uint16_t q, int64_t num, int segs) {
  assert(burst != nullptr && "burst is null");
  burst->hdr.hdr.num_pkts = num;
  burst->hdr.hdr.port_id = port;
  burst->hdr.hdr.q_id = q;
  burst->hdr.hdr.num_segs = segs;
}

void adv_net_free_tx_burst(AdvNetBurstParams* burst) {
  ASSERT_ANO_MGR_INITIALIZED();
  g_ano_mgr->free_tx_burst(burst);
}

void adv_net_free_rx_burst(AdvNetBurstParams* burst) {
  ASSERT_ANO_MGR_INITIALIZED();
  g_ano_mgr->free_rx_burst(burst);
}

void* adv_net_get_seg_pkt_ptr(AdvNetBurstParams* burst, int seg, int idx) {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->get_seg_pkt_ptr(burst, seg, idx);
}

void* adv_net_get_pkt_ptr(AdvNetBurstParams* burst, int idx) {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->get_pkt_ptr(burst, idx);
}

std::optional<uint16_t> adv_net_get_port_from_ifname(const std::string& name) {
  ASSERT_ANO_MGR_INITIALIZED();
  return g_ano_mgr->get_port_from_ifname(name);
}

void adv_net_shutdown() {
  ASSERT_ANO_MGR_INITIALIZED();
  g_ano_mgr->shutdown();
}

void adv_net_print_stats() {
  ASSERT_ANO_MGR_INITIALIZED();
  g_ano_mgr->print_stats();
}

std::unordered_set<std::string> adv_net_get_port_names(const Config& conf, const std::string& dir) {
  std::unordered_set<std::string> output_ports;
  std::string default_output_name;

  if (dir == "rx") {
    default_output_name = "bench_rx_out";
  } else if (dir == "tx") {
    default_output_name = "bench_tx_out";
  } else {
    return output_ports;
  }

  try {
    auto& yaml_nodes = conf.yaml_nodes();
    for (const YAML::Node& node : yaml_nodes) {
      const auto& intfs = node["advanced_network"]["cfg"]["interfaces"];
      for (const auto& intf : intfs) {
        try {
          const auto& intf_dir = intf[dir];
          for (const auto& q_item : intf_dir["queues"]) {
            auto out_port_name = q_item["output_port"].as<std::string>(default_output_name);
            output_ports.insert(out_port_name);
          }
        } catch (const std::exception& e) {
          continue;  // No queues defined for this direction
        }
      }
    }
  } catch (const std::exception& e) { GXF_LOG_ERROR(e.what()); }
  return output_ports;
}

};  // namespace holoscan::ops

/**
 * @brief Parse flow configuration from a YAML node.
 *
 * @param flow_item The YAML node containing the flow configuration.
 * @param flow The FlowConfig object to populate.
 * @return true if parsing was successful, false otherwise.
 */
bool YAML::convert<holoscan::ops::AdvNetConfigYaml>::parse_flow_config(
    const YAML::Node& flow_item, holoscan::ops::FlowConfig& flow) {
  try {
    flow.name_ = flow_item["name"].as<std::string>();
    flow.id_ = flow_item["id"].as<int>();
    flow.action_.type_ = holoscan::ops::FlowType::QUEUE;
    flow.action_.id_ = flow_item["action"]["id"].as<int>();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Error parsing FlowConfig: {}", e.what());
    return false;
  }

  try {
    flow.match_.udp_src_ = flow_item["match"]["udp_src"].as<uint16_t>();
    flow.match_.udp_dst_ = flow_item["match"]["udp_dst"].as<uint16_t>();
  } catch (const std::exception& e) {
    flow.match_.udp_src_ = 0;
    flow.match_.udp_dst_ = 0;
  }

  try {
    flow.match_.ipv4_len_ = flow_item["match"]["ipv4_len"].as<uint16_t>();
  } catch (const std::exception& e) {
    flow.match_.ipv4_len_ = 0;
  }
  return true;
}

/**
 * @brief Parse memory region configuration from a YAML node.
 *
 * @param mr The YAML node containing the memory region configuration.
 * @param tmr The MemoryRegion object to populate.
 * @return true if parsing was successful, false otherwise.
 */
bool YAML::convert<holoscan::ops::AdvNetConfigYaml>::parse_memory_region_config(
    const YAML::Node& mr, holoscan::ops::MemoryRegion& tmr) {
  try {
    tmr.name_ = mr["name"].as<std::string>();
    tmr.kind_ = holoscan::ops::GetMemoryKindFromString(mr["kind"].template as<std::string>());
    tmr.buf_size_ = mr["buf_size"].as<size_t>();
    tmr.num_bufs_ = mr["num_bufs"].as<size_t>();
    tmr.affinity_ = mr["affinity"].as<uint32_t>();
    try {
      tmr.access_ = holoscan::ops::GetMemoryAccessPropertiesFromList(mr["access"]);
    } catch (const std::exception& e) {
      tmr.access_ = holoscan::ops::MEM_ACCESS_LOCAL;
    }
    tmr.owned_ = mr["owned"].template as<bool>(true);
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Error parsing MemoryRegion: {}", e.what());
    return false;
  }
  return true;
}

/**
 * @brief Parse common queue configuration from a YAML node.
 *
 * @param q_item The YAML node containing the queue configuration.
 * @param common The CommonQueueConfig object to populate.
 * @return true if parsing was successful, false otherwise.
 */
bool parse_common_queue_config(const YAML::Node& q_item, holoscan::ops::CommonQueueConfig& common) {
  try {
    common.name_ = q_item["name"].as<std::string>();
    common.id_ = q_item["id"].as<int>();
    common.cpu_core_ = q_item["cpu_core"].as<std::string>();
    common.batch_size_ = q_item["batch_size"].as<int>();
    common.extra_queue_config_ = nullptr;
    if (q_item["memory_regions"].IsDefined()) {
      const auto& mrs = q_item["memory_regions"];
      if (mrs.size() > 0) { common.mrs_.reserve(mrs.size()); }
      for (const auto& mr : mrs) { common.mrs_.push_back(mr.as<std::string>()); }
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Error parsing CommonQueueConfig: {}", e.what());
    return false;
  }
  if (common.mrs_.empty()) {
    HOLOSCAN_LOG_ERROR("No memory regions defined for queue: {}", common.name_);
    return false;
  }
  return true;
}

/**
 * @brief Parse common RX queue configuration from a YAML node.
 *
 * @param q_item The YAML node containing the RX queue configuration.
 * @param q The RxQueueConfig object to populate.
 * @return true if parsing was successful, false otherwise.
 */
bool YAML::convert<holoscan::ops::AdvNetConfigYaml>::parse_rx_queue_common_config(
    const YAML::Node& q_item, holoscan::ops::RxQueueConfig& q) {
  if (!parse_common_queue_config(q_item, q.common_)) { return false; }
  try {
    q.output_port_ = q_item["output_port"].as<std::string>();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Error parsing RxQueueConfig: {}", e.what());
    return false;
  }
  return true;
}

/**
 * @brief Parse RX queue configuration from a YAML node.
 *
 * @param q_item The YAML node containing the RX queue configuration.
 * @param manager_type The manager type.
 * @param q The RxQueueConfig object to populate.
 * @return true if parsing was successful, false otherwise.
 */
bool YAML::convert<holoscan::ops::AdvNetConfigYaml>::parse_rx_queue_config(
    const YAML::Node& q_item, const holoscan::ops::AnoMgrType& manager_type,
    holoscan::ops::RxQueueConfig& q) {
  try {
    holoscan::ops::AnoMgrType _manager_type = manager_type;

    if (!parse_rx_queue_common_config(q_item, q)) { return false; }

    if (manager_type == holoscan::ops::AnoMgrType::DEFAULT) {
      _manager_type = holoscan::ops::AnoMgrFactory::get_default_manager_type();
    }
#if ANO_MGR_RIVERMAX
    if (_manager_type == holoscan::ops::AnoMgrType::RIVERMAX) {
      holoscan::ops::AdvNetStatus status =
          holoscan::ops::RmaxMgr::parse_rx_queue_rivermax_config(q_item, q);
      if (status != holoscan::ops::AdvNetStatus::SUCCESS) {
        HOLOSCAN_LOG_ERROR("Failed to parse RX Queue config for Rivermax");
        return false;
      }
    }
#endif
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Error parsing RxQueueConfig: {}", e.what());
    return false;
  }
  return true;
}

/**
 * @brief Parse common TX queue configuration from a YAML node.
 *
 * @param q_item The YAML node containing the TX queue configuration.
 * @param q The TxQueueConfig object to populate.
 * @return true if parsing was successful, false otherwise.
 */
bool YAML::convert<holoscan::ops::AdvNetConfigYaml>::parse_tx_queue_common_config(
    const YAML::Node& q_item, holoscan::ops::TxQueueConfig& q) {
  if (!parse_common_queue_config(q_item, q.common_)) { return false; }
  try {
    const auto& offload = q_item["offloads"];
    q.common_.offloads_.reserve(offload.size());
    for (const auto& off : offload) { q.common_.offloads_.push_back(off.as<std::string>()); }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Error parsing TxQueueConfig: {}", e.what());
    return false;
  }
  return true;
}

/**
 * @brief Parse TX queue configuration from a YAML node.
 *
 * @param q_item The YAML node containing the TX queue configuration.
 * @param manager_type The manager type.
 * @param q The TxQueueConfig object to populate.
 * @return true if parsing was successful, false otherwise.
 */
bool YAML::convert<holoscan::ops::AdvNetConfigYaml>::parse_tx_queue_config(
    const YAML::Node& q_item, const holoscan::ops::AnoMgrType& manager_type,
    holoscan::ops::TxQueueConfig& q) {
  try {
    holoscan::ops::AnoMgrType _manager_type = manager_type;

    if (manager_type == holoscan::ops::AnoMgrType::DEFAULT) {
      _manager_type = holoscan::ops::AnoMgrFactory::get_default_manager_type();
    }

    if (!parse_tx_queue_common_config(q_item, q)) { return false; }

#if ANO_MGR_RIVERMAX
    if (_manager_type == holoscan::ops::AnoMgrType::RIVERMAX) {
      holoscan::ops::AdvNetStatus status =
          holoscan::ops::RmaxMgr::parse_tx_queue_rivermax_config(q_item, q);
      if (status != holoscan::ops::AdvNetStatus::SUCCESS) {
        HOLOSCAN_LOG_ERROR("Failed to parse TX Queue config for Rivermax");
        return false;
      }
    }
#endif
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Error parsing TxQueueConfig: {}", e.what());
    return false;
  }
  return true;
}
