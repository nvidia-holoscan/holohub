#include "adv_network_yaml.h"
#include "adv_network_mgr.h"

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
    tmr.access_ = holoscan::ops::GetMemoryAccessPropertiesFromList(mr["access"]);
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