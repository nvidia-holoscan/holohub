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

#ifndef RIVERMAX_CONFIG_MANAGER_H_
#define RIVERMAX_CONFIG_MANAGER_H_

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "advanced_network/manager.h"
#include "rivermax_ano_data_types.h"
#include "rivermax_queue_configs.h"

namespace holoscan::advanced_network {

/**
 * @brief: Base configuration holder class.
 *
 * This class serves as the base for all configuration holder types
 * and provides a common interface for identifying the configuration type.
 */
class ConfigBuilderHolder {
 public:
  /**
   * @brief: Virtual destructor for ConfigBuilderHolder.
   */
  virtual ~ConfigBuilderHolder() = default;

  /**
   * @brief: Returns the type of the configuration.
   *
   * @return: The configuration type enum value.
   */
  virtual QueueConfigType get_type() const = 0;
};

/**
 * @brief: Typed configuration holder for specific builder types.
 *
 * This template class holds a builder of a specific type along with
 * its type enum for identification purposes.
 *
 * @tparam ConfigBuilderType: The type of the configuration builder to hold.
 */
template <typename ConfigBuilderType>
class TypedConfigBuilderHolder : public ConfigBuilderHolder {
 public:
  /**
   * @brief: Constructor for TypedConfigBuilderHolder.
   *
   * @param [in] type: The configuration type enum value.
   * @param [in] config_builder: A shared pointer to the configuration builder.
   */
  TypedConfigBuilderHolder(QueueConfigType type, std::shared_ptr<ConfigBuilderType> config_builder)
      : type_(type), config_builder_(std::move(config_builder)) {}

  /**
   * @brief: Returns the type of the configuration.
   *
   * @return: The configuration type enum value.
   */
  QueueConfigType get_type() const override { return type_; }

  /**
   * @brief: Gets the configuration builder pointer.
   *
   * @return: A shared pointer to the configuration builder.
   */
  std::shared_ptr<ConfigBuilderType> get_config_builder() const { return config_builder_; }

 private:
  QueueConfigType type_;
  std::shared_ptr<ConfigBuilderType> config_builder_;
};

/**
 * @brief: Container for multiple configuration builders.
 *
 * This class provides a mapping from configuration numbers to
 * configuration builder holders, allowing for storage and retrieval
 * of different types of configuration builders.
 */
class ConfigBuilderContainer {
 public:
  /**
   * @brief: Iterator types for the container.
   */
  using ConstIterator =
      typename std::map<uint32_t, std::shared_ptr<ConfigBuilderHolder>>::const_iterator;

  /**
   * @brief: Iterator access methods.
   */
  ConstIterator begin() const { return holders_.begin(); }
  ConstIterator end() const { return holders_.end(); }
  ConstIterator cbegin() const { return holders_.cbegin(); }
  ConstIterator cend() const { return holders_.cend(); }

  /**
   * @brief: Adds a configuration builder to the container.
   *
   * @tparam ConfigBuilderType: The type of the configuration builder.
   * @param [in] config_num: The configuration number (key).
   * @param [in] type: The configuration type enum value.
   * @param [in] config_builder: A shared pointer to the configuration builder.
   */
  template <typename ConfigBuilderType>
  void add_config_builder(uint32_t config_num, QueueConfigType type,
                          std::shared_ptr<ConfigBuilderType> config_builder) {
    holders_[config_num] = std::make_shared<TypedConfigBuilderHolder<ConfigBuilderType>>(
        type, std::move(config_builder));
  }

  /**
   * @brief: Gets a configuration holder by configuration number.
   *
   * @param [in] config_num: The configuration number (key).
   * @return: A shared pointer to the ConfigBuilderHolder, or nullptr if not found.
   */
  std::shared_ptr<ConfigBuilderHolder> get_holder(uint32_t config_num) const {
    auto it = holders_.find(config_num);
    if (it != holders_.end()) { return it->second; }
    return nullptr;
  }

  /**
   * @brief: Gets a configuration builder by configuration number and casts to specific type.
   *
   * @tparam ConfigBuilderType: The type of the configuration builder.
   * @param [in] config_num: The configuration number (key).
   * @return: A shared pointer to the specific type of configuration builder, or nullptr if not
   * found.
   */
  template <typename ConfigBuilderType>
  std::shared_ptr<ConfigBuilderType> get_config_builder(uint32_t config_num) const {
    auto holder = get_holder(config_num);
    if (holder) {
      auto typed_holder = dynamic_cast<TypedConfigBuilderHolder<ConfigBuilderType>*>(holder.get());
      if (typed_holder) { return typed_holder->get_config_builder(); }
    }
    return nullptr;
  }

  /**
   * @brief: Gets configuration builders by type.
   *
   * @tparam ConfigBuilderType: The type of the configuration builder.
   * @param [in] type: The configuration type enum value.
   * @return: A vector of pairs containing config number and builder pointer.
   */
  template <typename ConfigBuilderType>
  std::vector<std::pair<uint32_t, std::shared_ptr<ConfigBuilderType>>> get_config_builders_by_type(
      QueueConfigType type) const {
    std::vector<std::pair<uint32_t, std::shared_ptr<ConfigBuilderType>>> result;
    for (const auto& pair : holders_) {
      if (pair.second->get_type() == type) {
        auto typed_holder =
            dynamic_cast<TypedConfigBuilderHolder<ConfigBuilderType>*>(pair.second.get());
        if (typed_holder) { result.emplace_back(pair.first, typed_holder->get_config_builder()); }
      }
    }
    return result;
  }

  /**
   * @brief: Checks if the container has a configuration with the given number.
   *
   * @param [in] config_num: The configuration number to check.
   * @return: True if the configuration exists, false otherwise.
   */
  bool has_config(uint32_t config_num) const {
    return holders_.find(config_num) != holders_.end();
  }

  /**
   * @brief: Gets all configuration numbers in the container.
   *
   * @return: A vector of configuration numbers.
   */
  std::vector<uint32_t> get_config_nums() const {
    std::vector<uint32_t> keys;
    keys.reserve(holders_.size());
    for (const auto& pair : holders_) { keys.push_back(pair.first); }
    return keys;
  }

  /**
   * @brief: Gets the number of configurations in the container.
   *
   * @return: The number of configurations.
   */
  size_t size() const { return holders_.size(); }

  /**
   * @brief: Clears all configurations from the container.
   */
  void clear() { holders_.clear(); }

 private:
  std::map<uint32_t, std::shared_ptr<ConfigBuilderHolder>> holders_;
};

/**
 * @brief Interface for configuration managers.
 *
 * The IConfigManager interface provides a common set of operations for
 * managing configurations. It defines methods for setting configurations,
 * iterating through configuration entries, and accessing specific configurations.
 * This serves as the base interface for specialized configuration managers like
 * RxConfigManager and TxConfigManager.
 */
class IConfigManager {
 public:
  /**
   * @brief Maximum number of Rivermax memory regions supported.
   */
  static constexpr uint16_t MAX_RMAX_MEMORY_REGIONS = 2;

  /**
   * @brief Type definition for constant iterators used to traverse configurations.
   */
  using ConstIterator = typename ConfigBuilderContainer::ConstIterator;

  /**
   * @brief Virtual destructor to ensure proper cleanup of derived classes.
   */
  virtual ~IConfigManager() = default;

  /**
   * @brief Sets the network configuration.
   *
   * @param [in] cfg The network configuration to set.
   * @return true if the configuration was set successfully, false otherwise.
   */
  virtual bool set_configuration(const NetworkConfig& cfg) = 0;

  /**
   * @brief Gets an iterator to the beginning of the configuration collection.
   *
   * @return A constant iterator pointing to the beginning of the collection.
   */
  virtual ConstIterator begin() const = 0;

  /**
   * @brief Gets an iterator to the end of the configuration collection.
   *
   * @return A constant iterator pointing to the end of the collection.
   */
  virtual ConstIterator end() const = 0;

  /**
   * @brief Gets a constant iterator to the beginning of the configuration collection.
   *
   * This method explicitly returns a constant iterator, useful when the const
   * nature of the access needs to be emphasized.
   *
   * @return A constant iterator pointing to the beginning of the collection.
   */
  virtual ConstIterator cbegin() const = 0;

  /**
   * @brief Gets a constant iterator to the end of the configuration collection.
   *
   * This method explicitly returns a constant iterator, useful when the const
   * nature of the access needs to be emphasized.
   *
   * @return A constant iterator pointing to the end of the collection.
   */
  virtual ConstIterator cend() const = 0;
};

/**
 * @brief Interface for RX configuration managers.
 *
 * The IRxConfigManager interface extends IConfigManager and defines additional operations
 * specific to RX configuration managers in Rivermax.
 */
class IRxConfigManager : public IConfigManager {
 public:
  /**
   * @brief Appends a candidate for RX queue configuration.
   *
   * @param port_id The port ID.
   * @param q The RX queue configuration.
   * @return True if the configuration was appended successfully, false otherwise.
   */
  virtual bool append_candidate_for_rx_queue(uint16_t port_id, const RxQueueConfig& q) = 0;
};

/**
 * @brief Interface for TX configuration managers.
 *
 * The ITxConfigManager interface extends IConfigManager and defines additional operations
 * specific to TX configuration managers in Rivermax.
 */
class ITxConfigManager : public IConfigManager {
 public:
  /**
   * @brief Appends a candidate for TX queue configuration.
   *
   * @param port_id The port ID.
   * @param q The TX queue configuration.
   * @return True if the configuration was appended successfully, false otherwise.
   */
  virtual bool append_candidate_for_tx_queue(uint16_t port_id, const TxQueueConfig& q) = 0;
};

/**
 * @brief Manages the RX configuration for Rivermax.
 *
 * The RxConfigManager class is responsible for managing the configuration settings
 * for RX queues in Rivermax. It validates and appends RX queue configurations for given ports.
 */
class RxConfigManager : public IRxConfigManager {
 public:
  // Regular const iterators
  ConstIterator begin() const override { return config_builder_container_.begin(); }
  ConstIterator end() const override { return config_builder_container_.end(); }
  // Explicit const iterators
  ConstIterator cbegin() const override { return config_builder_container_.cbegin(); }
  ConstIterator cend() const override { return config_builder_container_.cend(); }

  bool set_configuration(const NetworkConfig& cfg) override {
    cfg_ = cfg;
    is_configuration_set_ = true;
    return true;
  }

  bool append_candidate_for_rx_queue(uint16_t port_id, const RxQueueConfig& q) override;

 private:
  /**
   * @brief Appends a candidate for RX queue configuration.
   *
   * @param config_index The configuration index.
   * @param q The RX queue configuration.
   * @return True if the configuration was appended successfully, false otherwise.
   */
  bool append_ipo_receiver_candidate_for_rx_queue(
      uint32_t config_index, const RxQueueConfig& q,
      RivermaxIPOReceiverQueueConfig& rivermax_rx_config);
  /**
   * @brief Appends a candidate for RX queue configuration.
   *
   * @param config_index The configuration index.
   * @param q The RX queue configuration.
   * @return True if the configuration was appended successfully, false otherwise.
   */
  bool append_rtp_receiver_candidate_for_rx_queue(
      uint32_t config_index, const RxQueueConfig& q,
      RivermaxRTPReceiverQueueConfig& rivermax_rx_config);
  /**
   * @brief Configures the memory allocator for the Rivermax RX queue.
   *
   * @param rivermax_rx_config The Rivermax RX queue configuration.
   * @param q The RX queue configuration.
   * @return true if the configuration is successful, false otherwise.
   */
  bool config_memory_allocator(RivermaxCommonRxQueueConfig& rivermax_rx_config,
                               const RxQueueConfig& q);

  /**
   * @brief Configures the memory allocator for a single memory region.
   *
   * Configures the memory allocator for a single memory region.
   * The allocator will be used for both the header and payload memory.
   *
   * @param rivermax_rx_config The Rivermax RX queue configuration.
   * @param mr The memory region.
   * @return true if the configuration is successful, false otherwise.
   */
  bool config_memory_allocator_from_single_mrs(RivermaxCommonRxQueueConfig& rivermax_rx_config,
                                               const MemoryRegionConfig& mr);

  /**
   * @brief Configures the memory allocator for dual memory regions.
   *
   * Configures the memory allocator for dual memory regions.
   * If GPU is in use, it will be used for the payload memory region,
   * and the CPU allocator will be used for the header memory region.
   * Otherwise, the function expects that the same allocator is configured
   * for both memory regions.
   *
   * @param rivermax_rx_config The Rivermax RX queue configuration.
   * @param mr_header The header memory region.
   * @param mr_payload The payload memory region.
   * @return true if the configuration is successful, false otherwise.
   */

  bool config_memory_allocator_from_dual_mrs(RivermaxCommonRxQueueConfig& rivermax_rx_config,
                                             const MemoryRegionConfig& mr_header,
                                             const MemoryRegionConfig& mr_payload);

 private:
  ConfigBuilderContainer config_builder_container_;
  NetworkConfig cfg_;
  bool is_configuration_set_ = false;
};

/**
 * @brief Manages the TX configuration for Rivermax.
 *
 * The TxConfigManager class is responsible for managing the configuration settings
 * for TX queues in Rivermax. It validates and appends TX queue configurations for given ports.
 */
class TxConfigManager : public ITxConfigManager {
 public:
  // Regular const iterators
  ConstIterator begin() const override { return config_builder_container_.begin(); }
  ConstIterator end() const override { return config_builder_container_.end(); }
  // Explicit const iterators
  ConstIterator cbegin() const override { return config_builder_container_.cbegin(); }
  ConstIterator cend() const override { return config_builder_container_.cend(); }

  bool set_configuration(const NetworkConfig& cfg) override {
    cfg_ = cfg;
    is_configuration_set_ = true;
    return true;
  }

  bool append_candidate_for_tx_queue(uint16_t port_id, const TxQueueConfig& q) override;

 private:
  /**
   * @brief Appends a candidate for TX queue configuration.
   *
   * @param config_index The configuration index.
   * @param q The TX queue configuration.
   * @return True if the configuration was appended successfully, false otherwise.
   */
  bool append_media_sender_candidate_for_tx_queue(
      uint32_t config_index, const TxQueueConfig& q,
      RivermaxMediaSenderQueueConfig& rivermax_tx_config);
  /**
   * @brief Configures the memory allocator for the Rivermax TX queue.
   *
   * @param rivermax_rx_config The Rivermax TX queue configuration.
   * @param q The TX queue configuration.
   * @return true if the configuration is successful, false otherwise.
   */
  bool config_memory_allocator(RivermaxCommonTxQueueConfig& rivermax_tx_config,
                               const TxQueueConfig& q);

  /**
   * @brief Configures the memory allocator for a single memory region.
   *
   * Configures the memory allocator for a single memory region.
   * The allocator will be used for both the header and payload memory.
   *
   * @param rivermax_rx_config The Rivermax TX queue configuration.
   * @param mr The memory region.
   * @return true if the configuration is successful, false otherwise.
   */
  bool config_memory_allocator_from_single_mrs(RivermaxCommonTxQueueConfig& rivermax_tx_config,
                                               const MemoryRegionConfig& mr);

  /**
   * @brief Configures the memory allocator for dual memory regions.
   *
   * Configures the memory allocator for dual memory regions.
   * If GPU is in use, it will be used for the payload memory region,
   * and the CPU allocator will be used for the header memory region.
   * Otherwise, the function expects that the same allocator is configured
   * for both memory regions.
   *
   * @param rivermax_rx_config The Rivermax TX queue configuration.
   * @param mr_header The header memory region.
   * @param mr_payload The payload memory region.
   * @return true if the configuration is successful, false otherwise.
   */

  bool config_memory_allocator_from_dual_mrs(RivermaxCommonTxQueueConfig& rivermax_tx_config,
                                             const MemoryRegionConfig& mr_header,
                                             const MemoryRegionConfig& mr_payload);

 private:
  ConfigBuilderContainer config_builder_container_;
  NetworkConfig cfg_;
  bool is_configuration_set_ = false;
};

/**
 * @brief Manages the configuration for Rivermax.
 *
 * The RivermaxConfigContainer class is responsible for parsing and managing the configuration
 * settings for Rivermax via dedicated configuration managers.
 */
class RivermaxConfigContainer {
 public:
  enum class ConfigType { RX, TX };

  /**
   * @brief Constructs a new RivermaxConfigContainer object.
   */
  RivermaxConfigContainer() { initialize_managers(); }

  /**
   * @brief Parses the configuration from the YAML file.
   *
   * This function iterates over the interfaces and their respective RX an TX queues
   * defined in the configuration YAML, extracting and validating the necessary
   * settings for each queue. It then populates the RX and TX service configuration
   * structures with these settings. The parsing is done via dedicated configuration managers.
   *
   * @param cfg The configuration YAML.
   * @return True if the configuration was successfully parsed, false otherwise.
   */
  bool parse_configuration(const NetworkConfig& cfg);

  std::shared_ptr<IConfigManager> get_config_manager(ConfigType type) const {
    auto it = config_managers_.find(type);
    if (it != config_managers_.end()) { return it->second; }
    return nullptr;
  }

  /**
   * @brief Gets the current log level for Rivermax.
   *
   * @return The current log level.
   */
  RivermaxLogLevel::Level get_rivermax_log_level() const { return rivermax_log_level_; }

 private:
  /**
   * @brief Initializes the configuration managers.
   *
   * This function initializes the configuration managers for RX and TX services.
   */
  void initialize_managers();

  /**
   * @brief Adds a configuration manager.
   *
   * This function adds a configuration manager for the specified type.
   *
   * @param type The type of configuration manager to add.
   * @param config_manager The shared pointer to the configuration manager.
   */
  void add_config_manager(ConfigType type, std::shared_ptr<IConfigManager> config_manager);

  /**
   * @brief Parses the RX queues configuration.
   *
   * This function parses the configuration for RX queues for the specified port ID.
   *
   * @param port_id The port ID for which to parse the RX queues.
   * @param queues The vector of RX queue configurations.
   * @return An integer indicating the success or failure of the parsing operation.
   */
  int parse_rx_queues(uint16_t port_id, const std::vector<RxQueueConfig>& queues);

  /**
   * @brief Parses the TX queues configuration.
   *
   * This function parses the configuration for TX queues for the specified port ID.
   *
   * @param port_id The port ID for which to parse the TX queues.
   * @param queues The vector of TX queue configurations.
   * @return An integer indicating the success or failure of the parsing operation.
   */
  int parse_tx_queues(uint16_t port_id, const std::vector<TxQueueConfig>& queues);

  /**
   * @brief Sets the Rivermax log level based on the provided advanced_network log level.
   *
   * This function converts the provided advanced_network log level to the corresponding
   * Rivermax log level and sets it as the current log level for Rivermax.
   *
   * @param level The advanced_network log level to be converted and set.
   */
  void set_rivermax_log_level(LogLevel::Level level) {
    rivermax_log_level_ = RivermaxLogLevel::from_adv_net_log_level(level);
  }

 private:
  RivermaxLogLevel::Level rivermax_log_level_ = RivermaxLogLevel::OFF;
  std::unordered_map<ConfigType, std::shared_ptr<IConfigManager>> config_managers_;
  NetworkConfig cfg_;
  bool is_configured_ = false;
};

/**
 * @brief Parses the configuration for Rivermax queues from YAML.
 *
 * The RivermaxConfigParser class is a static utility class responsible for parsing
 * configuration settings for Rivermax from YAML nodes. It handles both RX and TX
 * queue configurations, supporting multiple receiver types (IPO, RTP) and sender types
 * (media sender, generic sender).
 *
 * This class separates the parsing logic into common settings shared across all queue types
 * and specialized settings for specific receiver/sender implementations. It works together
 * with RxConfigManager and TxConfigManager which use the parsed configurations to build
 * the actual Rivermax service configurations.
 *
 * All methods in this class are static and do not require instantiation of the class.
 */
class RivermaxConfigParser {
 public:
  /**
   * @brief Parses the RX queue configuration from a YAML node.
   *
   * This function extracts the RX queue configuration settings from the provided YAML node
   * and populates the RxQueueConfig structure with the extracted values.
   *
   * @param q_item The YAML node containing the RX queue configuration.
   * @param q The RxQueueConfig structure to be populated.
   * @return Status indicating the success or failure of the operation.
   */
  static Status parse_rx_queue_rivermax_config(const YAML::Node& q_item, RxQueueConfig& q);

  /**
   * @brief Parses common RX settings from a YAML node.
   *
   * This function extracts common RX configuration settings from the provided YAML node
   * and populates the RivermaxCommonRxQueueConfig structure with the extracted values.
   * These settings are shared across all RX queue types.
   *
   * @param [in] rx_settings The YAML node containing the RX settings.
   * @param [in] q_item The YAML node containing the queue item.
   * @param [out] rivermax_rx_config The common RX queue configuration to be populated.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_common_rx_settings(const YAML::Node& rx_settings, const YAML::Node& q_item,
                                       RivermaxCommonRxQueueConfig& rivermax_rx_config);

  /**
   * @brief Parses IPO receiver specific settings from a YAML node.
   *
   * This function extracts IPO receiver specific configuration settings from the provided YAML node
   * and populates the RivermaxIPOReceiverQueueConfig structure with the extracted values.
   * These settings include network addresses, ports, and IPO-specific parameters.
   *
   * @param [in] rx_settings The YAML node containing the RX settings.
   * @param [out] rivermax_rx_config The IPO receiver RX queue configuration to be populated.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_ipo_receiver_settings(const YAML::Node& rx_settings,
                                          RivermaxIPOReceiverQueueConfig& rivermax_rx_config);

  /**
   * @brief Parses RTP receiver specific settings from a YAML node.
   *
   * This function extracts RTP receiver specific configuration settings from the provided YAML node
   * and populates the RivermaxRTPReceiverQueueConfig structure with the extracted values.
   * These settings include network addresses and port parameters specific to RTP.
   *
   * @param [in] rx_settings The YAML node containing the RX settings.
   * @param [out] rivermax_rx_config The RTP receiver RX queue configuration to be populated.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_rtp_receiver_settings(const YAML::Node& rx_settings,
                                          RivermaxRTPReceiverQueueConfig& rivermax_rx_config);

  /**
   * @brief Parses the TX queue Rivermax configuration.
   *
   * This function extracts the TX queue configuration settings from the provided YAML node
   * and populates the TxQueueConfig structure with the extracted values.
   *
   * @param q_item The YAML node containing the queue item.
   * @param q The TX queue configuration to be populated.
   * @return Status indicating the success or failure of the operation.
   */
  static Status parse_tx_queue_rivermax_config(const YAML::Node& q_item, TxQueueConfig& q);

  /**
   * @brief Parses common TX settings from a YAML node.
   *
   * This function extracts common TX configuration settings from the provided YAML node
   * and populates the RivermaxCommonTxQueueConfig structure with the extracted values.
   * These settings are shared across all TX queue types.
   *
   * @param [in] tx_settings The YAML node containing the TX settings.
   * @param [in] q_item The YAML node containing the queue item.
   * @param [out] rivermax_tx_config The common TX queue configuration to be populated.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_common_tx_settings(const YAML::Node& tx_settings, const YAML::Node& q_item,
                                       RivermaxCommonTxQueueConfig& rivermax_tx_config);

  /**
   * @brief Parses media sender specific settings from a YAML node.
   *
   * This function extracts media sender specific configuration settings from the provided YAML node
   * and populates the RivermaxMediaSenderQueueConfig structure with the extracted values.
   * These settings include video format, bit depth, frame dimensions, and frame rate.
   *
   * @param [in] tx_settings The YAML node containing the TX settings.
   * @param [out] rivermax_tx_config The media sender TX queue configuration to be populated.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_media_sender_settings(const YAML::Node& tx_settings,
                                          RivermaxMediaSenderQueueConfig& rivermax_tx_config);
};

class ConfigManagerUtilities {
 public:
  template <typename T>
  static void set_cpu_allocator_type(T& rivermax_config, const MemoryRegionConfig& mr);

  template <typename T>
  static void set_gpu_is_not_in_use(T& rivermax_config);

  template <typename T>
  static bool set_gpu_is_in_use_if_applicable(T& rivermax_config, const MemoryRegionConfig& mr);

  static bool parse_and_set_cores(std::vector<int>& app_threads_cores, const std::string& cores);

  static bool validate_cores(const std::string& cores);

  static void set_allocator_type(AppSettings& app_settings_config,
                                 const std::string& allocator_type);

  static VideoSampling convert_video_sampling(const std::string& sampling);

  static ColorBitDepth convert_bit_depth(uint16_t bit_depth);

  static bool validate_memory_regions_config(
      const std::vector<std::string>& queue_mr_names,
      const std::unordered_map<std::string, MemoryRegionConfig>& memory_regions);

  static bool validate_memory_regions_config_from_single_mrs(const MemoryRegionConfig& mr);

  static bool validate_memory_regions_config_from_dual_mrs(const MemoryRegionConfig& mr_header,
                                                           const MemoryRegionConfig& mr_payload);
};

}  // namespace holoscan::advanced_network

#endif  // RIVERMAX_CONFIG_MANAGER_H_
