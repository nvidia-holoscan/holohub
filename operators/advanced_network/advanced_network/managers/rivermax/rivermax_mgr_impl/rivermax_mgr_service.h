/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RIVERMAX_MANAGER_SERVICE_H_
#define RIVERMAX_MANAGER_SERVICE_H_

#include <memory>
#include <cstdint>
#include <sstream>

#include <holoscan/logger/logger.hpp>

#include "rdk/rivermax_dev_kit.h"
#include "rdk/apps/rmax_xstream_media_sender/rmax_xstream_media_sender.h"
#include "rdk/apps/rmax_ipo_receiver/rmax_ipo_receiver.h"
#include "rdk/apps/rmax_rtp_receiver/rmax_rtp_receiver.h"

#include "adv_network_rivermax_mgr.h"
#include "rivermax_mgr_impl/rivermax_config_manager.h"
#include "rivermax_mgr_impl/burst_manager.h"
#include "rivermax_mgr_impl/packet_processor.h"

namespace holoscan::advanced_network {

using namespace rivermax::dev_kit::apps;
using namespace rivermax::dev_kit::apps::rmax_xstream_media_sender;
using namespace rivermax::dev_kit::apps::rmax_ipo_receiver;
using namespace rivermax::dev_kit::apps::rmax_rtp_receiver;

/**
 * @brief Base class for all Rivermax manager services.
 *
 * This class defines the common interface and functionality for all Rivermax manager services,
 * providing lifecycle management methods and service identification.
 */
class RivermaxManagerService {
 public:
  /**
   * @brief Constructor for the RivermaxManagerService class.
   *
   * @param service_id Unique identifier for the service.
   */
  explicit RivermaxManagerService(uint32_t service_id) : service_id_(service_id) {}

  /**
   * @brief Virtual destructor for the RivermaxManagerService class.
   */
  virtual ~RivermaxManagerService() = default;

  /**
   * @brief Returns the service ID.
   *
   * @return The unique identifier for this service.
   */
  uint32_t service_id() const { return service_id_; }

  /**
   * @brief Initializes the service.
   *
   * This method must be implemented by derived classes to perform service-specific initialization.
   *
   * @return True if initialization was successful, false otherwise.
   */
  virtual bool initialize() = 0;

  /**
   * @brief Runs the service.
   *
   * This method must be implemented by derived classes to start the service operation.
   */
  virtual void run() = 0;

  /**
   * @brief Shuts down the service.
   *
   * This method must be implemented by derived classes to perform clean shutdown of the service.
   */
  virtual void shutdown() = 0;

 protected:
  uint32_t service_id_;       ///< Unique identifier for the service
  uint16_t port_id_;          ///< Port ID derived from the service ID
  uint16_t queue_id_;         ///< Queue ID derived from the service ID
  bool initialized_ = false;  ///< Flag indicating whether the service is initialized
};

/**
 * @brief Base class for all Rivermax receiver services.
 *
 * This class provides common functionality for Rivermax receiver services,
 * including burst management, packet processing, and service lifecycle management.
 */
class RivermaxManagerRxService : public RivermaxManagerService {
 public:
  /**
   * @brief Constructor for the RivermaxManagerRxService class.
   *
   * @param service_id Unique identifier for the service.
   * @param rx_bursts_out_queue Shared queue for outgoing received bursts.
   */
  RivermaxManagerRxService(uint32_t service_id,
                           std::shared_ptr<AnoBurstsQueue> rx_bursts_out_queue);

  /**
   * @brief Virtual destructor for the RivermaxManagerRxService class.
   */
  virtual ~RivermaxManagerRxService() = default;

  /**
   * @brief Frees a burst buffer that was previously obtained.
   *
   * @param burst Pointer to the burst to be freed.
   */
  virtual void free_rx_burst(BurstParams* burst);

  /**
   * @brief Gets a received burst.
   *
   * @param burst Pointer to a pointer where the burst will be stored.
   * @return Status indicating the success or failure of the operation.
   */
  virtual Status get_rx_burst(BurstParams** burst);

  /**
   * @brief Prints service statistics.
   *
   * @param ss Stream to which statistics should be printed.
   */
  virtual void print_stats(std::stringstream& ss) const {}

  /**
   * @brief Gets the burst manager for this service.
   *
   * @return Shared pointer to the burst manager.
   */
  std::shared_ptr<RxBurstsManager> get_burst_manager() const { return rx_burst_manager_; }

  /**
   * @brief Applies the parsed burst pool configuration to the burst manager.
   *
   * This method configures the burst manager with the burst pool adaptive dropping
   * settings that were parsed from the YAML configuration.
   */
  void apply_burst_pool_configuration();

  bool initialize() override;
  void run() override;
  void shutdown() override;

 protected:
  /**
   * @brief Creates a service instance.
   *
   * This method must be implemented by derived classes to create the specific receiver service.
   *
   * @return Unique pointer to the created service.
   */
  virtual std::unique_ptr<RmaxReceiverBaseApp> create_service() = 0;

  /**
   * @brief Configures the service with service-specific settings.
   *
   * This method must be implemented by derived classes to configure the service.
   *
   * @return True if configuration was successful, false otherwise.
   */
  virtual bool configure_service() = 0;

 protected:
  static constexpr double GIGABYTE = 1073741824.0;          ///< Constant for gigabyte conversion
  static constexpr double MEGABYTE = 1048576.0;             ///< Constant for megabyte conversion
  std::unique_ptr<RmaxReceiverBaseApp> rx_service_;         ///< The receiver service instance
  std::shared_ptr<RxBurstsManager> rx_burst_manager_;       ///< Manager for received bursts
  std::shared_ptr<RxPacketProcessor> rx_packet_processor_;  ///< Processor for received packets
  std::shared_ptr<AnoBurstsQueue> rx_bursts_out_queue_;     ///< Output queue for received bursts
  bool send_packet_ext_info_ = false;                       ///< Flag for extended packet info
  int gpu_id_ = INVALID_GPU_ID;                             ///< GPU device ID
  size_t max_chunk_size_ = 0;  ///< Maximum chunk size for received data

  // Burst pool adaptive dropping configuration (private)
  bool burst_pool_adaptive_dropping_enabled_ = false;
  uint32_t burst_pool_low_threshold_percent_ = 25;
  uint32_t burst_pool_critical_threshold_percent_ = 10;
  uint32_t burst_pool_recovery_threshold_percent_ = 50;
};

/**
 * @brief IPO receiver service implementation.
 *
 * This class provides functionality for receiving and processing IPO protocol data.
 */
class IPOReceiverService : public RivermaxManagerRxService {
 public:
  /**
   * @brief Constructor for the IPOReceiverService class.
   *
   * @param service_id Unique identifier for the service.
   * @param ipo_receiver_builder Builder for IPO receiver settings.
   * @param rx_bursts_out_queue Shared queue for outgoing received bursts.
   */
  IPOReceiverService(
      uint32_t service_id,
      std::shared_ptr<RivermaxQueueToIPOReceiverSettingsBuilder> ipo_receiver_builder,
      std::shared_ptr<AnoBurstsQueue> rx_bursts_out_queue);

  /**
   * @brief Virtual destructor for the IPOReceiverService class.
   */
  virtual ~IPOReceiverService() = default;

  void print_stats(std::stringstream& ss) const override;

 protected:
  std::unique_ptr<RmaxReceiverBaseApp> create_service() override {
    return std::make_unique<IPOReceiverApp>(ipo_receiver_builder_);
  }
  bool configure_service() override;

 private:
  ///< Builder for IPO receiver settings
  std::shared_ptr<RivermaxQueueToIPOReceiverSettingsBuilder> ipo_receiver_builder_;
};

/**
 * @brief RTP receiver service implementation.
 *
 * This class provides functionality for receiving and processing RTP protocol data.
 */
class RTPReceiverService : public RivermaxManagerRxService {
 public:
  /**
   * @brief Constructor for the RTPReceiverService class.
   *
   * @param service_id Unique identifier for the service.
   * @param rtp_receiver_builder Builder for RTP receiver settings.
   * @param rx_bursts_out_queue Shared queue for outgoing received bursts.
   */
  RTPReceiverService(
      uint32_t service_id,
      std::shared_ptr<RivermaxQueueToRTPReceiverSettingsBuilder> rtp_receiver_builder,
      std::shared_ptr<AnoBurstsQueue> rx_bursts_out_queue);

  /**
   * @brief Virtual destructor for the RTPReceiverService class.
   */
  virtual ~RTPReceiverService() = default;

  void print_stats(std::stringstream& ss) const override;

 protected:
  std::unique_ptr<RmaxReceiverBaseApp> create_service() override {
    return std::make_unique<RTPReceiverApp>(rtp_receiver_builder_);
  }
  bool configure_service() override;

 private:
  ///< Builder for RTP receiver settings
  std::shared_ptr<RivermaxQueueToRTPReceiverSettingsBuilder> rtp_receiver_builder_;
};

/**
 * @brief Base class for all Rivermax transmitter services.
 *
 * This class provides common functionality for Rivermax transmitter services,
 * including burst management and service lifecycle management.
 */
class RivermaxManagerTxService : public RivermaxManagerService {
 public:
  /**
   * @brief Constructor for the RivermaxManagerTxService class.
   *
   * @param service_id Unique identifier for the service.
   */
  explicit RivermaxManagerTxService(uint32_t service_id);

  /**
   * @brief Virtual destructor for the RivermaxManagerTxService class.
   */
  virtual ~RivermaxManagerTxService() = default;

  /**
   * @brief Gets a burst buffer for transmission.
   *
   * @param burst Pointer to the burst buffer.
   * @return Status indicating the success or failure of the operation.
   */
  virtual Status get_tx_packet_burst(BurstParams* burst) = 0;

  /**
   * @brief Sends a burst.
   *
   * @param burst Pointer to the burst to be sent.
   * @return Status indicating the success or failure of the operation.
   */
  virtual Status send_tx_burst(BurstParams* burst) = 0;

  /**
   * @brief Checks if a burst is available for transmission.
   *
   * @param burst Pointer to the burst to check.
   * @return True if the burst is available, false otherwise.
   */
  virtual bool is_tx_burst_available(BurstParams* burst) = 0;

  /**
   * @brief Frees a burst buffer that was previously obtained.
   *
   * @param burst Pointer to the burst to be freed.
   */
  virtual void free_tx_burst(BurstParams* burst) = 0;

  /**
   * @brief Prints service statistics.
   *
   * @param ss Stream to which statistics should be printed.
   */
  virtual void print_stats(std::stringstream& ss) const {}

  bool initialize() override;
  void run() override;
  void shutdown() override;

 protected:
  /**
   * @brief Creates a service instance.
   *
   * This method must be implemented by derived classes to create the specific transmitter service.
   *
   * @return Unique pointer to the created service.
   */
  virtual std::unique_ptr<MediaSenderApp> create_service() = 0;

  /**
   * @brief Configures the service with service-specific settings.
   *
   * This method must be implemented by derived classes to configure the service.
   *
   * @return True if configuration was successful, false otherwise.
   */
  virtual bool configure_service() = 0;

  /**
   * @brief Performs additional setup after initialization.
   *
   * This method must be implemented by derived classes to perform any additional setup
   * required after the service is initialized.
   *
   * @return True if post-initialization setup was successful, false otherwise.
   */
  virtual bool post_init_setup() = 0;

  /**
   * @brief Prepares burst parameters for transmission.
   *
   * @param burst Pointer to the burst to prepare.
   * @return Pointer to the prepared burst.
   */
  BurstParams* prepare_burst_params(BurstParams* burst);

 protected:
  std::unique_ptr<MediaSenderApp> tx_service_;  ///< The transmitter service instance
};

/**
 * @brief Base class for media sender services.
 *
 * This class provides common functionality for media sender services.
 */
class MediaSenderBaseService : public RivermaxManagerTxService {
 public:
  /**
   * @brief Constructor for the MediaSenderBaseService class.
   *
   * @param service_id Unique identifier for the service.
   * @param media_sender_builder Builder for media sender settings.
   */
  MediaSenderBaseService(
      uint32_t service_id,
      std::shared_ptr<RivermaxQueueToMediaSenderSettingsBuilder> media_sender_builder);

  /**
   * @brief Virtual destructor for the MediaSenderBaseService class.
   */
  virtual ~MediaSenderBaseService() = default;

 protected:
  std::unique_ptr<MediaSenderApp> create_service() override {
    return std::make_unique<MediaSenderApp>(media_sender_builder_);
  }

  bool configure_service() override;

  /**  @brief Allocates memory for a pool buffer.
   *
   * @param pool_buffer_size Size of the buffer to allocate.
   * @param settings Application settings.
   * @return Pointer to the allocated memory.
   */
  void* allocate_pool_memory(size_t pool_buffer_size, const AppSettings* settings);

 protected:
  ///< Builder for media sender settings
  std::shared_ptr<RivermaxQueueToMediaSenderSettingsBuilder> media_sender_builder_;
  std::shared_ptr<MemoryAllocator> allocator_;
};

/**
 * @brief Mock implementation of media sender service.
 *
 * This class provides a mock implementation of the media sender service for testing purposes.
 */
class MediaSenderMockService : public MediaSenderBaseService {
 public:
  /**
   * @brief Constructor for the MediaSenderMockService class.
   *
   * @param service_id Unique identifier for the service.
   * @param media_sender_builder Builder for media sender settings.
   */
  MediaSenderMockService(
      uint32_t service_id,
      std::shared_ptr<RivermaxQueueToMediaSenderSettingsBuilder> media_sender_builder);

  /**
   * @brief Virtual destructor for the MediaSenderMockService class.
   */
  virtual ~MediaSenderMockService() = default;

  Status get_tx_packet_burst(BurstParams* burst) override;
  Status send_tx_burst(BurstParams* burst) override;
  bool is_tx_burst_available(BurstParams* burst) override { return true; }
  void free_tx_burst(BurstParams* burst) override {}

 protected:
  bool post_init_setup() override;

 private:
  std::unique_ptr<MediaFrame> processing_frame_;  ///< Currently processing media frame
};

/**
 * @brief Implementation of media sender service.
 *
 * This class provides functionality for sending media data.
 */
class MediaSenderService : public MediaSenderBaseService {
 public:
  /**
   * @brief Constructor for the MediaSenderService class.
   *
   * @param service_id Unique identifier for the service.
   * @param media_sender_builder Builder for media sender settings.
   */
  MediaSenderService(
      uint32_t service_id,
      std::shared_ptr<RivermaxQueueToMediaSenderSettingsBuilder> media_sender_builder);

  /**
   * @brief Virtual destructor for the MediaSenderService class.
   */
  virtual ~MediaSenderService() = default;

  Status get_tx_packet_burst(BurstParams* burst) override;
  Status send_tx_burst(BurstParams* burst) override;
  bool is_tx_burst_available(BurstParams* burst) override;
  void free_tx_burst(BurstParams* burst) override;
  void shutdown() override;

 protected:
  bool post_init_setup() override;

 private:
  std::unique_ptr<MediaFramePool> tx_media_frame_pool_;  ///< Pool for media frames
  std::shared_ptr<BufferedMediaFrameProvider>
      tx_media_frame_provider_;                   ///< Provider for buffered media frames
  std::shared_ptr<MediaFrame> processing_frame_;  ///< Currently processing media frame
  mutable std::mutex mutex_;

  static constexpr int MEDIA_FRAME_POOL_SIZE = 16;      ///< Size of the media frame pool
  static constexpr int MEDIA_FRAME_PROVIDER_SIZE = 32;  ///< Size of the media frame provider
};

/**
 * @brief Implementation of media sender service without internal memory pool.
 *
 * This class provides functionality for sending media data.
 * It doesn't have an internal memory pool and forwards media frames to the Rivermax
 * via @ref MediaFrameProvider.
 */
class MediaSenderZeroCopyService : public MediaSenderBaseService {
 public:
  /**
   * @brief Constructor for the MediaSenderZeroCopyService class.
   *
   * @param service_id Unique identifier for the service.
   * @param media_sender_builder Builder for media sender settings.
   */
  MediaSenderZeroCopyService(
      uint32_t service_id,
      std::shared_ptr<RivermaxQueueToMediaSenderSettingsBuilder> media_sender_builder);

  /**
   * @brief Virtual destructor for the MediaSenderZeroCopyService class.
   */
  virtual ~MediaSenderZeroCopyService() = default;

  Status get_tx_packet_burst(BurstParams* burst) override;
  Status send_tx_burst(BurstParams* burst) override;
  bool is_tx_burst_available(BurstParams* burst) override;
  void free_tx_burst(BurstParams* burst) override;
  void shutdown() override;

 protected:
  bool post_init_setup() override;

 private:
  std::shared_ptr<BufferedMediaFrameProvider>
      tx_media_frame_provider_;  ///< Provider for buffered media frames
  bool is_frame_in_process_ = false;
  mutable std::mutex mutex_;

  static constexpr int MEDIA_FRAME_PROVIDER_SIZE = 32;  ///< Size of the media frame provider
};

}  // namespace holoscan::advanced_network

#endif /* RIVERMAX_MANAGER_SERVICE_H_ */
