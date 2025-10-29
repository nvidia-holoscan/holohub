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
#include <sstream>
#include <string>

#include <holoscan/logger/logger.hpp>

#include "rivermax_mgr_impl/rivermax_chunk_consumer_ano.h"
#include "rivermax_mgr_service.h"

namespace holoscan::advanced_network {

RivermaxManagerRxService::RivermaxManagerRxService(
    uint32_t service_id, std::shared_ptr<AnoBurstsQueue> rx_bursts_out_queue)
    : RivermaxManagerService(service_id), rx_bursts_out_queue_(std::move(rx_bursts_out_queue)) {
  port_id_ = RivermaxBurst::burst_port_id_from_burst_tag(service_id);
  queue_id_ = RivermaxBurst::burst_queue_id_from_burst_tag(service_id);
}

bool RivermaxManagerRxService::initialize() {
  HOLOSCAN_LOG_INFO("Initializing Receiver:{}", service_id_);

  rx_service_ = create_service();
  if (!rx_service_) {
    HOLOSCAN_LOG_ERROR("Failed to create receiver service");
    return false;
  }

  auto init_status = rx_service_->initialize();
  if (init_status != ReturnStatus::obj_init_success) {
    HOLOSCAN_LOG_ERROR("Failed to initialize RX service, status: {}", (int)init_status);
    return false;
  }

  HOLOSCAN_LOG_INFO("Receiver:{} was successfully initialized", service_id_);

  if (!configure_service()) {
    HOLOSCAN_LOG_ERROR("Failed to configure service");
    return false;
  }

  rx_burst_manager_ = std::make_shared<RxBurstsManager>(
      send_packet_ext_info_, port_id_, queue_id_, max_chunk_size_, gpu_id_, rx_bursts_out_queue_);

  rx_packet_processor_ = std::make_shared<RxPacketProcessor>(rx_burst_manager_);

  auto rivermax_chunk_consumer =
      std::make_unique<RivermaxChunkConsumerAno>(rx_packet_processor_, max_chunk_size_);

  auto status = rx_service_->set_receive_data_consumer(0, std::move(rivermax_chunk_consumer));
  if (status != ReturnStatus::success) {
    HOLOSCAN_LOG_ERROR("Failed to set receive data consumer, status: {}", (int)status);
    return false;
  }

  initialized_ = true;
  return true;
}

void RivermaxManagerRxService::apply_burst_pool_configuration() {
  if (rx_burst_manager_) {
    // Apply the configuration to the burst manager
    rx_burst_manager_->set_adaptive_burst_dropping(burst_pool_adaptive_dropping_enabled_);
    rx_burst_manager_->configure_pool_thresholds(burst_pool_low_threshold_percent_,
                                                 burst_pool_critical_threshold_percent_,
                                                 burst_pool_recovery_threshold_percent_);

    HOLOSCAN_LOG_INFO("Applied burst pool configuration: enabled={}, thresholds={}%/{}%/{}%",
                      burst_pool_adaptive_dropping_enabled_,
                      burst_pool_low_threshold_percent_,
                      burst_pool_critical_threshold_percent_,
                      burst_pool_recovery_threshold_percent_);
  } else {
    HOLOSCAN_LOG_ERROR("Cannot apply burst pool configuration: burst manager not initialized");
  }
}

void RivermaxManagerRxService::free_rx_burst(BurstParams* burst) {
  if (!rx_burst_manager_) {
    HOLOSCAN_LOG_ERROR("RX burst manager not initialized");
    return;
  }

  rx_burst_manager_->rx_burst_done(static_cast<RivermaxBurst*>(burst));
}

void RivermaxManagerRxService::run() {
  if (!initialized_) {
    HOLOSCAN_LOG_ERROR("Cannot run uninitialized receiver service");
    return;
  }

  HOLOSCAN_LOG_INFO("Starting Receiver:{}", service_id_);
  ReturnStatus status = rx_service_->run();

  if (status != ReturnStatus::success) {
    HOLOSCAN_LOG_ERROR("Failed to run receiver service, status: {}", (int)status);
  }
}

void RivermaxManagerRxService::shutdown() {
  if (rx_service_) {
    HOLOSCAN_LOG_INFO("Shutting down Receiver:{}", service_id_);
  }
  initialized_ = false;
}

Status RivermaxManagerRxService::get_rx_burst(BurstParams** burst) {
  if (!rx_bursts_out_queue_) {
    HOLOSCAN_LOG_ERROR("RX bursts out queue not initialized");
    return Status::NOT_READY;
  }
  auto out_burst = rx_bursts_out_queue_->dequeue_burst().get();
  *burst = static_cast<BurstParams*>(out_burst);
  if (*burst == nullptr) {
    return Status::NOT_READY;
  }
  return Status::SUCCESS;
}

IPOReceiverService::IPOReceiverService(
    uint32_t service_id,
    std::shared_ptr<RivermaxQueueToIPOReceiverSettingsBuilder> ipo_receiver_builder,
    std::shared_ptr<AnoBurstsQueue> rx_bursts_out_queue)
    : RivermaxManagerRxService(service_id, rx_bursts_out_queue),
      ipo_receiver_builder_(ipo_receiver_builder) {}

bool IPOReceiverService::configure_service() {
  if (!ipo_receiver_builder_) {
    HOLOSCAN_LOG_ERROR("IPO receiver builder is null");
    return false;
  }
  send_packet_ext_info_ = ipo_receiver_builder_->send_packet_ext_info_;
  gpu_id_ = ipo_receiver_builder_->built_settings_.gpu_id;
  max_chunk_size_ = ipo_receiver_builder_->built_settings_.max_packets_in_rx_chunk;

  // Copy burst pool configuration
  burst_pool_adaptive_dropping_enabled_ =
      ipo_receiver_builder_->burst_pool_adaptive_dropping_enabled_;
  burst_pool_low_threshold_percent_ = ipo_receiver_builder_->burst_pool_low_threshold_percent_;
  burst_pool_critical_threshold_percent_ =
      ipo_receiver_builder_->burst_pool_critical_threshold_percent_;
  burst_pool_recovery_threshold_percent_ =
      ipo_receiver_builder_->burst_pool_recovery_threshold_percent_;
  return true;
}

void IPOReceiverService::print_stats(std::stringstream& ss) const {
  if (!rx_service_) {
    ss << "IPO Receiver Service ID: " << service_id_ << " (not initialized)" << std::endl;
    return;
  }

  IPOReceiverApp& ipo_service = static_cast<IPOReceiverApp&>(*rx_service_);
  auto stream_stats = ipo_service.get_streams_total_statistics();

  ss << "IPO Receiver Service ID: " << service_id_ << std::endl;

  for (uint32_t i = 0; i < stream_stats.size(); ++i) {
    ss << "[stream_index " << std::setw(3) << i << "]"
       << " Got " << std::setw(7) << stream_stats[i].rx_count << " packets | ";

    if (stream_stats[i].received_bytes >= RivermaxManagerRxService::GIGABYTE) {
      ss << std::fixed << std::setprecision(2)
         << (stream_stats[i].received_bytes / RivermaxManagerRxService::GIGABYTE) << " GB |";
    } else if (stream_stats[i].received_bytes >= RivermaxManagerRxService::MEGABYTE) {
      ss << std::fixed << std::setprecision(2)
         << (stream_stats[i].received_bytes / RivermaxManagerRxService::MEGABYTE) << " MB |";
    } else {
      ss << stream_stats[i].received_bytes << " bytes |";
    }

    ss << " dropped: ";
    for (uint32_t s_index = 0; s_index < stream_stats[i].path_stats.size(); ++s_index) {
      if (s_index > 0) {
        ss << ", ";
      }
      ss << stream_stats[i].path_stats[s_index].rx_dropped + stream_stats[i].rx_dropped;
    }
    ss << " |"
       << " consumed: " << stream_stats[i].consumed_packets << " |"
       << " unconsumed: " << stream_stats[i].unconsumed_packets << " |"
       << " lost: " << stream_stats[i].rx_dropped << " |"
       << " exceed MD: " << stream_stats[i].rx_exceed_md << " |"
       << " bad RTP hdr: " << stream_stats[i].rx_corrupt_header << " | ";

    for (uint32_t s_index = 0; s_index < stream_stats[i].path_stats.size(); ++s_index) {
      if (stream_stats[i].rx_count > 0) {
        uint32_t number = static_cast<uint32_t>(
            floor(100 * static_cast<double>(stream_stats[i].path_stats[s_index].rx_count) /
                  static_cast<double>(stream_stats[i].rx_count)));
        ss << " " << std::setw(3) << number << "%";
      } else {
        ss << "   0%";
      }
    }
    ss << std::endl;
  }
}

RTPReceiverService::RTPReceiverService(
    uint32_t service_id,
    std::shared_ptr<RivermaxQueueToRTPReceiverSettingsBuilder> rtp_receiver_builder,
    std::shared_ptr<AnoBurstsQueue> rx_bursts_out_queue)
    : RivermaxManagerRxService(service_id, rx_bursts_out_queue),
      rtp_receiver_builder_(rtp_receiver_builder) {}

bool RTPReceiverService::configure_service() {
  if (!rtp_receiver_builder_) {
    HOLOSCAN_LOG_ERROR("RTP receiver builder is null");
    return false;
  }
  send_packet_ext_info_ = rtp_receiver_builder_->send_packet_ext_info_;
  gpu_id_ = rtp_receiver_builder_->built_settings_.gpu_id;
  max_chunk_size_ = rtp_receiver_builder_->max_chunk_size_;

  // Copy burst pool configuration
  burst_pool_adaptive_dropping_enabled_ =
      rtp_receiver_builder_->burst_pool_adaptive_dropping_enabled_;
  burst_pool_low_threshold_percent_ = rtp_receiver_builder_->burst_pool_low_threshold_percent_;
  burst_pool_critical_threshold_percent_ =
      rtp_receiver_builder_->burst_pool_critical_threshold_percent_;
  burst_pool_recovery_threshold_percent_ =
      rtp_receiver_builder_->burst_pool_recovery_threshold_percent_;
  return true;
}

void RTPReceiverService::print_stats(std::stringstream& ss) const {
  if (!rx_service_) {
    ss << "RTP Receiver Service ID: " << service_id_ << " (not initialized)" << std::endl;
    return;
  }

  RTPReceiverApp& rtp_service = static_cast<RTPReceiverApp&>(*rx_service_);
  auto stream_stats = rtp_service.get_streams_total_statistics();

  ss << "RTP Receiver Service ID: " << service_id_ << std::endl;

  for (uint32_t i = 0; i < stream_stats.size(); ++i) {
    ss << "[stream_index " << std::setw(3) << i << "]"
       << " Got " << std::setw(7) << stream_stats[i].rx_count << " packets | ";

    if (stream_stats[i].received_bytes >= RivermaxManagerRxService::GIGABYTE) {
      ss << std::fixed << std::setprecision(2)
         << (stream_stats[i].received_bytes / RivermaxManagerRxService::GIGABYTE) << " GB |";
    } else if (stream_stats[i].received_bytes >= RivermaxManagerRxService::MEGABYTE) {
      ss << std::fixed << std::setprecision(2)
         << (stream_stats[i].received_bytes / RivermaxManagerRxService::MEGABYTE) << " MB |";
    } else {
      ss << stream_stats[i].received_bytes << " bytes |";
    }
    ss << " |"
       << " consumed: " << stream_stats[i].consumed_packets << " |"
       << " unconsumed: " << stream_stats[i].unconsumed_packets << " |"
       << " lost: " << stream_stats[i].rx_dropped << " |"
       << " bad RTP hdr: " << stream_stats[i].rx_corrupt_header << " | ";
    ss << std::endl;
  }
}

RivermaxManagerTxService::RivermaxManagerTxService(uint32_t service_id)
    : RivermaxManagerService(service_id) {
  port_id_ = RivermaxBurst::burst_port_id_from_burst_tag(service_id);
  queue_id_ = RivermaxBurst::burst_queue_id_from_burst_tag(service_id);
}

bool RivermaxManagerTxService::initialize() {
  HOLOSCAN_LOG_INFO("Initializing TX Service:{}", service_id_);

  tx_service_ = create_service();
  if (!tx_service_) {
    HOLOSCAN_LOG_ERROR("Failed to create TX service");
    return false;
  }

  if (!configure_service()) {
    HOLOSCAN_LOG_ERROR("Failed to configure TX service");
    return false;
  }

  auto init_status = tx_service_->initialize();
  if (init_status != ReturnStatus::obj_init_success) {
    HOLOSCAN_LOG_ERROR("Failed to initialize TX service, status: {}", (int)init_status);
    return false;
  }

  HOLOSCAN_LOG_INFO("TX Service:{} was successfully initialized", service_id_);

  if (!post_init_setup()) {
    HOLOSCAN_LOG_ERROR("Failed in post-initialization setup");
    return false;
  }

  initialized_ = true;
  return true;
}

void RivermaxManagerTxService::run() {
  if (!initialized_) {
    HOLOSCAN_LOG_ERROR("Cannot run uninitialized TX service");
    return;
  }

  HOLOSCAN_LOG_INFO("Starting TX Service:{}", service_id_);
  ReturnStatus status = tx_service_->run();

  if (status != ReturnStatus::success) {
    HOLOSCAN_LOG_ERROR("Failed to run TX service, status: {}", (int)status);
  }
}

void RivermaxManagerTxService::shutdown() {
  if (tx_service_) {
    HOLOSCAN_LOG_INFO("Shutting down TX Service:{}", service_id_);
  }
  initialized_ = false;
}

BurstParams* RivermaxManagerTxService::prepare_burst_params(BurstParams* burst) {
  burst->hdr.hdr.port_id = port_id_;
  burst->hdr.hdr.q_id = queue_id_;
  return burst;
}

MediaSenderBaseService::MediaSenderBaseService(
    uint32_t service_id,
    std::shared_ptr<RivermaxQueueToMediaSenderSettingsBuilder> media_sender_builder)
    : RivermaxManagerTxService(service_id), media_sender_builder_(media_sender_builder) {}

bool MediaSenderBaseService::configure_service() {
  if (!media_sender_builder_) {
    HOLOSCAN_LOG_ERROR("Media sender builder is null");
    return false;
  }
  return true;
}

void* MediaSenderBaseService::allocate_pool_memory(size_t pool_buffer_size,
                                                   const AppSettings* settings) {
  void* pool_buffer = nullptr;
  if (!media_sender_builder_->use_internal_memory_pool_) {
    HOLOSCAN_LOG_ERROR("Media sender service is configured not to use internal memory pool");
    return nullptr;
  }

  if (media_sender_builder_->memory_pool_location_ == MemoryKind::DEVICE) {
    cudaError_t cuda_status = cudaMalloc(&pool_buffer, pool_buffer_size);
    if (cuda_status != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Failed to allocate GPU memory: {}", cudaGetErrorString(cuda_status));
      return nullptr;
    }
    HOLOSCAN_LOG_INFO("Allocated GPU memory for frames memory pool at address: {}", pool_buffer);
  } else if (media_sender_builder_->memory_pool_location_ == MemoryKind::HOST_PINNED) {
    cudaError_t cuda_status = cudaHostAlloc(&pool_buffer, pool_buffer_size, cudaHostAllocDefault);
    if (cuda_status != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Failed to allocate pinned host memory: {}",
                         cudaGetErrorString(cuda_status));
      cudaFree(pool_buffer);
      return nullptr;
    }
    HOLOSCAN_LOG_INFO("Allocated pinned host memory for frames memory pool at address: {}",
                      pool_buffer);
  } else if (media_sender_builder_->memory_pool_location_ == MemoryKind::HUGE) {
    RivermaxDevKitFacade& rivermax_dev_kit(RivermaxDevKitFacade::get_instance());
    std::shared_ptr<AppSettings> app_settings(const_cast<AppSettings*>(settings),
                                              [](AppSettings*) {});
    allocator_ =
        rivermax_dev_kit.get_memory_allocator(AllocatorType::HugePageDefault, app_settings);
    if (allocator_ == nullptr) {
      HOLOSCAN_LOG_ERROR("Failed to create header memory allocator");
      return nullptr;
    }
    pool_buffer_size = allocator_->align_length(pool_buffer_size);
    pool_buffer = allocator_->allocate_aligned(pool_buffer_size, allocator_->get_page_size());
    if (pool_buffer == nullptr) {
      HOLOSCAN_LOG_ERROR("Failed to allocate huge memory");
      return nullptr;
    }
    HOLOSCAN_LOG_INFO("Allocated huge memory for frames memory pool at address: {}", pool_buffer);
  } else {
    pool_buffer = new byte_t[pool_buffer_size];
    if (pool_buffer == nullptr) {
      HOLOSCAN_LOG_ERROR("Failed to allocate host memory");
      return nullptr;
    }
    HOLOSCAN_LOG_INFO("Allocated host memory for frames memory pool at address: {}", pool_buffer);
  }
  return pool_buffer;
}

MediaSenderMockService::MediaSenderMockService(
    uint32_t service_id,
    std::shared_ptr<RivermaxQueueToMediaSenderSettingsBuilder> media_sender_builder)
    : MediaSenderBaseService(service_id, media_sender_builder) {}

bool MediaSenderMockService::post_init_setup() {
  const AppSettings* settings = nullptr;
  auto status = tx_service_->get_app_settings(settings);

  if (status != ReturnStatus::success) {
    HOLOSCAN_LOG_ERROR("Failed to get settings from TX service");
    return false;
  }
  size_t pool_buffer_size = settings->media.bytes_per_frame * 1;
  void* pool_buffer = allocate_pool_memory(pool_buffer_size, settings);
  if (pool_buffer == nullptr) {
    HOLOSCAN_LOG_ERROR("Failed to allocate memory for media frame pool");
    return false;
  }

  processing_frame_ = std::make_unique<MediaFrame>(static_cast<byte_t*>(pool_buffer),
                                                   settings->media.bytes_per_frame);
  return true;
}

Status MediaSenderMockService::get_tx_packet_burst(BurstParams* burst) {
  if (!initialized_) {
    HOLOSCAN_LOG_ERROR("Media Sender service not initialized");
    return Status::INVALID_PARAMETER;
  }

  prepare_burst_params(burst);
  burst->pkts[0][0] = processing_frame_->data->get();
  burst->hdr.hdr.max_pkt = 1;  // Single packet in burst

  return Status::SUCCESS;
}

Status MediaSenderMockService::send_tx_burst(BurstParams* burst) {
  if (!initialized_) {
    HOLOSCAN_LOG_ERROR("Media Sender service not initialized");
    return Status::INVALID_PARAMETER;
  }

  if (!processing_frame_) {
    HOLOSCAN_LOG_ERROR("MediaSenderMockService{}:{}::send_tx_burst(): No frame in processing",
                       port_id_,
                       queue_id_);
    return Status::INVALID_PARAMETER;
  }

  return Status::SUCCESS;
}

MediaSenderService::MediaSenderService(
    uint32_t service_id,
    std::shared_ptr<RivermaxQueueToMediaSenderSettingsBuilder> media_sender_builder)
    : MediaSenderBaseService(service_id, media_sender_builder) {}

bool MediaSenderService::post_init_setup() {
  const AppSettings* settings = nullptr;
  auto status = tx_service_->get_app_settings(settings);

  if (status != ReturnStatus::success) {
    HOLOSCAN_LOG_ERROR("Failed to get settings from TX service");
    return false;
  }
  size_t pool_buffer_size = settings->media.bytes_per_frame * MEDIA_FRAME_POOL_SIZE;
  void* pool_buffer = allocate_pool_memory(pool_buffer_size, settings);
  if (pool_buffer == nullptr) {
    HOLOSCAN_LOG_ERROR("Failed to allocate memory for media frame pool");
    return false;
  }
  HOLOSCAN_LOG_INFO("Creating memory pool at address: {}", pool_buffer);
  tx_media_frame_pool_ = std::make_unique<MediaFramePool>(MEDIA_FRAME_POOL_SIZE,
                                                          settings->media.bytes_per_frame,
                                                          static_cast<byte_t*>(pool_buffer),
                                                          pool_buffer_size);

  tx_media_frame_provider_ =
      std::make_shared<BufferedMediaFrameProvider>(MEDIA_FRAME_PROVIDER_SIZE);
  // Set the media frame provider to the TX service
  status = tx_service_->set_frame_provider(0, tx_media_frame_provider_);
  if (status != ReturnStatus::success) {
    HOLOSCAN_LOG_ERROR("Failed to set frame provider to TX service");
    return false;
  }
  return true;
}

Status MediaSenderService::get_tx_packet_burst(BurstParams* burst) {
  if (!initialized_) {
    HOLOSCAN_LOG_ERROR("Media Sender service not initialized");
    return Status::INVALID_PARAMETER;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (processing_frame_) {
    HOLOSCAN_LOG_ERROR(
        "MediaSenderService{}:{}::get_tx_packet_burst(): Frame is already in process",
        port_id_,
        queue_id_);
    return Status::INVALID_PARAMETER;
  }
  if (burst->hdr.hdr.q_id != queue_id_ || burst->hdr.hdr.port_id != port_id_) {
    HOLOSCAN_LOG_ERROR("MediaSenderService{}:{}::get_tx_packet_burst(): Burst queue ID mismatch",
                       port_id_,
                       queue_id_);
    return Status::INVALID_PARAMETER;
  }

  auto frame = tx_media_frame_pool_->get_frame();
  if (!frame) {
    HOLOSCAN_LOG_ERROR(
        "MediaSenderService{}:{}::get_tx_packet_burst(): Failed to get frame from pool",
        port_id_,
        queue_id_);
    return Status::NO_FREE_BURST_BUFFERS;
  }

  prepare_burst_params(burst);
  burst->pkts[0][0] = frame->data->get();
  burst->pkt_lens[0][0] = frame->data->get_size();
  burst->hdr.hdr.max_pkt = 1;  // Single packet in burst

  processing_frame_ = std::move(frame);
  HOLOSCAN_LOG_TRACE(
      "MediaSenderService{}:{}::get_tx_packet_burst(): Processing Frame is set,"
      "frame address: {} with size {}",
      port_id_,
      queue_id_,
      static_cast<const void*>(processing_frame_->data->get()),
      processing_frame_->data->get_size());
  return Status::SUCCESS;
}

Status MediaSenderService::send_tx_burst(BurstParams* burst) {
  if (!initialized_) {
    HOLOSCAN_LOG_ERROR("Media Sender service not initialized");
    return Status::INVALID_PARAMETER;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!processing_frame_) {
    HOLOSCAN_LOG_ERROR(
        "MediaSenderService{}:{}::send_tx_burst(): No frame in processing", port_id_, queue_id_);
    return Status::INVALID_PARAMETER;
  }
  HOLOSCAN_LOG_TRACE("MediaSenderService{}:{}::send_tx_burst(): Out frame size is {}",
                     port_id_,
                     queue_id_,
                     tx_media_frame_provider_->get_queue_size());
  auto status = tx_media_frame_provider_->add_frame(std::move(processing_frame_));
  if (status != ReturnStatus::success) {
    HOLOSCAN_LOG_ERROR(
        "MediaSenderService{}:{}::send_tx_burst(): Failed to add frame to provider! Frame will be "
        "dropped",
        port_id_,
        queue_id_);
    processing_frame_.reset();  // Clear the frame to avoid leaks
    return Status::NO_SPACE_AVAILABLE;
  }
  HOLOSCAN_LOG_TRACE(
      "MediaSenderService{}:{}::send_tx_burst(): Frame was sent", port_id_, queue_id_);
  processing_frame_.reset();  // Clear the reference as it's now managed by the provider
  return Status::SUCCESS;
}

bool MediaSenderService::is_tx_burst_available(BurstParams* burst) {
  if (!initialized_ || !tx_media_frame_pool_) {
    return false;
  }
  //  Check if we have available frames in the pool and no current processing frame
  return (tx_media_frame_pool_->get_available_frames_count() > 0 && !processing_frame_);
}

void MediaSenderService::free_tx_burst(BurstParams* burst) {
  // If we have a processing frame but we're told to free the burst,
  // we should clear the processing frame to avoid leaks
  std::lock_guard<std::mutex> lock(mutex_);
  HOLOSCAN_LOG_TRACE(
      "MediaSenderService{}:{}::free_tx_burst(): Processing frame was reset", port_id_, queue_id_);
  if (processing_frame_) {
    processing_frame_.reset();
  }
}

void MediaSenderService::shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (processing_frame_) {
      processing_frame_.reset();
    }
  }

  if (tx_media_frame_provider_) {
    tx_media_frame_provider_->stop();
  }
  if (tx_media_frame_pool_) {
    tx_media_frame_pool_->stop();
  }
}

MediaSenderZeroCopyService::MediaSenderZeroCopyService(
    uint32_t service_id,
    std::shared_ptr<RivermaxQueueToMediaSenderSettingsBuilder> media_sender_builder)
    : MediaSenderBaseService(service_id, media_sender_builder) {}

bool MediaSenderZeroCopyService::post_init_setup() {
  tx_media_frame_provider_ =
      std::make_shared<BufferedMediaFrameProvider>(MEDIA_FRAME_PROVIDER_SIZE);
  // Set the media frame provider to the TX service
  auto status = tx_service_->set_frame_provider(0, tx_media_frame_provider_);
  if (status != ReturnStatus::success) {
    HOLOSCAN_LOG_ERROR("Failed to set frame provider to TX service");
    return false;
  }
  return true;
}

Status MediaSenderZeroCopyService::get_tx_packet_burst(BurstParams* burst) {
  if (!initialized_) {
    HOLOSCAN_LOG_ERROR("Media Sender service not initialized");
    return Status::INVALID_PARAMETER;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (is_frame_in_process_) {
    HOLOSCAN_LOG_ERROR(
        "MediaSenderZeroCopyService{}:{}::get_tx_packet_burst(): Frame is already in process",
        port_id_,
        queue_id_);
    return Status::INVALID_PARAMETER;
  }
  if (burst->hdr.hdr.q_id != queue_id_ || burst->hdr.hdr.port_id != port_id_) {
    HOLOSCAN_LOG_ERROR(
        "MediaSenderZeroCopyService{}:{}::get_tx_packet_burst(): Burst queue ID "
        "mismatch",
        port_id_,
        queue_id_);
    return Status::INVALID_PARAMETER;
  }

  prepare_burst_params(burst);
  burst->hdr.hdr.max_pkt = 1;  // Single packet in burst
  burst->hdr.hdr.burst_flags = FLAGS_NONE;
  burst->custom_pkt_data = nullptr;

  is_frame_in_process_ = true;
  HOLOSCAN_LOG_TRACE(
      "MediaSenderZeroCopyService{}:{}::get_tx_packet_burst(): Processing Frame is set,",
      port_id_,
      queue_id_);
  return Status::SUCCESS;
}

Status MediaSenderZeroCopyService::send_tx_burst(BurstParams* burst) {
  if (!initialized_) {
    HOLOSCAN_LOG_ERROR("Media Sender service not initialized");
    return Status::INVALID_PARAMETER;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if ((burst->hdr.hdr.burst_flags & FRAME_BUFFER_IS_OWNED)) {
    HOLOSCAN_LOG_ERROR(
        "MediaSenderZeroCopyService{}:{}::send_tx_burst(): Illegal burst flags."
        "Buffer is owned by frame",
        port_id_,
        queue_id_);
    return Status::INVALID_PARAMETER;
  }
  std::shared_ptr<MediaFrame> out_frame =
      std::static_pointer_cast<MediaFrame>(burst->custom_pkt_data);
  burst->custom_pkt_data.reset();
  if (!out_frame) {
    HOLOSCAN_LOG_ERROR(
        "MediaSenderZeroCopyService{}:{}::send_tx_burst(): No frame was provided in the burst",
        port_id_,
        queue_id_);
    return Status::INVALID_PARAMETER;
  }
  HOLOSCAN_LOG_TRACE("MediaSenderZeroCopyService{}:{}::send_tx_burst(): Out frame size is {}",
                     port_id_,
                     queue_id_,
                     tx_media_frame_provider_->get_queue_size());

  auto status = tx_media_frame_provider_->add_frame(std::move(out_frame));
  if (status != ReturnStatus::success) {
    HOLOSCAN_LOG_ERROR(
        "MediaSenderZeroCopyService{}:{}::send_tx_burst(): Failed to add frame to provider! Frame "
        "will be dropped",
        port_id_,
        queue_id_);
    is_frame_in_process_ = false;
    return Status::NO_SPACE_AVAILABLE;
  }
  HOLOSCAN_LOG_TRACE(
      "MediaSenderZeroCopyService{}:{}::send_tx_burst(): Frame was sent", port_id_, queue_id_);
  is_frame_in_process_ = false;
  return Status::SUCCESS;
}

bool MediaSenderZeroCopyService::is_tx_burst_available(BurstParams* burst) {
  if (!initialized_) {
    return false;
  }
  return (!is_frame_in_process_ &&
          tx_media_frame_provider_->get_queue_size() < MEDIA_FRAME_PROVIDER_SIZE);
}

void MediaSenderZeroCopyService::free_tx_burst(BurstParams* burst) {
  // If we have a processing frame but we're told to free the burst,
  // we should clear the processing frame flag
  std::lock_guard<std::mutex> lock(mutex_);
  HOLOSCAN_LOG_TRACE("MediaSenderZeroCopyService{}:{}::free_tx_burst(): Processing frame was reset",
                     port_id_,
                     queue_id_);
  is_frame_in_process_ = false;
}

void MediaSenderZeroCopyService::shutdown() {
  if (tx_media_frame_provider_) {
    tx_media_frame_provider_->stop();
  }
}

}  // namespace holoscan::advanced_network
