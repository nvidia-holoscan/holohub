/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <hsb_roce_receiver_nmd/roce_receiver_op.hpp>

#include <unistd.h>

#include <netinet/in.h>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/networking.hpp>
#include <holoscan/logger/logger.hpp>

#include <hololink/operators/roce_receiver/roce_receiver.hpp>
#include <hsb_roce_receiver_nmd/roce_receiver_no_host_metadata.hpp>

namespace hololink::operators {

RoceReceiverOp::~RoceReceiverOp() {
  if (receiver_thread_ && receiver_thread_->joinable()) { receiver_thread_->join(); }
}

void RoceReceiverOp::initialize() {
  // Set default identity function if rename_metadata is not set
  if (!rename_metadata_) {
    rename_metadata_ = [](const std::string& name) { return name; };
  }

  // Cache the metadata key names using the rename callback
  const auto& rename_fn = rename_metadata_;
  received_frame_number_metadata_ = rename_fn("received_frame_number");
  rx_write_requests_metadata_ = rename_fn("rx_write_requests");
  received_s_metadata_ = rename_fn("received_s");
  received_ns_metadata_ = rename_fn("received_ns");
  imm_data_metadata_ = rename_fn("imm_data");
  frame_memory_metadata_ = rename_fn("frame_memory");
  dropped_metadata_ = rename_fn("dropped");
  frame_number_metadata_ = rename_fn("frame_number");
  timestamp_s_metadata_ = rename_fn("timestamp_s");
  timestamp_ns_metadata_ = rename_fn("timestamp_ns");
  metadata_s_metadata_ = rename_fn("metadata_s");
  metadata_ns_metadata_ = rename_fn("metadata_ns");
  crc_metadata_ = rename_fn("crc");
  bytes_written_ = rename_fn("bytes_written");

  // Call base class initialize
  BaseReceiverOp::initialize();
}

void RoceReceiverOp::setup(holoscan::OperatorSpec& spec) {
  // call base class
  BaseReceiverOp::setup(spec);

  // and add our own parameters
  spec.param(ibv_name_, "ibv_name", "IBVName", "IBV device to use", std::string("roceP5p3s0f0"));
  spec.param(ibv_port_, "ibv_port", "IBVPort", "Port number of IBV device", 1u);
  spec.param(pages_, "pages", "Pages", "Number of pages to use for the receiver memory", 2u);
  spec.param(skip_host_metadata_,
             "skip_host_metadata",
             "SkipHostMetadata",
             "Skip copying metadata to host (for GPU-resident processing)",
             false);
  // Note: rename_metadata is handled programmatically via set_rename_metadata() method
  // to avoid YAML-CPP serialization issues with std::function
}

void RoceReceiverOp::set_rename_metadata(std::function<std::string(const std::string&)> rename_fn) {
  rename_metadata_ = rename_fn;
}

CUdeviceptr RoceReceiverOp::frame_memory_base() const {
  return frame_memory_ ? frame_memory_->get() : 0;
}

void RoceReceiverOp::start_receiver() {
  size_t metadata_address = hololink::core::round_up(frame_size_.get(), hololink::core::PAGE_SIZE);
  // received_frame_size wants to be page aligned; prove that METADATA_SIZE doesn't upset that.
  // Prove that PAGE_SIZE is a power of two
  static_assert((hololink::core::PAGE_SIZE & (hololink::core::PAGE_SIZE - 1)) == 0);
  // Prove that METADATA_SIZE is an even multiple of PAGE_SIZE
  static_assert((hololink::METADATA_SIZE & (hololink::core::PAGE_SIZE - 1)) == 0);
  size_t received_frame_size = metadata_address + hololink::METADATA_SIZE;
  size_t buffer_size = hololink::core::round_up(received_frame_size * pages_.get(), getpagesize());
  frame_memory_.reset(new ReceiverMemoryDescriptor(frame_context_, buffer_size));
  HOLOSCAN_LOG_INFO("frame_size={:#x} frame={:#x} buffer_size={:#x}",
                    frame_size_.get(),
                    frame_memory_->get(),
                    buffer_size);

  const std::string& peer_ip = hololink_channel_->peer_ip();
  HOLOSCAN_LOG_INFO("ibv_name_={} ibv_port_={} peer_ip={} skip_host_metadata={}",
                    ibv_name_.get(),
                    ibv_port_.get(),
                    peer_ip,
                    skip_host_metadata_.get());
  if (skip_host_metadata_.get()) {
    receiver_ = std::make_shared<RoceReceiverNoHostMetadata>(ibv_name_.get().c_str(),
                                                             ibv_port_.get(),
                                                             frame_memory_->get(),
                                                             buffer_size,
                                                             frame_size_.get(),
                                                             received_frame_size,
                                                             pages_.get(),
                                                             metadata_address,
                                                             peer_ip.c_str());
  } else {
    receiver_ = std::make_shared<RoceReceiver>(ibv_name_.get().c_str(),
                                               ibv_port_.get(),
                                               frame_memory_->get(),
                                               buffer_size,
                                               frame_size_.get(),
                                               received_frame_size,
                                               pages_.get(),
                                               metadata_address,
                                               peer_ip.c_str());
  }
  auto boolean_condition = this->condition<holoscan::BooleanCondition>("receiver_tick");
  receiver_->set_frame_ready([this, boolean_condition](const RoceReceiver&) {
    // Check boolean condition so distributed app shuts down properly
    if (boolean_condition && !boolean_condition->check_tick_enabled()) {
      HOLOSCAN_LOG_DEBUG(
          "boolean_condition not there or boolean condition "
          "check_tick_enabled() is false, returning");
      this->frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_NEVER);
      return;
    }
    this->frame_ready();
  });
  if (!receiver_->start()) {
    throw std::runtime_error("Failed to start RoceReceiver");
  }
  hololink_channel_->authenticate(receiver_->get_qp_number(), receiver_->get_rkey());

  receiver_thread_.reset(new std::thread(&hololink::operators::RoceReceiverOp::run, this));
  const int error = pthread_setname_np(receiver_thread_->native_handle(), name().c_str());
  if (error != 0) {
    throw std::runtime_error("Failed to set thread name");
  }

  auto [local_ip, local_port] = local_ip_and_port();
  HOLOSCAN_LOG_INFO("local_ip={} local_port={}", local_ip, local_port);

  hololink_channel_->configure_roce(receiver_->external_frame_memory(),
                                    frame_size_,
                                    received_frame_size,
                                    pages_.get(),
                                    local_port);
}

void RoceReceiverOp::run() {
  CudaCheck(cuCtxSetCurrent(frame_context_));
  receiver_->blocking_monitor();
}

void RoceReceiverOp::stop_receiver() {
  hololink_channel_->unconfigure();
  data_socket_.reset();
  receiver_->close();
  if (receiver_thread_ && receiver_thread_->joinable()) { receiver_thread_->join(); }
  receiver_thread_.reset();
  frame_memory_.reset();
}

std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> RoceReceiverOp::get_next_frame(
    double timeout_ms) {
  RoceReceiverMetadata roce_receiver_metadata;
  if (!receiver_->get_next_frame(timeout_ms, roce_receiver_metadata)) {
    return {};
  }

  auto metadata = std::make_shared<Metadata>();
  (*metadata)[received_frame_number_metadata_] =
      int64_t(roce_receiver_metadata.received_frame_number);
  (*metadata)[rx_write_requests_metadata_] = int64_t(roce_receiver_metadata.rx_write_requests);
  (*metadata)[received_s_metadata_] = int64_t(roce_receiver_metadata.received_s);
  (*metadata)[received_ns_metadata_] = int64_t(roce_receiver_metadata.received_ns);
  (*metadata)[imm_data_metadata_] = int64_t(roce_receiver_metadata.imm_data);
  CUdeviceptr frame_memory = roce_receiver_metadata.frame_memory;
  (*metadata)[frame_memory_metadata_] = int64_t(frame_memory);
  (*metadata)[dropped_metadata_] = int64_t(roce_receiver_metadata.dropped);
  (*metadata)[frame_number_metadata_] = int64_t(roce_receiver_metadata.frame_number);
  (*metadata)[timestamp_s_metadata_] = int64_t(roce_receiver_metadata.frame_metadata.timestamp_s);
  (*metadata)[timestamp_ns_metadata_] = int64_t(roce_receiver_metadata.frame_metadata.timestamp_ns);
  (*metadata)[metadata_s_metadata_] = int64_t(roce_receiver_metadata.frame_metadata.metadata_s);
  (*metadata)[metadata_ns_metadata_] = int64_t(roce_receiver_metadata.frame_metadata.metadata_ns);
  (*metadata)[crc_metadata_] = int64_t(roce_receiver_metadata.frame_metadata.crc);
  (*metadata)[bytes_written_] = int64_t(roce_receiver_metadata.frame_metadata.bytes_written);

  return {frame_memory, metadata};
}

std::tuple<std::string, uint32_t> RoceReceiverOp::local_ip_and_port() {
  sockaddr_in ip{};
  ip.sin_family = AF_UNSPEC;
  socklen_t ip_len = sizeof(ip);
  if (getsockname(data_socket_.get(), (sockaddr*)&ip, &ip_len) < 0) {
    throw std::runtime_error(
        fmt::format("getsockname failed with errno={}: \"{}\"", errno, strerror(errno)));
  }

  const std::string local_ip = inet_ntoa(ip.sin_addr);
  // This is what you'd normally use
  // const in_port_t local_port = ntohs(ip.sin_port);
  // But we're going to tell the other side that we're listening
  // to the ROCE receiver port at 4791.
  const in_port_t local_port = 4791;
  return {local_ip, local_port};
}

}  // namespace hololink::operators
