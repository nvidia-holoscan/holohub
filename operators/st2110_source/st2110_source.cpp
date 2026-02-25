/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, XRlabs. All rights reserved.
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

#include "st2110_source.hpp"
#include "st2110_kernels.cuh"

#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <errno.h>

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"
#include "gxf/multimedia/video.hpp"

#define CUDA_TRY(stmt)                                                                          \
  ({                                                                                            \
    cudaError_t _holoscan_cuda_err = stmt;                                                      \
    if (cudaSuccess != _holoscan_cuda_err) {                                                    \
      HOLOSCAN_LOG_ERROR("CUDA Runtime call {} in line {} of file {} failed with '{}' ({})",   \
                         #stmt,                                                                 \
                         __LINE__,                                                              \
                         __FILE__,                                                              \
                         cudaGetErrorString(_holoscan_cuda_err),                                \
                         static_cast<int>(_holoscan_cuda_err));                                 \
    }                                                                                           \
    _holoscan_cuda_err;                                                                         \
  })

namespace holoscan::ops {

ST2110SourceOp::~ST2110SourceOp() {
  HOLOSCAN_LOG_INFO("ST2110SourceOp shutting down. Stats: {}/{} bytes/packets received, {} dropped",
                    total_bytes_received_,
                    total_packets_received_,
                    total_packets_dropped_);
  free_buffers();
  close_socket();
}

void ST2110SourceOp::setup(OperatorSpec& spec) {
  auto& raw_output = spec.output<gxf::Entity>("raw_output");

  spec.param(raw_output_,
             "raw_output",
             "RawOutput",
             "Output port for raw video frames with format metadata (always available)",
             &raw_output);

  auto& rgba_output = spec.output<gxf::Entity>("rgba_output");

  spec.param(rgba_output_,
             "rgba_output",
             "RGBAOutput",
             "Output port for RGBA converted frames (optional)",
             &rgba_output);

  auto& nv12_output = spec.output<gxf::Entity>("nv12_output");

  spec.param(nv12_output_,
             "nv12_output",
             "NV12Output",
             "Output port for NV12 converted frames (optional)",
             &nv12_output);

  spec.param(multicast_address_,
             "multicast_address",
             "Multicast Address",
             "Multicast IP address for ST 2110 stream (e.g., \"239.255.66.60\")",
             std::string("239.0.0.1"));

  spec.param(port_,
             "port",
             "Port",
             "UDP port for ST 2110 stream",
             static_cast<uint16_t>(5004));

  spec.param(interface_name_,
             "interface_name",
             "Interface Name",
             "Linux network interface name (e.g., \"mgbe0_0\", \"eth0\")",
             std::string("eth0"));

  spec.param(width_,
             "width",
             "Width",
             "Width of the video stream in pixels",
             1920u);

  spec.param(height_,
             "height",
             "Height",
             "Height of the video stream in pixels",
             1080u);

  spec.param(framerate_,
             "framerate",
             "Frame Rate",
             "Expected frame rate of the video stream",
             60u);

  spec.param(stream_format_,
             "stream_format",
             "Stream Format",
             "Input stream format (YCbCr-4:2:2-10bit, YCbCr-4:2:2-8bit, RGBA-8bit)",
             std::string("YCbCr-4:2:2-10bit"));

  spec.param(enable_rgba_output_,
             "enable_rgba_output",
             "Enable RGBA Output",
             "Enable RGBA conversion and emission on rgba_output port",
             false);

  spec.param(enable_nv12_output_,
             "enable_nv12_output",
             "Enable NV12 Output",
             "Enable NV12 conversion and emission on nv12_output port",
             false);

  spec.param(batch_size_,
             "batch_size",
             "Batch Size",
             "Number of packets to receive per compute() call",
             1000u);

  spec.param(max_packet_size_,
             "max_packet_size",
             "Max Packet Size",
             "Maximum size of ST 2110 packets in bytes",
             static_cast<uint16_t>(1514));

  spec.param(header_size_,
             "header_size",
             "Header Size",
             "Size of L2-L4 headers (Ethernet + IP + UDP) in bytes",
             static_cast<uint16_t>(42));

  spec.param(rtp_header_size_,
             "rtp_header_size",
             "RTP Header Size",
             "Size of RTP header in bytes",
             static_cast<uint16_t>(12));

  spec.param(enable_reorder_kernel_,
             "enable_reorder_kernel",
             "Enable Reorder Kernel",
             "Enable CUDA kernel for packet reordering",
             true);

  // Add periodic condition to trigger compute() at framerate
  // This is essential for source operators with no inputs
  spec.param(periodic_condition_,
             "periodic_condition",
             "Periodic Condition",
             "Scheduling condition to trigger packet reception at video framerate",
             ParameterFlag::kOptional);

  // UnboundedAllocator prevents race conditions with async Python operators
  spec.param(tensor_allocator_,
             "tensor_allocator",
             "Tensor Allocator",
             "Allocator for Tensor memory management",
             std::static_pointer_cast<Allocator>(
                 fragment()->make_resource<UnboundedAllocator>("tensor_allocator")));
}

void ST2110SourceOp::initialize() {
  HOLOSCAN_LOG_INFO("ST2110SourceOp::initialize()");

  // Required when operator creates resource as default parameter value
  add_arg(tensor_allocator_.default_value());

  holoscan::Operator::initialize();

  configure_stream_format();

  // Calculate frame sizes based on enabled outputs
  if (enable_rgba_output_.get()) {
    rgba_frame_size_ = width_.get() * height_.get() * 4;  // RGBA = 4 bytes/pixel
    HOLOSCAN_LOG_INFO("RGBA output enabled: {} MB per frame", rgba_frame_size_ / (1024*1024));
  } else {
    rgba_frame_size_ = 0;
  }

  if (enable_nv12_output_.get()) {
    nv12_frame_size_ = width_.get() * height_.get() * 1.5;  // NV12 = 1.5 bytes/pixel
    HOLOSCAN_LOG_INFO("NV12 output enabled: {} MB per frame", nv12_frame_size_ / (1024*1024));
  } else {
    nv12_frame_size_ = 0;
  }

  // UDP socket recv() gives us payload only (no Eth/IP/UDP headers)
  // Video payload per packet = total - RTP header - ST2110 header
  payload_size_ = max_packet_size_.get() - rtp_header_size_.get() - sizeof(ST2110Header);

  // Create default PeriodicCondition if not provided
  // Run at 10x framerate for greedy socket polling while limiting CPU overhead
  // E.g., 50fps → 500Hz polling for low-latency packet ingestion
  if (!periodic_condition_.has_value()) {
    uint32_t polling_rate = framerate_.get() * 10;
    std::string recess_period = std::to_string(polling_rate) + "hz";
    periodic_condition_ = fragment()->make_condition<PeriodicCondition>(
        "st2110_periodic_condition",
        Arg("recess_period") = recess_period);
    add_arg(periodic_condition_.get());
    HOLOSCAN_LOG_INFO("Created default PeriodicCondition at {}Hz ({}x framerate for greedy ingestion)",
                      polling_rate, 10);
  }

  HOLOSCAN_LOG_INFO("ST2110 configuration:");
  HOLOSCAN_LOG_INFO("  Multicast: {}:{}", multicast_address_.get(), port_.get());
  HOLOSCAN_LOG_INFO("  Interface: {}", interface_name_.get());
  HOLOSCAN_LOG_INFO("  Resolution: {}x{}@{}", width_.get(), height_.get(), framerate_.get());
  HOLOSCAN_LOG_INFO("  Stream format: {}", stream_format_.get());
  HOLOSCAN_LOG_INFO("  Outputs enabled:");
  HOLOSCAN_LOG_INFO("    - raw_output: always enabled");
  if (enable_rgba_output_.get()) {
    HOLOSCAN_LOG_INFO("    - rgba_output: enabled ({} MB/frame)", rgba_frame_size_ / (1024*1024));
  }
  if (enable_nv12_output_.get()) {
    HOLOSCAN_LOG_INFO("    - nv12_output: enabled ({} MB/frame)", nv12_frame_size_ / (1024*1024));
  }
  HOLOSCAN_LOG_INFO("  Payload size: {} bytes/packet", payload_size_);

  // Reset state (in-class initializers provide defaults, but initialize() may be called again)
  socket_fd_ = -1;
  total_bytes_received_ = 0;
  total_packets_received_ = 0;
  total_packets_dropped_ = 0;
  last_sequence_number_ = 0;
  last_timestamp_ = 0;
  first_packet_received_ = false;
  gpu_raw_buffer_ = nullptr;
  gpu_rgba_buffer_ = nullptr;
  gpu_nv12_buffer_ = nullptr;
  cuda_stream_ = nullptr;

  assembling_index_.store(-1);
  last_emitted_rtp_timestamp_ = 0;
  first_frame_emitted_ = false;
  frames_dropped_ = 0;
}

void ST2110SourceOp::start() {
  HOLOSCAN_LOG_INFO("ST2110SourceOp::start()");

  CUDA_TRY(cudaSetDevice(0));
  CUDA_TRY(cudaFree(0));

  create_socket();
  allocate_buffers();

  HOLOSCAN_LOG_INFO("ST2110 source started successfully");
}

void ST2110SourceOp::stop() {
  HOLOSCAN_LOG_INFO("ST2110SourceOp::stop()");
  close_socket();
  free_buffers();
}

void ST2110SourceOp::allocate_buffers() {
  HOLOSCAN_LOG_INFO("Allocating triple-buffer with CUDA pinned memory...");

  constexpr size_t BUFFER_ALIGNMENT = 0x10000;
  size_t aligned_frame_size = (raw_frame_size_ + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);

  for (int i = 0; i < num_frame_buffers_; i++) {
    CUresult cu_result = cuMemHostAlloc(&frame_buffers_[i].pinned_memory,
                                         aligned_frame_size,
                                         CU_MEMHOSTALLOC_WRITECOMBINED);
    if (cu_result != CUDA_SUCCESS) {
      throw std::runtime_error(fmt::format("cuMemHostAlloc failed, cu_result={}",
                                            static_cast<int>(cu_result)));
    }

    frame_buffers_[i].current_timestamp = 0;
    frame_buffers_[i].packets_received = 0;
    frame_buffers_[i].bytes_received = 0;
    frame_buffers_[i].frame_complete = false;

    // Initialize buffer with pattern to detect missing data
    std::memset(frame_buffers_[i].pinned_memory, 0x00, aligned_frame_size);
  }

  // All buffers start as available (frame_complete = false)
  // assembling_index_ starts at -1 (will be set when first packet arrives)

  // Allocate GPU raw buffer (always allocated for raw YCbCr-4:2:2-10bit data)
  if (CUDA_TRY(cudaMalloc(&gpu_raw_buffer_, raw_frame_size_)) != cudaSuccess) {
    throw std::runtime_error("Failed to allocate GPU raw buffer (" +
                             std::to_string(raw_frame_size_) + " bytes)");
  }
  HOLOSCAN_LOG_INFO("Allocated GPU raw buffer: {} bytes ({} MB)",
                    raw_frame_size_,
                    raw_frame_size_ / (1024 * 1024));

  // Allocate GPU RGBA buffer (only if RGBA output is enabled)
  if (rgba_frame_size_ > 0) {
    if (CUDA_TRY(cudaMalloc(&gpu_rgba_buffer_, rgba_frame_size_)) != cudaSuccess) {
      throw std::runtime_error("Failed to allocate GPU RGBA buffer (" +
                               std::to_string(rgba_frame_size_) + " bytes)");
    }
    HOLOSCAN_LOG_INFO("Allocated GPU RGBA buffer: {} bytes ({} MB)",
                      rgba_frame_size_,
                      rgba_frame_size_ / (1024 * 1024));
  }

  // Allocate GPU NV12 buffer (only if NV12 output is enabled)
  if (nv12_frame_size_ > 0) {
    if (CUDA_TRY(cudaMalloc(&gpu_nv12_buffer_, nv12_frame_size_)) != cudaSuccess) {
      throw std::runtime_error("Failed to allocate GPU NV12 buffer (" +
                               std::to_string(nv12_frame_size_) + " bytes)");
    }
    HOLOSCAN_LOG_INFO("Allocated GPU NV12 buffer: {} bytes ({} MB)",
                      nv12_frame_size_,
                      nv12_frame_size_ / (1024 * 1024));
  }

  // Create CUDA stream for async H2D copy and conversions
  if (CUDA_TRY(cudaStreamCreate(&cuda_stream_)) != cudaSuccess) {
    throw std::runtime_error("Failed to create CUDA stream");
  }

  HOLOSCAN_LOG_INFO("Allocated {} pinned host buffers of {} bytes (aligned to 64k)",
                    num_frame_buffers_, aligned_frame_size);
}

void ST2110SourceOp::free_buffers() {
  // Free pinned memory buffers
  for (int i = 0; i < num_frame_buffers_; i++) {
    if (frame_buffers_[i].pinned_memory) {
      cuMemFreeHost(frame_buffers_[i].pinned_memory);
      frame_buffers_[i].pinned_memory = nullptr;
    }
  }

  // Free GPU raw buffer
  if (gpu_raw_buffer_) {
    cudaFree(gpu_raw_buffer_);
    gpu_raw_buffer_ = nullptr;
  }

  // Free GPU RGBA buffer
  if (gpu_rgba_buffer_) {
    cudaFree(gpu_rgba_buffer_);
    gpu_rgba_buffer_ = nullptr;
  }

  // Free GPU NV12 buffer
  if (gpu_nv12_buffer_) {
    cudaFree(gpu_nv12_buffer_);
    gpu_nv12_buffer_ = nullptr;
  }

  // Destroy CUDA stream
  if (cuda_stream_) {
    cudaStreamDestroy(cuda_stream_);
    cuda_stream_ = nullptr;
  }
}

void ST2110SourceOp::create_socket() {
  HOLOSCAN_LOG_INFO("Creating UDP socket for multicast reception...");

  // Create UDP socket with non-blocking flag
  socket_fd_ = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0);
  if (socket_fd_ < 0) {
    throw std::runtime_error(fmt::format("Failed to create socket: {}", strerror(errno)));
  }

  // Set socket to reuse address
  int reuse = 1;
  if (setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
    close(socket_fd_);
    throw std::runtime_error(fmt::format("Failed to set SO_REUSEADDR: {}", strerror(errno)));
  }

  // Bind to multicast port
  struct sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port_.get());
  addr.sin_addr.s_addr = INADDR_ANY;

  if (bind(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    close(socket_fd_);
    throw std::runtime_error(fmt::format("Failed to bind socket: {}", strerror(errno)));
  }

  // Join multicast group on specified interface
  struct ip_mreqn mreq = {};
  if (inet_pton(AF_INET, multicast_address_.get().c_str(), &mreq.imr_multiaddr) != 1) {
    close(socket_fd_);
    throw std::runtime_error(fmt::format("Invalid multicast address: {}", multicast_address_.get()));
  }

  mreq.imr_address.s_addr = INADDR_ANY;
  mreq.imr_ifindex = if_nametoindex(interface_name_.get().c_str());
  if (mreq.imr_ifindex == 0) {
    close(socket_fd_);
    throw std::runtime_error(fmt::format("Invalid interface name: {}", interface_name_.get()));
  }

  if (setsockopt(socket_fd_, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
    close(socket_fd_);
    throw std::runtime_error(fmt::format("Failed to join multicast group: {}", strerror(errno)));
  }

  // 256MB socket buffer for high-bandwidth streams (1080p50 @ ~3 Gbps)
  int socket_buffer_size = 256 * 1024 * 1024;
  if (setsockopt(socket_fd_, SOL_SOCKET, SO_RCVBUF, &socket_buffer_size, sizeof(socket_buffer_size)) < 0) {
    HOLOSCAN_LOG_WARN("Failed to set socket buffer size to {}: {}", socket_buffer_size, strerror(errno));
  }

  int actual_buffer_size = 0;
  socklen_t optlen = sizeof(actual_buffer_size);
  if (getsockopt(socket_fd_, SOL_SOCKET, SO_RCVBUF, &actual_buffer_size, &optlen) == 0) {
    HOLOSCAN_LOG_INFO("Socket receive buffer: requested {} MB, got {} MB",
                      socket_buffer_size / (1024*1024),
                      actual_buffer_size / (1024*1024));
    if (actual_buffer_size < socket_buffer_size / 2) {
      HOLOSCAN_LOG_WARN("Socket buffer significantly limited by system. Check net.core.rmem_max");
      HOLOSCAN_LOG_WARN("  Run: sudo sysctl -w net.core.rmem_max=268435456");
    }
  }

  HOLOSCAN_LOG_INFO("Socket created successfully:");
  HOLOSCAN_LOG_INFO("  Multicast group: {}", multicast_address_.get());
  HOLOSCAN_LOG_INFO("  Port: {}", port_.get());
  HOLOSCAN_LOG_INFO("  Interface: {} (index {})", interface_name_.get(), mreq.imr_ifindex);
}

void ST2110SourceOp::close_socket() {
  if (socket_fd_ >= 0) {
    close(socket_fd_);
    socket_fd_ = -1;
  }
}

void ST2110SourceOp::configure_stream_format() {
  std::string format = stream_format_.get();

  if (format == "YCbCr-4:2:2-10bit") {
    detected_format_.sampling = "YCbCr-4:2:2";
    detected_format_.depth = 10;
    detected_format_.pgroup = 5;
    detected_format_.bytes_per_pixel = 2.5;
    detected_format_.colorimetry = "BT709";
    detected_format_.detected = true;
  } else if (format == "YCbCr-4:2:2-8bit") {
    detected_format_.sampling = "YCbCr-4:2:2";
    detected_format_.depth = 8;
    detected_format_.pgroup = 4;
    detected_format_.bytes_per_pixel = 2.0;
    detected_format_.colorimetry = "BT709";
    detected_format_.detected = true;
  } else if (format == "RGBA-8bit") {
    detected_format_.sampling = "RGBA";
    detected_format_.depth = 8;
    detected_format_.pgroup = 4;
    detected_format_.bytes_per_pixel = 4.0;
    detected_format_.colorimetry = "BT709";
    detected_format_.detected = true;
  } else {
    HOLOSCAN_LOG_WARN("Unknown stream_format '{}', defaulting to YCbCr-4:2:2-10bit", format);
    detected_format_.sampling = "YCbCr-4:2:2";
    detected_format_.depth = 10;
    detected_format_.pgroup = 5;
    detected_format_.bytes_per_pixel = 2.5;
    detected_format_.colorimetry = "BT709";
    detected_format_.detected = true;
  }

  raw_frame_size_ = width_.get() * height_.get() * detected_format_.bytes_per_pixel;

  HOLOSCAN_LOG_INFO("ST2110 stream format configured:");
  HOLOSCAN_LOG_INFO("  Sampling: {}", detected_format_.sampling);
  HOLOSCAN_LOG_INFO("  Depth: {} bit", detected_format_.depth);
  HOLOSCAN_LOG_INFO("  Pixel group: {} bytes", detected_format_.pgroup);
  HOLOSCAN_LOG_INFO("  Bytes per pixel: {}", detected_format_.bytes_per_pixel);
  HOLOSCAN_LOG_INFO("  Colorimetry: {}", detected_format_.colorimetry);
  HOLOSCAN_LOG_INFO("  Raw frame size: {} bytes", raw_frame_size_);
}

void ST2110SourceOp::parse_and_copy_packet(const uint8_t* packet_data, size_t packet_size) {
  // Packet structure: [RTP 12B][ST2110 12B][Payload ~1450B]
  constexpr size_t MIN_PACKET_SIZE = 24;

  if (!logged_first_packet_) {
    HOLOSCAN_LOG_INFO("First packet received: {} bytes", packet_size);
    logged_first_packet_ = true;
  }

  if (packet_size < MIN_PACKET_SIZE) {
    HOLOSCAN_LOG_WARN("Packet too small: {} bytes (need at least {})", packet_size, MIN_PACKET_SIZE);
    return;
  }

  // RTP header is at the start of UDP payload (no L2-L4 headers to skip)
  const uint8_t* rtp_start = packet_data;
  size_t remaining_size = packet_size;

  // Parse RTP header
  const RTPHeader* rtp = reinterpret_cast<const RTPHeader*>(rtp_start);
  uint16_t sequence_number = ntohs(rtp->sequence_number);
  uint32_t timestamp = ntohl(rtp->timestamp);
  bool marker = (rtp->marker_payload_type & 0x80) != 0;

  // Skip to ST2110 header (after RTP header)
  const uint8_t* st2110_start = rtp_start + rtp_header_size_.get();
  remaining_size -= rtp_header_size_.get();

  if (remaining_size < sizeof(ST2110Header)) {
    return;
  }

  // Parse ST2110 header (SMPTE ST 2110-20 format)
  const ST2110Header* st2110 = reinterpret_cast<const ST2110Header*>(st2110_start);

  // Extract fields from network byte order
  uint16_t line_length = ntohs(st2110->line_length);
  uint16_t line_and_field = ntohs(st2110->line_and_field);
  uint16_t offset_and_cont = ntohs(st2110->offset_and_continuation);

  // Extract line number (lower 15 bits) and field bit (top bit)
  uint16_t line_number = line_and_field & 0x7FFF;
  bool field_bit = (line_and_field & 0x8000) != 0;

  // Extract offset (lower 15 bits) and continuation bit (top bit)
  uint16_t line_offset = offset_and_cont & 0x7FFF;
  bool continuation_bit = (offset_and_cont & 0x8000) != 0;

  // Debug: Log ST2110 header values for first few packets
  if (debug_packet_count_ < 5) {
    HOLOSCAN_LOG_INFO("ST2110 header debug [packet {}]: line={}, offset={}, length={}, F={}, C={}",
                      debug_packet_count_, line_number, line_offset, line_length, field_bit, continuation_bit);
    HOLOSCAN_LOG_INFO("  Raw bytes at ST2110 start: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
                      st2110_start[0], st2110_start[1], st2110_start[2], st2110_start[3],
                      st2110_start[4], st2110_start[5], st2110_start[6], st2110_start[7]);
    debug_packet_count_++;
  }

  // Skip to payload (after ST2110 header)
  const uint8_t* payload = st2110_start + sizeof(ST2110Header);
  size_t payload_size = remaining_size - sizeof(ST2110Header);

  // Validate line number and offset
  if (line_number >= height_.get()) {
    if (rejected_count_ < 3) {
      HOLOSCAN_LOG_WARN("Rejecting packet: line_number {} >= height {}", line_number, height_.get());
      rejected_count_++;
    }
    return;
  }

  // Track packet loss
  if (first_packet_received_) {
    uint16_t expected_seq = (last_sequence_number_ + 1) & 0xFFFF;
    if (sequence_number != expected_seq) {
      uint16_t dropped = (sequence_number - expected_seq) & 0xFFFF;
      total_packets_dropped_ += dropped;
    }
  }
  last_sequence_number_ = sequence_number;
  first_packet_received_ = true;

  // Get or allocate assembling buffer
  FrameBuffer* recv_buffer = get_assembling_buffer();
  if (!recv_buffer) {
    // All buffers full - drop this packet (rare, indicates we're falling behind)
    return;
  }

  int current_assembling_idx = assembling_index_.load();

  // Check for frame boundary (new timestamp)
  if (recv_buffer->current_timestamp != 0 && timestamp != recv_buffer->current_timestamp) {
    // Mark current frame complete
    recv_buffer->frame_complete = true;
    HOLOSCAN_LOG_DEBUG("Frame complete (timestamp change): RTP={}, packets={}, bytes={}",
                       recv_buffer->current_timestamp,
                       recv_buffer->packets_received,
                       recv_buffer->bytes_received);

    // Release assembling buffer and get new one for next frame
    assembling_index_.store(-1);
    recv_buffer = get_assembling_buffer();
    if (!recv_buffer) {
      return;  // No free buffers available
    }

    // Initialize new buffer for new frame
    recv_buffer->current_timestamp = timestamp;
    recv_buffer->packets_received = 0;
    recv_buffer->bytes_received = 0;
    recv_buffer->frame_complete = false;
    std::memset(recv_buffer->pinned_memory, 0x00, raw_frame_size_);
  } else if (recv_buffer->current_timestamp == 0) {
    // First packet of frame
    recv_buffer->current_timestamp = timestamp;
  }

  // Calculate destination offset in frame buffer using integer pgroup arithmetic
  // Each pgroup covers pgroup_pixels pixels in pgroup_bytes bytes (avoids floating-point truncation)
  uint32_t pgroup_bytes = detected_format_.pgroup;
  uint32_t pgroup_pixels = (detected_format_.sampling == "RGBA") ? 1 : 2;
  size_t line_bytes = static_cast<size_t>(width_.get() / pgroup_pixels) * pgroup_bytes;
  size_t line_start = line_number * line_bytes;
  size_t dest_offset = line_start + static_cast<size_t>(line_offset / pgroup_pixels) * pgroup_bytes;

  // Bounds check (use raw_frame_size_ for raw buffer)
  if (dest_offset + payload_size <= raw_frame_size_) {
    // Copy payload to pinned memory
    uint8_t* dest = static_cast<uint8_t*>(recv_buffer->pinned_memory) + dest_offset;
    std::memcpy(dest, payload, payload_size);

    recv_buffer->packets_received++;
    recv_buffer->bytes_received += payload_size;
  }

  // Check for RTP marker bit (indicates last packet of frame)
  if (marker && recv_buffer->current_timestamp != 0) {
    // Mark frame complete
    recv_buffer->frame_complete = true;
    HOLOSCAN_LOG_DEBUG("Frame complete (marker bit): RTP={}, packets={}, bytes={}",
                       recv_buffer->current_timestamp,
                       recv_buffer->packets_received,
                       recv_buffer->bytes_received);

    // Release assembling buffer (will be allocated again for next frame)
    assembling_index_.store(-1);
  }

  last_timestamp_ = timestamp;
  total_packets_received_++;
  total_bytes_received_ += payload_size;
}

// Circular queue helper methods

ST2110SourceOp::FrameBuffer* ST2110SourceOp::get_assembling_buffer() {
  // Check if we already have an assembling buffer
  int current_idx = assembling_index_.load();
  if (current_idx >= 0 && current_idx < num_frame_buffers_) {
    return &frame_buffers_[current_idx];
  }

  // Find a free buffer (frame_complete == false)
  for (int i = 0; i < num_frame_buffers_; i++) {
    if (!frame_buffers_[i].frame_complete) {
      // Try to atomically claim this buffer
      int expected = -1;
      if (assembling_index_.compare_exchange_strong(expected, i)) {
        return &frame_buffers_[i];
      }
    }
  }

  // All buffers are full (we're falling behind)
  HOLOSCAN_LOG_WARN("All frame buffers full - dropping packets until queue drains");
  return nullptr;
}

ST2110SourceOp::FrameBuffer* ST2110SourceOp::find_next_frame_to_emit() {
  // If this is the first frame, emit any complete frame
  if (!first_frame_emitted_) {
    for (int i = 0; i < num_frame_buffers_; i++) {
      if (frame_buffers_[i].frame_complete) {
        return &frame_buffers_[i];
      }
    }
    return nullptr;
  }

  // ST 2110 RTP timestamps increment by fixed amount per frame
  // At 50fps with 90kHz clock: delta = 90000/50 = 1800
  // At 60fps with 90kHz clock: delta = 90000/60 = 1500
  uint32_t timestamp_delta = 90000 / framerate_.get();
  uint32_t expected_timestamp = last_emitted_rtp_timestamp_ + timestamp_delta;

  // Look for exact match first (preferred - maintains sequence)
  for (int i = 0; i < num_frame_buffers_; i++) {
    if (frame_buffers_[i].frame_complete &&
        frame_buffers_[i].current_timestamp == expected_timestamp) {
      return &frame_buffers_[i];
    }
  }

  // Look for newer frames (handles dropped frames)
  uint32_t newest_timestamp = 0;
  int newest_idx = -1;

  for (int i = 0; i < num_frame_buffers_; i++) {
    if (frame_buffers_[i].frame_complete) {
      uint32_t ts = frame_buffers_[i].current_timestamp;
      // Check if this timestamp is newer than expected (accounting for 32-bit wraparound)
      int32_t diff = static_cast<int32_t>(ts - expected_timestamp);
      if (diff > 0 && (newest_idx < 0 || ts > newest_timestamp)) {
        newest_timestamp = ts;
        newest_idx = i;
      }
    }
  }

  if (newest_idx >= 0) {
    // Found a newer frame - we've skipped some frames
    uint32_t frames_skipped = (newest_timestamp - expected_timestamp) / timestamp_delta;
    HOLOSCAN_LOG_WARN("Skipping {} frames in queue (expected RTP={}, got RTP={})",
                      frames_skipped, expected_timestamp, newest_timestamp);
    frames_dropped_ += frames_skipped;
    return &frame_buffers_[newest_idx];
  }

  // No complete frames available
  return nullptr;
}

void ST2110SourceOp::mark_frame_emitted(int buffer_index) {
  if (buffer_index < 0 || buffer_index >= num_frame_buffers_) {
    return;
  }

  FrameBuffer* buffer = &frame_buffers_[buffer_index];

  buffer->frame_complete = false;
  buffer->current_timestamp = 0;
  buffer->packets_received = 0;
  buffer->bytes_received = 0;

  // Note: Don't clear pinned_memory - causes race with parse_and_copy_packet()
}

void ST2110SourceOp::drain_socket() {
  constexpr int MAX_BATCH = 256;
  constexpr size_t PACKET_BUFFER_SIZE = 2048;

  // Use thread_local storage to avoid large stack allocations (~512KB)
  static thread_local std::vector<uint8_t> packet_storage(MAX_BATCH * PACKET_BUFFER_SIZE);
  static thread_local struct mmsghdr msgs[MAX_BATCH];
  static thread_local struct iovec iovecs[MAX_BATCH];
  static thread_local bool initialized = false;

  if (!initialized) {
    memset(msgs, 0, sizeof(msgs));
    for (int i = 0; i < MAX_BATCH; i++) {
      iovecs[i].iov_base = &packet_storage[i * PACKET_BUFFER_SIZE];
      iovecs[i].iov_len = PACKET_BUFFER_SIZE;
      msgs[i].msg_hdr.msg_iov = &iovecs[i];
      msgs[i].msg_hdr.msg_iovlen = 1;
    }
    initialized = true;
  }

  uint32_t total_to_receive = batch_size_.get();
  uint32_t total_received = 0;

  while (total_received < total_to_receive) {
    int batch_count = std::min(MAX_BATCH, static_cast<int>(total_to_receive - total_received));
    int num_received = recvmmsg(socket_fd_, msgs, batch_count, MSG_DONTWAIT, nullptr);

    if (num_received < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        // Socket empty - normal condition
        break;
      } else {
        HOLOSCAN_LOG_ERROR("recvmmsg() error: {}", strerror(errno));
        break;
      }
    }

    if (num_received == 0) {
      break;  // No packets received
    }

    // Process all received packets
    for (int i = 0; i < num_received; i++) {
      parse_and_copy_packet(&packet_storage[i * PACKET_BUFFER_SIZE], msgs[i].msg_len);
    }

    total_received += num_received;

    // If we got fewer packets than requested, socket is empty
    if (num_received < batch_count) {
      break;
    }
  }
}

bool ST2110SourceOp::emit_next_frame() {
  // Find the next frame to emit in RTP timestamp sequence order
  FrameBuffer* frame_to_emit = find_next_frame_to_emit();
  if (!frame_to_emit) {
    // No complete frames ready - this is normal during startup or if we're ahead of network
    return false;
  }

  // Calculate buffer index for mark_frame_emitted()
  int buffer_index = frame_to_emit - &frame_buffers_[0];

  HOLOSCAN_LOG_DEBUG("Emitting frame: RTP={}, packets={}, bytes={}",
                     frame_to_emit->current_timestamp,
                     frame_to_emit->packets_received,
                     frame_to_emit->bytes_received);

  // Step 1: Copy raw frame from pinned memory to GPU
  CUDA_TRY(cudaMemcpyAsync(gpu_raw_buffer_,
                           frame_to_emit->pinned_memory,
                           raw_frame_size_,
                           cudaMemcpyHostToDevice,
                           cuda_stream_));
  CUDA_TRY(cudaStreamSynchronize(cuda_stream_));

  // Step 2: Zero the pinned buffer now that GPU has the data
  // This prevents stale data from previous frames appearing if we miss packets
  // Safe to do here because buffer is still "locked" (current_timestamp != 0)
  // so parse_and_copy_packet() won't write to it yet
  std::memset(frame_to_emit->pinned_memory, 0x00, raw_frame_size_);

  // Step 3: Always emit raw output (will be implemented by emit_raw_frame)
  // Step 4: Optionally convert and emit video_buffer_output (implemented by emit_video_frame)
  // These will be called from compute() after this returns

  // Update state
  last_emitted_rtp_timestamp_ = frame_to_emit->current_timestamp;
  first_frame_emitted_ = true;

  // Mark this buffer as emitted and reusable (now safe for new packets)
  mark_frame_emitted(buffer_index);

  return true;  // Frame ready and copied to GPU
}

void ST2110SourceOp::emit_raw_frame(OutputContext& op_output) {
  // Emit raw frame buffer with format metadata
  // Note: GPU buffer already populated by emit_next_frame()

  HOLOSCAN_LOG_DEBUG("Emitting raw frame: format={} {}bit",
                     detected_format_.sampling,
                     detected_format_.depth);

  // Create GXF Entity with Tensor
  auto maybe_entity = nvidia::gxf::Entity::New(fragment()->executor().context());
  if (!maybe_entity) {
    HOLOSCAN_LOG_ERROR("Failed to create GXF entity for raw output");
    return;
  }

  auto entity = maybe_entity.value();

  // Add Tensor component for raw video data
  auto maybe_tensor = entity.add<nvidia::gxf::Tensor>("raw_video");
  if (!maybe_tensor) {
    HOLOSCAN_LOG_ERROR("Failed to add Tensor component");
    return;
  }

  auto tensor = maybe_tensor.value();

  // Setup tensor with raw video data
  // Shape: [height, width * bytes_per_pixel]
  // For YCbCr-4:2:2 10-bit: [1080, 4800] (1920 * 2.5)
  nvidia::gxf::Shape shape{static_cast<int32_t>(height_.get()),
                           static_cast<int32_t>(width_.get() * detected_format_.bytes_per_pixel)};

  // Wrap GPU raw buffer in tensor
  tensor->wrapMemory(shape,
                     nvidia::gxf::PrimitiveType::kUnsigned8,
                     1,  // bytes per element (uint8)
                     nvidia::gxf::ComputeTrivialStrides(shape, 1),
                     nvidia::gxf::MemoryStorageType::kDevice,
                     gpu_raw_buffer_,
                     [](void*) { return nvidia::gxf::Success; });  // No deallocation (we manage it)

  // Add metadata as tensor attributes (if supported) or log it
  // Note: For full metadata support, downstream operators can query detected_format_

  // Emit on raw_output port
  op_output.emit(entity, "raw_output");

  HOLOSCAN_LOG_DEBUG("Raw frame emitted successfully");
}

void ST2110SourceOp::convert_and_emit_rgba(OutputContext& op_output) {
  HOLOSCAN_LOG_DEBUG("Converting {} → RGBA", detected_format_.sampling);

  // Step 1: Convert raw YCbCr to RGBA using CUDA kernel
  convert_ycbcr422_10bit_to_rgba(gpu_rgba_buffer_,
                                  gpu_raw_buffer_,
                                  width_.get(),
                                  height_.get(),
                                  cuda_stream_);
  CUDA_TRY(cudaStreamSynchronize(cuda_stream_));

  // Step 2: Create GXF entity with VideoBuffer
  auto maybe_entity = nvidia::gxf::Entity::New(fragment()->executor().context());
  if (!maybe_entity) {
    HOLOSCAN_LOG_ERROR("Failed to create GXF entity for RGBA output");
    return;
  }

  auto entity = maybe_entity.value();

  auto maybe_video_buffer = entity.add<nvidia::gxf::VideoBuffer>("video");
  if (!maybe_video_buffer) {
    HOLOSCAN_LOG_ERROR("Failed to add VideoBuffer component");
    return;
  }

  auto video_buffer = maybe_video_buffer.value();

  nvidia::gxf::VideoBufferInfo video_info;
  video_info.width = width_.get();
  video_info.height = height_.get();
  video_info.color_format = nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA;
  video_info.color_planes.resize(1);
  video_info.color_planes[0].width = width_.get();
  video_info.color_planes[0].height = height_.get();
  video_info.color_planes[0].size = rgba_frame_size_;
  video_info.color_planes[0].stride = width_.get() * 4;  // RGBA = 4 bytes/pixel

  video_buffer->wrapMemory(video_info,
                           rgba_frame_size_,
                           nvidia::gxf::MemoryStorageType::kDevice,
                           gpu_rgba_buffer_,
                           nullptr);

  // Step 3: Add Tensor component for Python operators (GXF allocator prevents races)
  auto maybe_tensor = entity.add<nvidia::gxf::Tensor>("video_tensor");
  if (maybe_tensor) {
    auto tensor = maybe_tensor.value();

    nvidia::gxf::Shape shape{
      static_cast<int32_t>(height_.get()),
      static_cast<int32_t>(width_.get() * 4)  // RGBA = 4 bytes/pixel
    };

    if (!tensor_allocator_.get()) {
      HOLOSCAN_LOG_ERROR("Tensor allocator is null");
      return;
    }

    auto gxf_cid = tensor_allocator_.get()->gxf_cid();
    auto maybe_gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      fragment()->executor().context(), gxf_cid);

    if (!maybe_gxf_allocator) {
      HOLOSCAN_LOG_ERROR("Failed to get GXF allocator handle (cid={})", gxf_cid);
      return;
    }

    auto result = tensor->reshapeCustom(
      shape,
      nvidia::gxf::PrimitiveType::kUnsigned8,
      1,
      nvidia::gxf::ComputeTrivialStrides(shape, 1),
      nvidia::gxf::MemoryStorageType::kDevice,
      maybe_gxf_allocator.value()
    );

    if (!result) {
      HOLOSCAN_LOG_ERROR("Failed to allocate Tensor from pool");
      return;
    }

    CUDA_TRY(cudaMemcpyAsync(
      tensor->pointer(),
      gpu_rgba_buffer_,
      rgba_frame_size_,
      cudaMemcpyDeviceToDevice,
      cuda_stream_
    ));

    CUDA_TRY(cudaStreamSynchronize(cuda_stream_));

    HOLOSCAN_LOG_DEBUG("Added RGBA Tensor component (GPU copy: {} bytes)", rgba_frame_size_);
  }

  // Step 4: Emit on rgba_output port
  op_output.emit(entity, "rgba_output");

  if (++rgba_frame_count_ % 100 == 0) {
    HOLOSCAN_LOG_INFO("Emitted RGBA frame {}: {} total packets received, {} dropped",
                      rgba_frame_count_,
                      total_packets_received_,
                      total_packets_dropped_);
  }
}

void ST2110SourceOp::convert_and_emit_nv12(OutputContext& op_output) {
  HOLOSCAN_LOG_DEBUG("Converting {} → NV12", detected_format_.sampling);

  // Step 1: Convert raw YCbCr to NV12 using CUDA kernel
  convert_ycbcr422_10bit_to_nv12(gpu_nv12_buffer_,
                                  gpu_raw_buffer_,
                                  width_.get(),
                                  height_.get(),
                                  cuda_stream_);
  CUDA_TRY(cudaStreamSynchronize(cuda_stream_));

  // Step 2: Create GXF entity with VideoBuffer
  auto maybe_entity = nvidia::gxf::Entity::New(fragment()->executor().context());
  if (!maybe_entity) {
    HOLOSCAN_LOG_ERROR("Failed to create GXF entity for NV12 output");
    return;
  }

  auto entity = maybe_entity.value();

  auto maybe_video_buffer = entity.add<nvidia::gxf::VideoBuffer>("video");
  if (!maybe_video_buffer) {
    HOLOSCAN_LOG_ERROR("Failed to add VideoBuffer component");
    return;
  }

  auto video_buffer = maybe_video_buffer.value();

  // Use VideoFormatSize to get the correct color plane configuration for NV12
  nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12> video_type;
  nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12> color_format;
  auto color_planes = color_format.getDefaultColorPlanes(width_.get(), height_.get(), false);

  nvidia::gxf::VideoBufferInfo video_info;
  video_info.width = width_.get();
  video_info.height = height_.get();
  video_info.color_format = video_type.value;
  video_info.color_planes = std::move(color_planes);

  video_buffer->wrapMemory(video_info,
                           nv12_frame_size_,
                           nvidia::gxf::MemoryStorageType::kDevice,
                           gpu_nv12_buffer_,
                           nullptr);

  // Step 3: Add Tensor component for Python operators (GXF allocator prevents races)
  auto maybe_tensor = entity.add<nvidia::gxf::Tensor>("video_tensor");
  if (maybe_tensor) {
    auto tensor = maybe_tensor.value();

    nvidia::gxf::Shape shape{
      static_cast<int32_t>(height_.get() * 3 / 2),  // NV12: Y + UV/2
      static_cast<int32_t>(width_.get())
    };

    if (!tensor_allocator_.get()) {
      HOLOSCAN_LOG_ERROR("Tensor allocator is null");
      return;
    }

    auto gxf_cid = tensor_allocator_.get()->gxf_cid();
    auto maybe_gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      fragment()->executor().context(), gxf_cid);

    if (!maybe_gxf_allocator) {
      HOLOSCAN_LOG_ERROR("Failed to get GXF allocator handle (cid={})", gxf_cid);
      return;
    }

    auto result = tensor->reshapeCustom(
      shape,
      nvidia::gxf::PrimitiveType::kUnsigned8,
      1,
      nvidia::gxf::ComputeTrivialStrides(shape, 1),
      nvidia::gxf::MemoryStorageType::kDevice,
      maybe_gxf_allocator.value()
    );

    if (!result) {
      HOLOSCAN_LOG_ERROR("Failed to allocate Tensor from pool");
      return;
    }

    CUDA_TRY(cudaMemcpyAsync(
      tensor->pointer(),
      gpu_nv12_buffer_,
      nv12_frame_size_,
      cudaMemcpyDeviceToDevice,
      cuda_stream_
    ));

    CUDA_TRY(cudaStreamSynchronize(cuda_stream_));

    HOLOSCAN_LOG_DEBUG("Added NV12 Tensor component (GPU copy: {} bytes)", nv12_frame_size_);
  }

  // Step 4: Emit on nv12_output port
  op_output.emit(entity, "nv12_output");

  if (++nv12_frame_count_ % 100 == 0) {
    HOLOSCAN_LOG_INFO("Emitted NV12 frame {}: {} total packets received, {} dropped",
                      nv12_frame_count_,
                      total_packets_received_,
                      total_packets_dropped_);
  }
}

void ST2110SourceOp::compute(InputContext& op_input, OutputContext& op_output,
                             ExecutionContext& context) {
  // Phase 1: Drain socket (fills circular queue)
  // Phase 2: Emit oldest complete frame in RTP timestamp order
  //
  // This decouples network packet arrival (variable/bursty) from
  // frame emission (steady 50Hz via PeriodicCondition).
  //
  // Benefits:
  // - Absorbs network jitter via 2-3 frame buffer
  // - Emits frames in correct RTP sequence order
  // - Drops old frames if we fall behind
  // - No memcpy between ingestion and emission (virtual queue)
  // ========================================================================

  // Phase 1: Aggressively drain socket into circular queue
  drain_socket();

  // Log stats periodically
  if (++compute_count_ % 50 == 0) {  // Every 50 calls (~1 second at 50Hz)
    int queue_depth = 0;
    for (int i = 0; i < num_frame_buffers_; i++) {
      if (frame_buffers_[i].frame_complete) queue_depth++;
    }
    HOLOSCAN_LOG_INFO("compute #{}: packets={}, dropped={}, frames_dropped={}, queue_depth={}",
                      compute_count_,
                      total_packets_received_,
                      total_packets_dropped_,
                      frames_dropped_,
                      queue_depth);
  }

  // Phase 2: Emit next frame in RTP timestamp sequence (only if ready)
  // This function handles GPU copy and prepares buffers
  // Returns true if a frame was ready and copied to GPU
  if (!emit_next_frame()) {
    // No frame ready - normal during startup or when queue is empty
    // Continue draining socket on next compute() call
    return;
  }

  // Frame ready - emit downstream

  // Step 2: Always emit raw output
  emit_raw_frame(op_output);

  // Step 3: Optionally convert and emit RGBA
  if (enable_rgba_output_.get()) {
    if (detected_format_.depth != 10 || detected_format_.sampling != "YCbCr-4:2:2") {
      HOLOSCAN_LOG_ERROR("RGBA conversion only supports YCbCr-4:2:2-10bit, got {}-{}bit",
                         detected_format_.sampling, detected_format_.depth);
    } else {
      convert_and_emit_rgba(op_output);
    }
  }

  // Step 4: Optionally convert and emit NV12
  if (enable_nv12_output_.get()) {
    if (detected_format_.depth != 10 || detected_format_.sampling != "YCbCr-4:2:2") {
      HOLOSCAN_LOG_ERROR("NV12 conversion only supports YCbCr-4:2:2-10bit, got {}-{}bit",
                         detected_format_.sampling, detected_format_.depth);
    } else {
      convert_and_emit_nv12(op_output);
    }
  }
}

}  // namespace holoscan::ops
