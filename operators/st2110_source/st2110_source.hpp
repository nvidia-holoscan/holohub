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

#ifndef HOLOSCAN_OPERATORS_ST2110_SOURCE_ST2110_SOURCE_HPP
#define HOLOSCAN_OPERATORS_ST2110_SOURCE_ST2110_SOURCE_HPP

#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/io_context.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/conditions/gxf/periodic.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to receive SMPTE ST 2110-20 video streams via Linux sockets.
 *
 * This operator uses standard Linux UDP sockets with CUDA pinned memory to receive
 * ST 2110-20 uncompressed video streams over IP networks. It handles multicast
 * subscription, RTP packet reception, frame reassembly, and outputs VideoBuffer frames.
 *
 * This implementation follows the pattern used by NVIDIA's Holoscan Sensor Bridge for
 * efficient CPU-to-GPU transfer on platforms without DPDK support (e.g., Thor AGX MGBE).
 *
 * ==Named Outputs==
 *
 * - **raw_output** : `holoscan::gxf::Entity` with TensorMap
 *   - Raw video frame buffer with format metadata (always available)
 *   - Metadata includes: sampling, depth, pgroup, colorimetry, width, height, framerate
 * - **rgba_output** : `nvidia::gxf::VideoBuffer` (optional)
 *   - RGBA 8-bit converted frame (enabled via enable_rgba_output parameter)
 *   - Suitable for visualization with Holoviz
 * - **nv12_output** : `nvidia::gxf::VideoBuffer` (optional)
 *   - NV12 8-bit converted frame (enabled via enable_nv12_output parameter)
 *   - Suitable for video encoding
 *
 * ==Parameters==
 *
 * - **multicast_address**: Multicast IP address for ST 2110 stream (e.g., "239.255.66.60").
 *   Optional (default: "239.0.0.1").
 * - **port**: UDP port for ST 2110 stream. Optional (default: `5004`).
 * - **interface_name**: Linux network interface name (e.g., "mgbe0_0", "eth0").
 *   Optional (default: "eth0").
 * - **width**: Width of the video stream in pixels. Optional (default: `1920`).
 * - **height**: Height of the video stream in pixels. Optional (default: `1080`).
 * - **framerate**: Expected frame rate of the video stream. Optional (default: `60`).
 * - **stream_format**: Input stream format from ST 2110 source.
 *   Supported: "YCbCr-4:2:2-10bit" (default), "YCbCr-4:2:2-8bit", "RGBA-8bit".
 *   Optional (default: "YCbCr-4:2:2-10bit").
 * - **enable_rgba_output**: Enable RGBA conversion and emission on rgba_output port.
 *   Optional (default: false).
 * - **enable_nv12_output**: Enable NV12 conversion and emission on nv12_output port.
 *   Optional (default: false).
 * - **batch_size**: Number of packets to receive per compute() call. Optional (default: `1000`).
 * - **max_packet_size**: Maximum size of ST 2110 packets in bytes. Optional (default: `1514`).
 * - **header_size**: Size of L2-L4 headers (Ethernet + IP + UDP). Optional (default: `42`).
 * - **rtp_header_size**: Size of RTP header. Optional (default: `12`).
 * - **enable_reorder_kernel**: Enable CUDA kernel for packet reordering. Optional (default: `true`).
 *
 * ==Device Support==
 *
 * Supported on x86_64 and aarch64 (ARM64) platforms including NVIDIA Thor AGX.
 * Works with standard network interfaces including MGBE, no special NICs required.
 */
class ST2110SourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ST2110SourceOp)

  ST2110SourceOp() = default;
  ~ST2110SourceOp();

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  // ST 2110 specific helpers
  struct RTPHeader {
    uint8_t version_padding_extension;
    uint8_t marker_payload_type;
    uint16_t sequence_number;
    uint32_t timestamp;
    uint32_t ssrc;
  } __attribute__((packed));

  struct ST2110Header {
    uint16_t extended_sequence_number;
    uint16_t line_length;            // Length field (bytes 2-3)
    uint16_t line_and_field;         // F(1bit) + Line Number(15bits)  (bytes 4-5)
    uint16_t offset_and_continuation;  // C(1bit) + Offset(15bits) (bytes 6-7)
  } __attribute__((packed));

  // Stream format metadata (auto-detected from ST 2110 packets)
  struct StreamFormat {
    std::string sampling;     // e.g., "YCbCr-4:2:2"
    uint32_t depth;           // Bit depth (8, 10, 12, 16)
    uint32_t pgroup;          // Pixel group size in bytes
    std::string colorimetry;  // e.g., "BT709", "BT2020"
    double bytes_per_pixel;   // Calculated from pgroup
    bool detected;            // Format has been detected from stream
  };

  // Triple-buffer descriptor (following Sensor Bridge pattern)
  struct FrameBuffer {
    void* pinned_memory;         // CUDA pinned memory (CU_MEMHOSTALLOC_WRITECOMBINED)
    uint32_t current_timestamp;  // RTP timestamp of current frame
    uint32_t packets_received;   // Packets received for current frame
    uint32_t bytes_received;     // Bytes received for current frame
    std::atomic<bool> frame_complete;  // Frame reassembly complete (atomic for thread safety)
  };

  // Helper methods
  void allocate_buffers();
  void free_buffers();
  void create_socket();
  void close_socket();
  void parse_and_copy_packet(const uint8_t* packet_data, size_t packet_size);
  void configure_stream_format();  // Configure format from stream_format parameter
  void drain_socket();              // Aggressively drain socket into frame queue
  bool emit_next_frame();           // Emit oldest complete frame (true if frame ready)
  FrameBuffer* get_assembling_buffer();   // Get buffer currently being assembled
  FrameBuffer* find_next_frame_to_emit();  // Find next frame in RTP sequence
  void mark_frame_emitted(int buffer_index);  // Mark frame as emitted and reusable
  void emit_raw_frame(OutputContext& op_output);
  void convert_and_emit_rgba(OutputContext& op_output);
  void convert_and_emit_nv12(OutputContext& op_output);

  // Parameters
  Parameter<holoscan::IOSpec*> raw_output_;
  Parameter<holoscan::IOSpec*> rgba_output_;
  Parameter<holoscan::IOSpec*> nv12_output_;
  Parameter<std::string> multicast_address_;
  Parameter<uint16_t> port_;
  Parameter<std::string> interface_name_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<uint32_t> framerate_;
  Parameter<std::string> stream_format_;
  Parameter<bool> enable_rgba_output_;
  Parameter<bool> enable_nv12_output_;
  Parameter<uint32_t> batch_size_;
  Parameter<uint16_t> max_packet_size_;
  Parameter<uint16_t> header_size_;
  Parameter<uint16_t> rtp_header_size_;
  Parameter<bool> enable_reorder_kernel_;
  Parameter<std::shared_ptr<PeriodicCondition>> periodic_condition_;

  // Socket state
  int socket_fd_ = -1;

  // Internal state
  uint16_t payload_size_ = 0;  // max_packet_size - header_size - rtp_header_size
  int64_t total_bytes_received_ = 0;
  int64_t total_packets_received_ = 0;
  int64_t total_packets_dropped_ = 0;

  // Circular queue architecture (virtual queue - no memcpy)
  static constexpr int num_frame_buffers_ = 3;
  std::array<FrameBuffer, num_frame_buffers_> frame_buffers_;
  std::atomic<int> assembling_index_{-1};  // Index of buffer being assembled (-1 if none)
  uint32_t last_emitted_rtp_timestamp_ = 0;  // RTP timestamp of last emitted frame
  bool first_frame_emitted_ = false;        // Track if we've emitted our first frame
  uint32_t frames_dropped_ = 0;             // Count of frames dropped due to queue overflow

  // RTP sequence tracking for packet loss detection
  uint16_t last_sequence_number_ = 0;
  uint32_t last_timestamp_ = 0;
  bool first_packet_received_ = false;

  // CUDA resources
  void* gpu_raw_buffer_ = nullptr;       // GPU buffer for raw YCbCr-4:2:2-10bit data
  void* gpu_rgba_buffer_ = nullptr;      // GPU buffer for RGBA 8-bit data (if RGBA output enabled)
  void* gpu_nv12_buffer_ = nullptr;      // GPU buffer for NV12 8-bit data (if NV12 output enabled)
  cudaStream_t cuda_stream_ = nullptr;   // CUDA stream for async H2D copy and conversions

  // GXF allocator for Tensor memory management (prevents race conditions)
  Parameter<std::shared_ptr<Allocator>> tensor_allocator_;

  // Video format info
  size_t rgba_frame_size_ = 0;    // Size of RGBA frame buffer (if RGBA output enabled)
  size_t nv12_frame_size_ = 0;    // Size of NV12 frame buffer (if NV12 output enabled)

  // Stream format detection
  StreamFormat detected_format_;
  size_t raw_frame_size_ = 0;  // Size of raw frame buffer (based on stream_format)

  // Debug/stats counters (replaces static locals for multi-instance safety)
  bool logged_first_packet_ = false;
  int debug_packet_count_ = 0;
  int rejected_count_ = 0;
  int rgba_frame_count_ = 0;
  int nv12_frame_count_ = 0;
  uint64_t compute_count_ = 0;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_ST2110_SOURCE_ST2110_SOURCE_HPP */
