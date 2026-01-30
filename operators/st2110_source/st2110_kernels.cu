/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 XRlabs. All rights reserved.
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

#include "st2110_kernels.cuh"

/**
 * @brief CUDA kernel for ST 2110-20 frame reassembly
 *
 * Each block processes one packet. Threads within the block copy the packet's
 * payload to the correct line in the output frame.
 */
__global__ void st2110_frame_reassembly_kernel(
    void* __restrict__ out,
    const void* const* const __restrict__ in,
    const uint16_t* __restrict__ line_numbers,
    const uint16_t* __restrict__ line_offsets,
    const uint16_t* __restrict__ line_lengths,
    uint32_t num_pkts,
    uint32_t line_stride) {

  const int pkt_idx = blockIdx.x;

  if (pkt_idx >= num_pkts) return;

  // Get packet info
  const uint16_t line_num = line_numbers[pkt_idx];
  const uint16_t line_offset = line_offsets[pkt_idx];
  const uint16_t payload_len = line_lengths[pkt_idx];
  const void* in_pkt = in[pkt_idx];

  // Calculate output position: line_number * line_stride + line_offset
  uint8_t* out_ptr = static_cast<uint8_t*>(out) + (line_num * line_stride) + line_offset;
  const uint8_t* in_ptr = static_cast<const uint8_t*>(in_pkt);

  // Copy payload data with coalesced access (4-byte chunks)
  const int num_words = payload_len / 4;
  const int remainder = payload_len % 4;

  // Process 4-byte aligned data
  for (int pos = threadIdx.x; pos < num_words; pos += blockDim.x) {
    const uint32_t* in_word = reinterpret_cast<const uint32_t*>(in_ptr) + pos;
    uint32_t* out_word = reinterpret_cast<uint32_t*>(out_ptr) + pos;
    *out_word = *in_word;
  }

  // Handle remaining bytes (if payload not 4-byte aligned)
  if (remainder > 0 && threadIdx.x == 0) {
    const int offset = num_words * 4;
    for (int i = 0; i < remainder; i++) {
      out_ptr[offset + i] = in_ptr[offset + i];
    }
  }
}

/**
 * @brief Simple packet reorder kernel (fallback mode)
 *
 * Assumes packets are already in raster scan order. Simply concatenates
 * packet payloads sequentially.
 */
__global__ void st2110_simple_reorder_kernel(
    void* __restrict__ out,
    const void* const* const __restrict__ in,
    uint16_t pkt_len,
    uint32_t num_pkts) {

  // Warmup check
  if (out == nullptr) return;

  const int pkt_idx = blockIdx.x;

  if (pkt_idx >= num_pkts) return;

  const void* in_pkt = in[pkt_idx];

  // Copy packet data to output at appropriate offset (4-byte aligned chunks)
  const int num_words = pkt_len / 4;
  for (int pos = threadIdx.x; pos < num_words; pos += blockDim.x) {
    const uint32_t* in_ptr = static_cast<const uint32_t*>(in_pkt) + pos;
    uint32_t* out_ptr = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(out) + pkt_idx * pkt_len) + pos;
    *out_ptr = *in_ptr;
  }

  // Handle remaining bytes (if payload not 4-byte aligned)
  const int remainder = pkt_len % 4;
  if (remainder > 0 && threadIdx.x == 0) {
    const int offset = num_words * 4;
    const uint8_t* in_bytes = static_cast<const uint8_t*>(in_pkt);
    uint8_t* out_bytes = static_cast<uint8_t*>(out) + pkt_idx * pkt_len;
    for (int i = 0; i < remainder; i++) {
      out_bytes[offset + i] = in_bytes[offset + i];
    }
  }
}

// Host-callable wrapper functions

void st2110_frame_reassembly(void* out,
                              const void* const* const in,
                              const uint16_t* line_numbers,
                              const uint16_t* line_offsets,
                              const uint16_t* line_lengths,
                              uint32_t num_pkts,
                              uint32_t line_stride,
                              cudaStream_t stream) {
  if (num_pkts == 0) return;

  // Launch with one block per packet, 128 threads per block
  const int threads_per_block = 128;
  st2110_frame_reassembly_kernel<<<num_pkts, threads_per_block, 0, stream>>>(
      out, in, line_numbers, line_offsets, line_lengths, num_pkts, line_stride);
}

void st2110_simple_reorder(void* out,
                            const void* const* const in,
                            uint16_t pkt_len,
                            uint32_t num_pkts,
                            cudaStream_t stream) {
  if (num_pkts == 0) return;

  // Launch with one block per packet, 128 threads per block
  const int threads_per_block = 128;
  st2110_simple_reorder_kernel<<<num_pkts, threads_per_block, 0, stream>>>(
      out, in, pkt_len, num_pkts);
}

/**
 * @brief CUDA kernel for YCbCr-4:2:2 10-bit to RGBA 8-bit conversion
 *
 * Input format (SMPTE ST 2110-20):
 *   - 5 bytes for 2 pixels (pgroup)
 *   - Layout: Cb0 Y0 Cr0 Y1 Cb0[1:0]Y0[1:0]Cr0[1:0]Y1[1:0] (last byte has LSBs)
 *   - 10-bit values packed as: high 8 bits in first 4 bytes, low 2 bits in 5th byte
 *
 * Output format: RGBA 8-bit (4 bytes per pixel)
 *
 * ITU-R BT.709 conversion:
 *   R = 1.164 * (Y - 64) + 1.793 * (Cr - 512)
 *   G = 1.164 * (Y - 64) - 0.213 * (Cb - 512) - 0.533 * (Cr - 512)
 *   B = 1.164 * (Y - 64) + 2.112 * (Cb - 512)
 *
 * Each thread processes 2 pixels (1 pgroup).
 */
__global__ void convert_ycbcr422_10bit_to_rgba_kernel(
    uint8_t* __restrict__ out,
    const uint8_t* __restrict__ in,
    uint32_t width,
    uint32_t height) {

  // Calculate pixel pair index (each thread processes 2 pixels)
  const uint32_t pixel_pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total_pixel_pairs = (width * height) / 2;

  if (pixel_pair_idx >= total_pixel_pairs) return;

  // Input: 5 bytes per 2 pixels (pgroup)
  const uint32_t in_offset = pixel_pair_idx * 5;
  const uint8_t* in_pgroup = in + in_offset;

  // Parse 10-bit values using BIG-ENDIAN packing (Blackmagic format)
  // Pgroup: 5 bytes = 40 bits → 4 x 10-bit values
  // Read as big-endian bit stream: Cb(10 bits) Y0(10 bits) Cr(10 bits) Y1(10 bits)
  uint64_t pgroup_bits = (uint64_t(in_pgroup[0]) << 32) |
                         (uint64_t(in_pgroup[1]) << 24) |
                         (uint64_t(in_pgroup[2]) << 16) |
                         (uint64_t(in_pgroup[3]) << 8) |
                         (uint64_t(in_pgroup[4]));

  uint16_t Cb = (pgroup_bits >> 30) & 0x3FF;  // Bits 30-39 (top 10 bits)
  uint16_t Y0 = (pgroup_bits >> 20) & 0x3FF;  // Bits 20-29
  uint16_t Cr = (pgroup_bits >> 10) & 0x3FF;  // Bits 10-19
  uint16_t Y1 = (pgroup_bits >> 0) & 0x3FF;   // Bits 0-9 (bottom 10 bits)

  // BT.709 conversion coefficients for 10-bit to 8-bit
  // Input range: Y=[64-940], CbCr=[64-960] (10-bit legal range)
  // Output range: RGB=[0-255] (8-bit full range)

  // Convert both pixels
  for (int i = 0; i < 2; i++) {
    uint16_t Y = (i == 0) ? Y0 : Y1;

    // Scale 10-bit to normalized float [0.0, 1.0]
    // Legal range: Y=[64-940], CbCr=[64-960]
    float Y_norm = (float(Y) - 64.0f) / (940.0f - 64.0f);
    float Cb_norm = (float(Cb) - 64.0f) / (960.0f - 64.0f) - 0.5f;  // Center at 0
    float Cr_norm = (float(Cr) - 64.0f) / (960.0f - 64.0f) - 0.5f;  // Center at 0

    // BT.709 YCbCr to RGB conversion (limited range to full range)
    float R = Y_norm + 1.5748f * Cr_norm;
    float G = Y_norm - 0.1873f * Cb_norm - 0.4681f * Cr_norm;
    float B = Y_norm + 1.8556f * Cb_norm;

    // Scale to 8-bit range and clamp
    int R8 = min(max(int(R * 255.0f + 0.5f), 0), 255);
    int G8 = min(max(int(G * 255.0f + 0.5f), 0), 255);
    int B8 = min(max(int(B * 255.0f + 0.5f), 0), 255);

    // Write RGBA output (R-G-B-A byte order)
    const uint32_t out_pixel_idx = pixel_pair_idx * 2 + i;
    const uint32_t out_offset = out_pixel_idx * 4;
    out[out_offset + 0] = uint8_t(R8);
    out[out_offset + 1] = uint8_t(G8);
    out[out_offset + 2] = uint8_t(B8);
    out[out_offset + 3] = 255;  // Alpha = fully opaque
  }
}

/**
 * @brief CUDA kernel for YCbCr-4:2:2 10-bit to NV12 8-bit conversion
 *
 * Input: YCbCr-4:2:2 10-bit (2.5 bytes/pixel)
 * Output: NV12 (YUV 4:2:0, 1.5 bytes/pixel)
 *   - Y plane: full resolution
 *   - UV plane: half resolution, interleaved
 *
 * NV12 layout:
 *   Y plane: width * height bytes
 *   UV plane: width * height/2 bytes (interleaved U,V pairs)
 *
 * Each thread processes 2x2 pixel block (1 chroma sample).
 */
__global__ void convert_ycbcr422_10bit_to_nv12_kernel(
    uint8_t* __restrict__ out,
    const uint8_t* __restrict__ in,
    uint32_t width,
    uint32_t height) {

  // Calculate 2x2 block index
  const uint32_t block_x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const uint32_t block_y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

  if (block_x >= width || block_y >= height) return;

  // Y plane starts at offset 0
  uint8_t* y_plane = out;
  // UV plane starts after Y plane
  uint8_t* uv_plane = out + (width * height);

  // Process 2x2 block
  // For 4:2:0, we sample chroma at every 2x2 block
  // We'll average the chroma values from the two horizontal pixels

  // Top row: pixel 0 and pixel 1
  uint32_t pixel_pair_idx_top = (block_y * width + block_x) / 2;
  uint32_t in_offset_top = pixel_pair_idx_top * 5;
  const uint8_t* in_pgroup_top = in + in_offset_top;

  // Parse 10-bit values for top row using BIG-ENDIAN packing (Blackmagic format)
  uint64_t pgroup_bits_top = (uint64_t(in_pgroup_top[0]) << 32) |
                              (uint64_t(in_pgroup_top[1]) << 24) |
                              (uint64_t(in_pgroup_top[2]) << 16) |
                              (uint64_t(in_pgroup_top[3]) << 8) |
                              (uint64_t(in_pgroup_top[4]));
  uint16_t Cb_top = (pgroup_bits_top >> 30) & 0x3FF;
  uint16_t Y0_top = (pgroup_bits_top >> 20) & 0x3FF;
  uint16_t Cr_top = (pgroup_bits_top >> 10) & 0x3FF;
  uint16_t Y1_top = (pgroup_bits_top >> 0) & 0x3FF;

  // Write Y values for top row (convert 10-bit to 8-bit by right-shift)
  y_plane[block_y * width + block_x + 0] = uint8_t(Y0_top >> 2);
  y_plane[block_y * width + block_x + 1] = uint8_t(Y1_top >> 2);

  // Bottom row (if exists)
  uint16_t Cb_bottom = Cb_top;
  uint16_t Cr_bottom = Cr_top;

  if (block_y + 1 < height) {
    uint32_t pixel_pair_idx_bottom = ((block_y + 1) * width + block_x) / 2;
    uint32_t in_offset_bottom = pixel_pair_idx_bottom * 5;
    const uint8_t* in_pgroup_bottom = in + in_offset_bottom;

    // Parse 10-bit values for bottom row using BIG-ENDIAN packing (Blackmagic format)
    uint64_t pgroup_bits_bottom = (uint64_t(in_pgroup_bottom[0]) << 32) |
                                   (uint64_t(in_pgroup_bottom[1]) << 24) |
                                   (uint64_t(in_pgroup_bottom[2]) << 16) |
                                   (uint64_t(in_pgroup_bottom[3]) << 8) |
                                   (uint64_t(in_pgroup_bottom[4]));
    Cb_bottom = (pgroup_bits_bottom >> 30) & 0x3FF;
    uint16_t Y0_bottom = (pgroup_bits_bottom >> 20) & 0x3FF;
    Cr_bottom = (pgroup_bits_bottom >> 10) & 0x3FF;
    uint16_t Y1_bottom = (pgroup_bits_bottom >> 0) & 0x3FF;

    // Write Y values for bottom row
    y_plane[(block_y + 1) * width + block_x + 0] = uint8_t(Y0_bottom >> 2);
    y_plane[(block_y + 1) * width + block_x + 1] = uint8_t(Y1_bottom >> 2);
  }

  // Average chroma from top and bottom rows, convert 10-bit to 8-bit
  uint8_t U = uint8_t(((Cb_top + Cb_bottom) / 2) >> 2);
  uint8_t V = uint8_t(((Cr_top + Cr_bottom) / 2) >> 2);

  // Write UV values (interleaved)
  uint32_t uv_row = block_y / 2;
  uint32_t uv_col = block_x;
  uint32_t uv_offset = uv_row * width + uv_col;
  uv_plane[uv_offset + 0] = U;
  uv_plane[uv_offset + 1] = V;
}

// Host-callable wrapper functions

void convert_ycbcr422_10bit_to_rgba(void* out,
                                     const void* in,
                                     uint32_t width,
                                     uint32_t height,
                                     cudaStream_t stream) {
  // Total pixel pairs (width * height / 2)
  const uint32_t total_pixel_pairs = (width * height) / 2;

  // Launch with 256 threads per block
  const int threads_per_block = 256;
  const int num_blocks = (total_pixel_pairs + threads_per_block - 1) / threads_per_block;

  convert_ycbcr422_10bit_to_rgba_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      static_cast<uint8_t*>(out),
      static_cast<const uint8_t*>(in),
      width,
      height);
}

void convert_ycbcr422_10bit_to_nv12(void* out,
                                     const void* in,
                                     uint32_t width,
                                     uint32_t height,
                                     cudaStream_t stream) {
  // Process 2x2 blocks
  // Launch 2D grid: each thread processes one 2x2 block
  dim3 threads_per_block(16, 16);  // 256 threads per block
  dim3 num_blocks((width / 2 + threads_per_block.x - 1) / threads_per_block.x,
                   (height / 2 + threads_per_block.y - 1) / threads_per_block.y);

  convert_ycbcr422_10bit_to_nv12_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      static_cast<uint8_t*>(out),
      static_cast<const uint8_t*>(in),
      width,
      height);
}
