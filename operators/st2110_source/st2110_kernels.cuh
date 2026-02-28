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

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Reassemble ST 2110-20 packets into a video frame with line-based addressing
 *
 * ST 2110-20 packets contain:
 * - RTP header (12 bytes)
 * - ST 2110 extended header with line number and offset information
 * - Pixel data for that line segment
 *
 * This kernel copies pixel data from packet payloads to the correct position
 * in the output frame based on line number and line offset.
 *
 * @param out Output frame buffer
 * @param in Array of pointers to packet payloads (after headers)
 * @param line_numbers Array of line numbers from ST 2110 headers
 * @param line_offsets Array of line offsets from ST 2110 headers
 * @param line_lengths Array of payload lengths for each packet
 * @param num_pkts Number of packets to process
 * @param line_stride Stride (bytes) between lines in output frame
 * @param stream CUDA stream for async execution
 */
__attribute__((__visibility__("default")))
void st2110_frame_reassembly(void* out,
                              const void* const* const in,
                              const uint16_t* line_numbers,
                              const uint16_t* line_offsets,
                              const uint16_t* line_lengths,
                              uint32_t num_pkts,
                              uint32_t line_stride,
                              cudaStream_t stream);

/**
 * @brief Simple packet reorder for ST 2110 (fallback when line info not available)
 *
 * This is used when ST 2110 extended headers are not parsed, and packets
 * are assumed to be in scan order already.
 *
 * @param out Output buffer
 * @param in Array of pointers to input packet payloads
 * @param pkt_len Length of each packet payload
 * @param num_pkts Number of packets
 * @param stream CUDA stream for async execution
 */
__attribute__((__visibility__("default")))
void st2110_simple_reorder(void* out,
                            const void* const* const in,
                            uint16_t pkt_len,
                            uint32_t num_pkts,
                            cudaStream_t stream);

/**
 * @brief Convert YCbCr-4:2:2 10-bit to RGBA 8-bit
 *
 * Converts from SMPTE ST 2110-20 YCbCr-4:2:2 10-bit format to RGBA 8-bit
 * for visualization. Uses ITU-R BT.709 color space conversion.
 *
 * Input format: Pixel groups of 5 bytes for 2 pixels
 *   - Byte layout: Cb0[9:2] Y0[9:2] Cr0[9:2] Y1[9:2] Cb0[1:0]Y0[1:0]Cr0[1:0]Y1[1:0]
 *   - Chroma (Cb, Cr) is shared between adjacent pixels
 *
 * Output format: RGBA 8-bit (4 bytes per pixel)
 *
 * Color conversion (BT.709):
 *   R = 1.164 * (Y - 64) + 1.793 * (Cr - 512)
 *   G = 1.164 * (Y - 64) - 0.213 * (Cb - 512) - 0.533 * (Cr - 512)
 *   B = 1.164 * (Y - 64) + 2.112 * (Cb - 512)
 *
 * @param out Output RGBA buffer (width * height * 4 bytes)
 * @param in Input YCbCr-4:2:2 10-bit buffer (width * height * 2.5 bytes)
 * @param width Frame width in pixels
 * @param height Frame height in pixels
 * @param stream CUDA stream for async execution
 */
__attribute__((__visibility__("default")))
void convert_ycbcr422_10bit_to_rgba(void* out,
                                     const void* in,
                                     uint32_t width,
                                     uint32_t height,
                                     cudaStream_t stream);

/**
 * @brief Convert YCbCr-4:2:2 10-bit to NV12 8-bit
 *
 * Converts from SMPTE ST 2110-20 YCbCr-4:2:2 10-bit format to NV12 8-bit
 * (YUV 4:2:0) for video encoding.
 *
 * Input format: YCbCr-4:2:2 10-bit (2.5 bytes/pixel)
 * Output format: NV12 (1.5 bytes/pixel)
 *   - Y plane: full resolution (width * height bytes)
 *   - UV plane: half resolution (width * height/2 bytes, interleaved U/V)
 *
 * @param out Output NV12 buffer (width * height * 1.5 bytes)
 * @param in Input YCbCr-4:2:2 10-bit buffer (width * height * 2.5 bytes)
 * @param width Frame width in pixels
 * @param height Frame height in pixels
 * @param stream CUDA stream for async execution
 */
__attribute__((__visibility__("default")))
void convert_ycbcr422_10bit_to_nv12(void* out,
                                     const void* in,
                                     uint32_t width,
                                     uint32_t height,
                                     cudaStream_t stream);

#ifdef __cplusplus
}
#endif
