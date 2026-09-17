// SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Self-contained parser for the HSB firmware metadata blob embedded in raw
// image buffers by the STM32 board on Hololink-compatible camera modules.
//
// The blob starts at the first 128-byte-aligned offset past the last pixel
// byte (per HSB protocol specification; confirmed by Patrick O'Grady).
// All fields are big-endian.  The blob is 48 bytes; the buffer is padded to
// METADATA_SIZE (128 bytes) with zeros.
//
// Binary layout:
//   Offset  Size  Field
//    0       4    flags        uint32 BE
//    4       4    psn          uint32 BE
//    8       4    crc          uint32 BE
//   12       8    timestamp_s  uint64 BE
//   20       4    timestamp_ns uint32 BE
//   24       8    bytes_written uint64 BE
//   32       2    <reserved>   uint16 BE
//   34       2    frame_number uint16 BE
//   36       8    metadata_s   uint64 BE
//   44       4    metadata_ns  uint32 BE
//   48      80    <padding>
//
// No dependency on hololink headers; the layout was derived from
// hololink/src/hololink/core/hololink.cpp::deserialize_metadata().

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace holoscan::holoscan_camera {

struct HsbFrameMetadata {
  std::uint32_t flags{};
  std::uint32_t psn{};
  std::uint32_t crc{};
  std::uint64_t timestamp_s{};
  std::uint32_t timestamp_ns{};
  std::uint64_t bytes_written{};
  std::uint16_t frame_number{};
  std::uint64_t metadata_s{};
  std::uint32_t metadata_ns{};
};

// Parse HSB metadata from a CPU-mapped raw image buffer.
//
// buf         : CPU pointer to the start of the raw NvSciBuf image plane
// buf_size    : total accessible bytes in the buffer
// pixel_bytes : plane_pitch[0] * plane_height[0] — byte count of the pixel
//               region; the HSB blob starts at the next 128-byte boundary
//
// Returns nullopt when the blob would fall outside the buffer.
inline std::optional<HsbFrameMetadata> parse_hsb_metadata(
    const std::uint8_t* buf, std::size_t buf_size, std::size_t pixel_bytes) {

  constexpr std::size_t kAlign   = 128;
  constexpr std::size_t kBlobMin = 48;

  // Round pixel_bytes up to the next 128-byte page.
  const std::size_t offset = (pixel_bytes + kAlign - 1u) & ~(kAlign - 1u);
  if (offset + kBlobMin > buf_size) {
    return std::nullopt;
  }

  const std::uint8_t* p = buf + offset;

  auto u16 = [](const std::uint8_t* b) -> std::uint16_t {
    return static_cast<std::uint16_t>((static_cast<std::uint16_t>(b[0]) << 8u) | b[1]);
  };
  auto u32 = [](const std::uint8_t* b) -> std::uint32_t {
    return (static_cast<std::uint32_t>(b[0]) << 24u) |
           (static_cast<std::uint32_t>(b[1]) << 16u) |
           (static_cast<std::uint32_t>(b[2]) <<  8u) | b[3];
  };
  auto u64 = [](const std::uint8_t* b) -> std::uint64_t {
    return (static_cast<std::uint64_t>(b[0]) << 56u) |
           (static_cast<std::uint64_t>(b[1]) << 48u) |
           (static_cast<std::uint64_t>(b[2]) << 40u) |
           (static_cast<std::uint64_t>(b[3]) << 32u) |
           (static_cast<std::uint64_t>(b[4]) << 24u) |
           (static_cast<std::uint64_t>(b[5]) << 16u) |
           (static_cast<std::uint64_t>(b[6]) <<  8u) | b[7];
  };

  HsbFrameMetadata m;
  m.flags         = u32(p +  0);
  m.psn           = u32(p +  4);
  m.crc           = u32(p +  8);
  m.timestamp_s   = u64(p + 12);
  m.timestamp_ns  = u32(p + 20);
  m.bytes_written = u64(p + 24);
  // p + 32: reserved uint16 — skip
  m.frame_number  = u16(p + 34);
  m.metadata_s    = u64(p + 36);
  m.metadata_ns   = u32(p + 44);
  return m;
}

}  // namespace holoscan::holoscan_camera
