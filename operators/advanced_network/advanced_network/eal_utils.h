/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>

#include "holoscan/holoscan.hpp"

namespace holoscan::advanced_network {
namespace detail {

inline uint64_t fnv1a_mix_bytes(uint64_t hash, const void* data, std::size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

inline uint64_t fallback_eal_file_prefix_token() {
  try {
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
  } catch (...) {
    // Fall back to a host/time hash when random_device is unavailable.
  }

  char hostname[256] = {};
  (void)::gethostname(hostname, sizeof(hostname) - 1);

  struct timespec ts = {};
  (void)::clock_gettime(CLOCK_BOOTTIME, &ts);

  uint64_t hash = 1469598103934665603ULL;
  hash = fnv1a_mix_bytes(hash, hostname, sizeof(hostname));
  hash = fnv1a_mix_bytes(hash, &ts.tv_sec, sizeof(ts.tv_sec));
  hash = fnv1a_mix_bytes(hash, &ts.tv_nsec, sizeof(ts.tv_nsec));
  return hash;
}

}  // namespace detail

inline std::string make_eal_file_prefix(const char* tag) {
  static std::atomic<uint32_t> counter{0};
  uint64_t namespace_or_fallback = 0;
  struct stat st;
  if (::stat("/proc/self/ns/pid", &st) == 0) {
    namespace_or_fallback = static_cast<uint64_t>(st.st_ino);
  } else {
    namespace_or_fallback = detail::fallback_eal_file_prefix_token();
    HOLOSCAN_LOG_WARN(
        "Could not stat /proc/self/ns/pid for EAL file-prefix uniqueness; using fallback token. "
        "Hardened containers running as PID 1 may collide if fallback token generation is "
        "unavailable.");
  }

  return std::string(tag) + "_" + std::to_string(static_cast<uint32_t>(::getpid())) + "_" +
         std::to_string(namespace_or_fallback) + "_" +
         std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

}  // namespace holoscan::advanced_network
