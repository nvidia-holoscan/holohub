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

#ifndef RMAX_APPS_LIB_SERVICES_IPO_CHUNK_CONSUMER_BASE_H_
#define RMAX_APPS_LIB_SERVICES_IPO_CHUNK_CONSUMER_BASE_H_

#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>
#include <chrono>

#include <rivermax_api.h>

#include "api/rmax_apps_lib_api.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

namespace ral {
namespace io_node {

/**
 * @brief: IIPOChunkConsumer class.
 *
 * The interface to the IPO Chunk Consumer Class.
 */
class IIPOChunkConsumer {
 public:
  virtual ~IIPOChunkConsumer() = default;
  /**
   * @brief Consumes and processes packets from a given chunk.
   *
   * This function processes the packets contained in the provided chunk and returns a tuple
   * containing the return status, the number of consumed packets, and the number of unconsumed
   * packets.
   *
   * @param chunk Reference to the IPOReceiveChunk containing the packets.
   * @param stream Reference to the IPOReceiveStream associated with the chunk.
   * @return std::tuple<ReturnStatus, size_t, size_t> containing the return status, the number of
   * consumed packets, and the number of unconsumed packets.
   */
  virtual std::tuple<ReturnStatus, size_t, size_t> consume_chunk_packets(
      IPOReceiveChunk& chunk, IPOReceiveStream& stream) = 0;
};

}  // namespace io_node
}  // namespace ral

#endif /* RMAX_APPS_LIB_SERVICES_IPO_CHUNK_CONSUMER_BASE_H_ */
