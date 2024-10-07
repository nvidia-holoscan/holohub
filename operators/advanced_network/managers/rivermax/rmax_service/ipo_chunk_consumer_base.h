/*
 * Copyright © 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
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
