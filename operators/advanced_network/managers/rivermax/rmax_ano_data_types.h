/*
 * Copyright © 2017-2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RMAX_ANO_DATA_TYPES_H_
#define RMAX_ANO_DATA_TYPES_H_

#include "adv_network_types.h"

namespace holoscan::ops {

// using namespace holoscan::ops;

class IAnoBurstsCollection {
 public:
  virtual ~IAnoBurstsCollection() = default;
  virtual bool enqueue_burst(std::shared_ptr<AdvNetBurstParams> burst) = 0;
  virtual std::shared_ptr<AdvNetBurstParams> dequeue_burst() = 0;
  virtual size_t available_bursts() = 0;
  virtual bool empty() = 0;
};

enum BurstFlags : uint8_t {
  FLAGS_NONE = 0,
  INFO_PER_PACKET = 1,
};

struct AnoBurstExtendedInfo {
  uint32_t tag;
  BurstFlags burst_flags;
  uint16_t burst_id;
  bool hds_on;
  uint16_t header_stride_size;
  uint16_t payload_stride_size;
  bool header_on_cpu;
  bool payload_on_cpu;
  uint16_t header_seg_idx;
  uint16_t payload_seg_idx;
};

struct RmaxPacketExtendedInfo {
  uint32_t flow_tag;
  uint64_t timestamp;
};

};  // namespace holoscan::ops

#endif  // RMAX_ANO_DATA_TYPES_H_
