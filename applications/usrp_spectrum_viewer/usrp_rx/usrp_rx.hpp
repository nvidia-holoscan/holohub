// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <uhd/stream.hpp>
#include <uhd/types/time_spec.hpp>
#include <uhd/usrp/multi_usrp.hpp>

namespace holoscan::ops {

// Control operator that configures a USRP and asks UHD to start streaming UDP
// packets toward the NIC consumed by the rest of the application.
class UsrpRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(UsrpRxOp)

  UsrpRxOp() = default;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;
  void stop() override;

 private:
  // Radio and network streaming parameters loaded from config.
  Parameter<bool> enabled_;
  Parameter<std::string> args_;
  Parameter<double> rate_;
  Parameter<double> freq_;
  Parameter<double> gain_;
  Parameter<std::vector<int64_t>> channels_;
  Parameter<std::string> dest_addr_;
  Parameter<std::vector<int64_t>> dest_ports_;
  Parameter<std::string> adapter_;
  Parameter<std::string> dest_mac_addr_;
  Parameter<bool> keep_hdr_;
  Parameter<int64_t> spp_;
  Parameter<uint32_t> mtu_;
  Parameter<double> start_delay_seconds_;

  // UHD device handle plus one RX streamer per configured channel.
  std::shared_ptr<uhd::usrp::multi_usrp> usrp_;
  std::vector<uhd::rx_streamer::sptr> rx_streamers_;
  bool started_ = false;
};

}  // namespace holoscan::ops
