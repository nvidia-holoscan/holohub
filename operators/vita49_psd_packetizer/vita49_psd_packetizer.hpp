/*
 * SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <matx.h>
#include "holoscan/holoscan.hpp"
#include "packet_sender.hpp"

using namespace matx;

namespace holoscan::ops {
class V49PsdPacketizer : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(V49PsdPacketizer)

    void initialize() override;
    void setup(OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
    bool is_time_for_context();

    Parameter<int> burst_size;
    Parameter<std::string> dest_host;
    Parameter<unsigned short> dest_port;
    Parameter<uint32_t> manufacturer_oui;
    Parameter<uint32_t> device_code;

    int8_t *output_data;
    PacketSender packet_sender;
};
}  // namespace holoscan::ops
