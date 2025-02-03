/*
 * SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "vita49_psd_packetizer.hpp"

namespace holoscan::ops {
void V49PsdPacketizer::setup(OperatorSpec& spec) {
    spec.input<tensor_t<int8_t, 1>>("in");
    spec.param(burst_size,
        "burst_size",
        "Burst size",
        "Number of samples to process in each burst");
    spec.param(dest_host,
        "dest_host",
        "Destination host",
        "Destination host for VRT UDP packets");
    spec.param(base_dest_port,
        "base_dest_port",
        "Base destination port",
        "Base destination port for VRT UDP packets (+ channel)");
    spec.param(manufacturer_oui,
        "manufacturer_oui",
        "Manufacturer OUI",
        "VITA 49 organizational unique identifier (32 bits)");
    spec.param(device_code,
        "device_code",
        "Device code",
        "VITA 49 organizational unique identifier (32 bits)");
    spec.param(num_channels,
        "num_channels",
        "Number of channels",
        "Number of channels to support");
}

void V49PsdPacketizer::initialize() {
    holoscan::Operator::initialize();
    uint32_t moui = 0;
    uint32_t dcode = 0;
    if (manufacturer_oui.has_value()) {
        moui = manufacturer_oui.get();
    }
    if (device_code.has_value()) {
        dcode = device_code.get();
    }

    for (uint16_t i = 0; i < num_channels.get(); i++) {
        packet_senders.push_back(std::make_shared<PacketSender>(
            PacketSender(
                dest_host.get().c_str(),
                base_dest_port + i,
                moui,
                dcode)));
    }
}

void V49PsdPacketizer::start() {
    output_data = (int8_t *)malloc(burst_size.get() * sizeof(int8_t));
}

void V49PsdPacketizer::stop() {
    free(output_data);
}

void V49PsdPacketizer::compute(InputContext& op_input, OutputContext& _out, ExecutionContext&) {
    auto psd_data = op_input.receive<tensor_t<int8_t, 1>>("in").value();
    auto meta = metadata();
    if (!meta->has_key("channel_number")) {
        HOLOSCAN_LOG_CRITICAL("error - input metadata does not have channel_number set!");
        throw;
    }

    auto channel_num = meta->get<uint16_t>("channel_number", 0);
    auto packet_sender = packet_senders.at(channel_num);

    if (packet_sender->is_time_for_context() || meta->get<bool>("context_field_change", false)) {
        HOLOSCAN_LOG_INFO("Sending context packet (channel {}) to {}:{}/udp",
                          channel_num, packet_sender->dest_host, packet_sender->dest_port);
        packet_sender->send_context_packet(meta);
    }

    cudaMemcpy(
        output_data,
        psd_data.Data(),
        burst_size.get() * sizeof(int8_t),
        cudaMemcpyDeviceToHost);

    HOLOSCAN_LOG_INFO("Sending {} sample spectral data (channel {}) packet to {}:{}/udp",
        burst_size.get() * sizeof(int8_t),
        channel_num,
        packet_sender->dest_host,
        packet_sender->dest_port);
    packet_sender->send_signal_data(
        reinterpret_cast<uint8_t *>(output_data),
        burst_size.get() * sizeof(int8_t),
        meta);
}
}  // namespace holoscan::ops
