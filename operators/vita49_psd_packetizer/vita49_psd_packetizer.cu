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
    spec.param(dest_port,
               "dest_port",
               "Destination port",
               "Destination port for VRT UDP packets");
    spec.param(manufacturer_oui,
               "manufacturer_oui",
               "Manufacturer OUI",
               "VITA 49 organizational unique identifier (32 bits)");
    spec.param(device_code,
               "device_code",
               "Device code",
               "VITA 49 organizational unique identifier (32 bits)");
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

    packet_sender = PacketSender(
        dest_host.get().c_str(),
        dest_port,
        moui,
        dcode);
}

void V49PsdPacketizer::start() {
    output_data = (int8_t *)malloc(burst_size.get() * sizeof(int8_t));
}

void V49PsdPacketizer::stop() {
    free(output_data);
}

bool V49PsdPacketizer::is_time_for_context() {
    // Send a context packet every 10 signal data packets
    return packet_sender.get_packet_count() % 10 == 0;
}

void V49PsdPacketizer::compute(InputContext& op_input, OutputContext& _out, ExecutionContext&) {
    auto psd_data = op_input.receive<tensor_t<int8_t, 1>>("in").value();
    auto meta = metadata();
    if (is_time_for_context() || meta->get<bool>("context_field_change", false)) {
        HOLOSCAN_LOG_INFO("Sending context packet");
        packet_sender.send_context_packet(meta);
    }

    cudaMemcpy(
        output_data,
        psd_data.Data(),
        burst_size.get() * sizeof(int8_t),
        cudaMemcpyDeviceToHost);

    HOLOSCAN_LOG_INFO("Sending {} sample spectral data packet to {}:{}/udp",
        burst_size.get() * sizeof(int8_t),
        dest_host,
        dest_port);
    packet_sender.send_signal_data(
        reinterpret_cast<uint8_t *>(output_data),
        burst_size.get() * sizeof(int8_t),
        meta);
}
}  // namespace holoscan::ops
