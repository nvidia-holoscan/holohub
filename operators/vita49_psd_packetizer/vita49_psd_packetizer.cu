/*
 * SPDX-FileCopyrightText: 2025 Valley Tech Systems, Inc.
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
    spec.param(print_every_n_packets,
        "print_every_n_packets",
        "Print the time it takes to send N packets",
        "Print the time it takes to send N packets (0: no print, defaults to 10 * num_channels)");
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
            new_packet_sender(
                dest_host.get().c_str(),
                base_dest_port + i,
                moui,
                dcode)));
        rust::Vec<uint8_t> out;
        out.reserve(burst_size.get());
        // Need to add elements here otherwised reserved data
        // is deallocated when being added to wrapper vector.
        for (int i = 0; i < burst_size.get(); i++) {
            out.push_back(0);
        }
        output_data.push_back(out);
    }

    if (!print_every_n_packets.has_value()) {
        print_every_n_packets = num_channels.get() * 10;
    }

    start = std::chrono::steady_clock::now();
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

    uint32_t stream_id = 0;
    uint32_t integer_timestamp = 0;
    uint64_t fractional_timestamp = 0;
    if (meta->has_key("stream_id"))
        stream_id = meta->get<uint32_t>("stream_id");
    if (meta->has_key("integer_timestamp"))
        integer_timestamp = meta->get<uint32_t>("integer_timestamp");
    if (meta->has_key("fractional_timestamp"))
        fractional_timestamp = meta->get<uint64_t>("fractional_timestamp");

    if (packet_sender->is_time_for_context() || meta->get<bool>("change_indicator", false)) {
        SpectralContextPacket packet;
        packet.stream_id = stream_id;
        packet.integer_timestamp = integer_timestamp;
        packet.fractional_timestamp = fractional_timestamp;
        if (meta->has_key("spectrum_type"))
            packet.spectrum_type = meta->get<uint8_t>("spectrum_type");
        if (meta->has_key("averaging_type"))
            packet.averaging_type = meta->get<uint8_t>("averaging_type");
        if (meta->has_key("window_time_delta_interpretation")) {
            packet.window_time_delta_interpretation
                = meta->get<uint8_t>("window_time_delta_interpretation");
        }
        if (meta->has_key("window_type"))
            packet.window_type = meta->get<uint8_t>("window_type");
        if (meta->has_key("num_transform_points"))
            packet.num_transform_points = meta->get<uint32_t>("num_transform_points");
        if (meta->has_key("num_window_points"))
            packet.num_window_points = meta->get<uint32_t>("num_window_points");
        if (meta->has_key("resolution"))
            packet.resolution_hz = meta->get<uint64_t>("resolution");
        if (meta->has_key("span"))
            packet.span_hz = meta->get<uint64_t>("span");
        if (meta->has_key("num_averages"))
            packet.num_averages = meta->get<uint32_t>("num_averages");
        if (meta->has_key("weighting_factor"))
            packet.weighting_factor = meta->get<float>("weighting_factor");
        if (meta->has_key("f1_index"))
            packet.f1_index = meta->get<int32_t>("f1_index");
        if (meta->has_key("f2_index"))
            packet.f2_index = meta->get<int32_t>("f2_index");
        if (meta->has_key("window_time_delta"))
            packet.window_time_delta = meta->get<uint32_t>("window_time_delta");
        if (meta->has_key("rf_ref_freq_hz"))
            packet.rf_ref_freq_hz = meta->get<double>("rf_ref_freq_hz");
        if (meta->has_key("sample_rate_hz"))
            packet.sample_rate_sps = meta->get<double>("sample_rate_hz");
        if (meta->has_key("bandwidth_hz"))
            packet.bandwidth_hz = meta->get<double>("bandwidth_hz");

        packet.change_indicator = meta->get<bool>("change_indicator", false);

        HOLOSCAN_LOG_DEBUG("Sending context packet (channel {}) to {}/udp",
                          channel_num, packet_sender->destination.c_str());
        packet_sender->send_context_packet(packet);
    }

    rust::Vec<uint8_t> out = output_data.at(channel_num);
    cudaMemcpy(
        out.data(),
        psd_data.Data(),
        burst_size.get() * sizeof(int8_t),
        cudaMemcpyDeviceToHost);

    SpectralDataPacket packet;
    packet.stream_id = stream_id;
    packet.integer_timestamp = integer_timestamp;
    packet.fractional_timestamp = fractional_timestamp;
    packet.spectral_data = out;

    HOLOSCAN_LOG_DEBUG("Sending {} bytes of spectral data (channel {}) to {}/udp",
        packet.spectral_data.size(),
        channel_num,
        packet_sender->destination.c_str());

    packet_sender->send_data_packet(packet);

    if (print_every_n_packets.get() != 0 && ++packet_send_counter >= print_every_n_packets.get()) {
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_diff = end - start;
        HOLOSCAN_LOG_INFO("Sent {} packets in {} seconds", packet_send_counter, time_diff.count());
        packet_send_counter = 0;
        start = std::chrono::steady_clock::now();
    }
}
}  // namespace holoscan::ops
