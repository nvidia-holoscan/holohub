/*
 * SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "packet_sender.hpp"
#include "signal_data.hpp"
#include "context.hpp"

PacketSender::PacketSender(const char *_dest_host,
                           unsigned short _dest_port,
                           uint32_t _manufacturer_oui,
                           uint32_t _device_code) {
    dest_host_resolved = lookup_host(_dest_host);
    dest_host = std::string(_dest_host);
    dest_port = _dest_port;
    sock = socket(AF_INET, SOCK_DGRAM, 0);
    dest.sin_family = AF_INET;
    dest.sin_port = htons(_dest_port);
    dest.sin_addr.s_addr = inet_addr(dest_host_resolved.c_str());
    manufacturer_oui = _manufacturer_oui;
    device_code = _device_code;
}

bool PacketSender::is_time_for_context() {
    // Send a context packet every 10 signal data packets
    return packet_count % 10 == 0;
}

std::string PacketSender::lookup_host(const char *host) {
    struct addrinfo hints;
    struct addrinfo *res;
    struct addrinfo *result;
    int ret;
    constexpr size_t kBufSize = 128;
    char addr[kBufSize];
    void *p;

    memset(&hints, 0, sizeof (hints));
    hints.ai_family = PF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags |= AI_CANONNAME;

    ret = getaddrinfo(host, NULL, &hints, &result);
    if (ret != 0) {
        throw std::invalid_argument("hostname could not be parsed");
    }

    res = result;

    inet_ntop(res->ai_family, res->ai_addr->sa_data, addr, kBufSize);

    if (res->ai_family == AF_INET) {
        p = &((struct sockaddr_in *) res->ai_addr)->sin_addr;
    } else {
        p = &((struct sockaddr_in6 *) res->ai_addr)->sin6_addr;
    }
    inet_ntop(res->ai_family, p, addr, kBufSize);
    freeaddrinfo(result);
    return std::string(addr);
}

int PacketSender::send_signal_data(const uint8_t *psd_data,
                                   const size_t psd_data_size,
                                   std::shared_ptr<holoscan::MetadataDictionary> meta) {
    vrt::packets::SignalData data_packet;

    std::span<const uint8_t> data_span(psd_data, psd_data_size);

    data_packet.payload(data_span);
    if (meta->has_key("stream_id"))
        data_packet.stream_id(meta->get<uint32_t>("stream_id"));
    if (meta->has_key("integer_timestamp"))
        data_packet.integer_timestamp(meta->get<uint32_t>("integer_timestamp"));
    if (meta->has_key("fractional_timestamp"))
        data_packet.fractional_timestamp(meta->get<uint64_t>("fractional_timestamp"));
    data_packet.packet_count(static_cast<uint8_t>(packet_count++ % 16));

    auto packed_data = data_packet.data();

    return sendto(
        sock,
        packed_data.data(),
        packed_data.size(),
        0,
        reinterpret_cast<sockaddr*>(&dest),
        sizeof(dest));
}

int PacketSender::send_context_packet(std::shared_ptr<holoscan::MetadataDictionary> meta) {
    vrt::packets::Context context_packet;
    vrtgen::packing::Spectrum context_spectrum;
    if (meta->has_key("spectrum_type")) {
        context_spectrum.spectrum_type(
            vrtgen::packing::SpectrumType{meta->get<uint8_t>("spectrum_type")});
    }
    if (meta->has_key("averaging_type")) {
        context_spectrum.averaging_type(
            vrtgen::packing::AveragingType{meta->get<uint8_t>("averaging_type")});
    }
    if (meta->has_key("window_time_delta_interpretation")) {
        context_spectrum.window_time(
            vrtgen::packing::WindowTimeDelta{
                meta->get<uint8_t>("window_time_delta_interpretation")});
    }
    if (meta->has_key("window_type")) {
        context_spectrum.window_type(
            vrtgen::packing::WindowType{meta->get<uint8_t>("window_type")});
    }
    if (meta->has_key("num_transform_points"))
        context_spectrum.num_transform_points(meta->get<uint32_t>("num_transform_points"));
    if (meta->has_key("num_window_points"))
        context_spectrum.num_window_points(meta->get<uint32_t>("num_window_points"));
    if (meta->has_key("resolution"))
        context_spectrum.resolution(meta->get<uint64_t>("resolution"));
    if (meta->has_key("span"))
        context_spectrum.span(meta->get<uint64_t>("span"));
    if (meta->has_key("num_averages"))
        context_spectrum.num_averages(meta->get<uint32_t>("num_averages"));
    if (meta->has_key("weighting_factor"))
        context_spectrum.weighting_factor(meta->get<float>("weighting_factor"));
    if (meta->has_key("f1_index"))
        context_spectrum.f1_index(meta->get<int32_t>("f1_index"));
    if (meta->has_key("f2_index"))
        context_spectrum.f2_index(meta->get<int32_t>("f2_index"));
    if (meta->has_key("window_time_delta"))
        context_spectrum.window_time_delta(meta->get<uint32_t>("window_time_delta"));
    context_packet.spectrum(context_spectrum);

    vrtgen::packing::DeviceIdentifier context_device;
    context_device.manufacturer_oui(manufacturer_oui);
    context_device.device_code(device_code);
    context_packet.device_id(context_device);

    if (meta->has_key("stream_id"))
        context_packet.stream_id(meta->get<uint32_t>("stream_id"));
    if (meta->has_key("integer_timestamp"))
        context_packet.integer_timestamp(meta->get<uint32_t>("integer_timestamp"));
    if (meta->has_key("fractional_timestamp"))
        context_packet.fractional_timestamp(meta->get<uint64_t>("fractional_timestamp"));
    if (meta->has_key("change_indicator"))
        context_packet.change_indicator(meta->get<bool>("change_indicator"));
    if (meta->has_key("rf_ref_freq_hz"))
        context_packet.rf_ref_frequency(meta->get<double>("rf_ref_freq_hz"));
    if (meta->has_key("sample_rate_hz"))
        context_packet.sample_rate(meta->get<double>("sample_rate_hz"));
    if (meta->has_key("bandwidth_hz"))
        context_packet.bandwidth(meta->get<double>("bandwidth_hz"));

    context_packet.v49_spec_compliance(vrtgen::packing::V49StandardCompliance::V49_2);
    context_packet.packet_count(static_cast<uint8_t>(context_packet_count++ % 16));

    auto packed_data = context_packet.data();

    return sendto(
        sock,
        packed_data.data(),
        packed_data.size(),
        0,
        reinterpret_cast<sockaddr*>(&dest),
        sizeof(dest));
}
