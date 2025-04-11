// SPDX-FileCopyrightText: 2025 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
use vita49::{prelude::*, Spectrum, WindowTimeDelta};

#[cxx::bridge]
mod ffi {
    pub struct SpectralDataPacket {
        stream_id: u32,
        integer_timestamp: u32,
        fractional_timestamp: u64,
        spectral_data: Vec<u8>,
    }

    pub struct SpectralContextPacket {
        stream_id: u32,
        integer_timestamp: u32,
        fractional_timestamp: u64,
        change_indicator: bool,
        rf_ref_freq_hz: f64,
        sample_rate_sps: f64,
        bandwidth_hz: f64,
        spectrum_type: u8,
        averaging_type: u8,
        window_time_delta_interpretation: u8,
        window_type: u8,
        num_transform_points: u32,
        num_window_points: u32,
        resolution_hz: f64,
        span_hz: f64,
        num_averages: u32,
        weighting_factor: i32,
        f1_index: i32,
        f2_index: i32,
        window_time_delta: u32,
    }

    pub struct PacketSender {
        packet_count: u64,
        context_packet_count: u64,
        destination: String,
        socket: Box<UdpSocket>,
        manufacturer_oui: u32,
        device_code: u32,
    }

    extern "Rust" {
        type UdpSocket;
        fn new_packet_sender(
            dest_host: String,
            dest_port: u16,
            manufacturer_oui: u32,
            device_code: u32,
        ) -> PacketSender;
        fn send_data_packet(self: &mut PacketSender, data: &SpectralDataPacket);
        fn send_context_packet(self: &mut PacketSender, data: &SpectralContextPacket);
        fn is_time_for_context(self: &PacketSender) -> bool;
    }
}

pub struct UdpSocket(std::net::UdpSocket);

use ffi::*;

pub fn new_packet_sender(
    dest_host: String,
    dest_port: u16,
    manufacturer_oui: u32,
    device_code: u32,
) -> PacketSender {
    let destination = format!("{}:{}", &dest_host, dest_port);
    let socket = Box::new(UdpSocket(std::net::UdpSocket::bind("0.0.0.0:0").unwrap()));
    PacketSender {
        packet_count: 0,
        context_packet_count: 0,
        destination,
        socket,
        manufacturer_oui,
        device_code,
    }
}

impl crate::ffi::PacketSender {
    pub fn send_data_packet(&mut self, data: &SpectralDataPacket) {
        let mut vrt_packet = Vrt::new_signal_data_packet();
        vrt_packet.set_stream_id(Some(data.stream_id));
        vrt_packet
            .set_integer_timestamp(Some(data.integer_timestamp), Tsi::Utc)
            .unwrap();
        vrt_packet
            .set_fractional_timestamp(Some(data.fractional_timestamp), Tsf::RealTimePs)
            .unwrap();
        vrt_packet.set_signal_payload(&data.spectral_data).unwrap();
        vrt_packet
            .header_mut()
            .set_packet_count((self.packet_count % 16) as u8);
        vrt_packet.update_packet_size();
        self.socket
            .0
            .send_to(&vrt_packet.to_bytes().unwrap(), &self.destination)
            .unwrap();
        self.packet_count += 1;
    }

    pub fn send_context_packet(&mut self, data: &SpectralContextPacket) {
        let mut vrt_packet = Vrt::new_context_packet();
        vrt_packet.set_stream_id(Some(data.stream_id));
        vrt_packet
            .set_integer_timestamp(Some(data.integer_timestamp), Tsi::Utc)
            .unwrap();
        vrt_packet
            .set_fractional_timestamp(Some(data.fractional_timestamp), Tsf::RealTimePs)
            .unwrap();
        vrt_packet
            .header_mut()
            .set_packet_count((self.context_packet_count % 16) as u8);

        let context = vrt_packet.payload_mut().context_mut().unwrap();
        context.set_context_changed(data.change_indicator);
        let mut spectrum = Spectrum::default();
        spectrum
            .set_spectrum_type(data.spectrum_type.into())
            .unwrap();
        spectrum.set_window_type(data.window_type.into()).unwrap();
        spectrum.set_num_transform_points(data.num_transform_points);
        spectrum.set_num_window_points(data.num_window_points);
        spectrum.set_resolution_hz(data.resolution_hz);
        spectrum.set_span_hz(data.span_hz);
        spectrum.set_num_averages(data.num_averages);
        spectrum.set_weighting_factor(data.weighting_factor);
        spectrum.set_f1_index(data.f1_index);
        spectrum.set_f2_index(data.f2_index);
        spectrum.set_window_time_delta(WindowTimeDelta::from_samples(data.window_time_delta));
        context.set_spectrum(Some(spectrum));
        context.set_rf_ref_freq_hz(Some(data.rf_ref_freq_hz));
        context.set_sample_rate_sps(Some(data.sample_rate_sps));
        context.set_bandwidth_hz(Some(data.bandwidth_hz));

        vrt_packet.update_packet_size();
        self.socket
            .0
            .send_to(&vrt_packet.to_bytes().unwrap(), &self.destination)
            .unwrap();
        self.context_packet_count += 1;
    }

    pub fn is_time_for_context(&self) -> bool {
        self.packet_count % 10 == 0
    }
}
