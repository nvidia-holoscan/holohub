// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "usrp_rx.hpp"

#include <cmath>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace holoscan::ops {

namespace {

// Build the UHD stream-args string that tells the radio where and how to emit
// its outgoing UDP/CHDR traffic.
std::string make_stream_args(const std::string& dest_addr,
                             int64_t dest_port,
                             const std::string& adapter,
                             const std::string& dest_mac_addr,
                             bool keep_hdr,
                             uint32_t mtu) {
  std::ostringstream args;
  args << "dest_addr=" << dest_addr
       << ",dest_port=" << dest_port
       << ",stream_mode=" << (keep_hdr ? "full_packet" : "raw_payload");
  if (!adapter.empty()) {
    args << ",adapter=" << adapter;
  }
  if (!dest_mac_addr.empty()) {
    args << ",dest_mac_addr=" << dest_mac_addr;
  }
  if (mtu > 0) {
    args << ",mtu=" << mtu;
  }
  return args.str();
}

}  // namespace

void UsrpRxOp::setup(OperatorSpec& spec) {
  spec.param(enabled_, "enabled", "Enabled", "Enable in-app USRP startup control.", false);
  spec.param(args_, "args", "USRP args", "UHD device args string.", std::string(""));
  spec.param(rate_, "rate", "Sample rate", "RX sample rate in samples per second.", 1e6);
  spec.param(freq_, "freq", "Center frequency", "RX center frequency in Hz.", 1e9);
  spec.param(gain_, "gain", "Gain", "RX gain in dB.", 0.0);
  spec.param(channels_,
             "channels",
             "Channels",
             "USRP RX channels to enable.",
             std::vector<int64_t>{0, 1});
  spec.param(dest_addr_,
             "dest_addr",
             "Destination IP",
             "Remote destination IP for UDP streaming.",
             std::string(""));
  spec.param(dest_ports_,
             "dest_ports",
             "Destination ports",
             "Remote destination UDP ports, one per channel.",
             std::vector<int64_t>{1234, 1235});
  spec.param(adapter_, "adapter", "Adapter", "USRP egress adapter name.", std::string("sfp0"));
  spec.param(dest_mac_addr_,
             "dest_mac_addr",
             "Destination MAC",
             "Remote MAC address; leave empty to use ARP.",
             std::string(""));
  spec.param(keep_hdr_,
             "keep_hdr",
             "Keep CHDR header",
             "When true, preserve CHDR headers in outgoing packets.",
             true);
  spec.param(spp_,
             "spp",
             "Samples per packet",
             "Requested samples per packet for remote UDP streaming.",
             static_cast<int64_t>(1024));
  spec.param(mtu_, "mtu", "MTU", "MTU size for remote UDP streaming.", static_cast<uint32_t>(8064));
  spec.param(start_delay_seconds_,
             "start_delay_seconds",
             "Start delay",
             "Delay before multi-channel stream start so channels begin together.",
             0.05);
}

void UsrpRxOp::start() {
  started_ = false;
  rx_streamers_.clear();

  // This operator is optional so the application can either control the radio
  // directly or attach to an already-running external streamer.
  if (!enabled_.get()) {
    HOLOSCAN_LOG_INFO("UsrpRxOp disabled; skipping radio startup");
    return;
  }

  if (!keep_hdr_.get()) {
    throw std::runtime_error(
        "usrp_rx.keep_hdr must be true for this pipeline: raw_payload streaming "
        "removes the CHDR header and misaligns the IQ samples UhdChdrRxOp reads");
  }

  const auto& channels = channels_.get();
  const auto& dest_ports = dest_ports_.get();
  if (channels.empty()) {
    throw std::runtime_error("usrp_rx.channels must not be empty");
  }
  if (channels.size() != dest_ports.size()) {
    throw std::runtime_error("usrp_rx.channels and dest_ports must have the same length");
  }
  if (dest_addr_.get().empty()) {
    throw std::runtime_error("usrp_rx.dest_addr must be set when enabled=true");
  }
  const double start_delay_seconds = start_delay_seconds_.get();
  if (!std::isfinite(start_delay_seconds) || start_delay_seconds < 0.0) {
    throw std::runtime_error("usrp_rx.start_delay_seconds must be finite and >= 0");
  }

  std::unordered_set<int64_t> unique_channels;
  std::unordered_set<int64_t> unique_ports;
  for (size_t index = 0; index < channels.size(); ++index) {
    if (!unique_channels.insert(channels[index]).second) {
      throw std::runtime_error(fmt::format(
          "usrp_rx.channels contains duplicate channel {}", channels[index]));
    }
    if (dest_ports[index] <= 0 || dest_ports[index] > 65535) {
      throw std::runtime_error(fmt::format(
          "usrp_rx.dest_ports contains invalid UDP port {}", dest_ports[index]));
    }
    if (!unique_ports.insert(dest_ports[index]).second) {
      throw std::runtime_error(fmt::format(
          "usrp_rx.dest_ports contains duplicate UDP port {}", dest_ports[index]));
    }
  }

  // Discover and open the requested USRP before applying per-channel RF settings.
  usrp_ = uhd::usrp::multi_usrp::make(args_.get());

  const size_t available_channels = usrp_->get_rx_num_channels();
  for (const auto channel_value : channels) {
    if (channel_value < 0 || static_cast<size_t>(channel_value) >= available_channels) {
      throw std::runtime_error("usrp_rx.channels contains an out-of-range channel index");
    }

    const size_t channel = static_cast<size_t>(channel_value);
    usrp_->set_rx_rate(rate_.get(), channel);
    usrp_->set_rx_freq(freq_.get(), channel);
    usrp_->set_rx_gain(gain_.get(), channel);
  }

  // Create one UHD streamer per channel so each RF path can target its own
  // destination UDP port / DAQIRI receive queue.
  rx_streamers_.reserve(channels.size());
  for (size_t index = 0; index < channels.size(); ++index) {
    const size_t channel = static_cast<size_t>(channels[index]);
    if (spp_.get() > 0) {
      usrp_->get_radio_control(channel).set_properties(
          "spp=" + std::to_string(spp_.get()), usrp_->get_rx_radio_channel(channel));
    }
    // Each UHD streamer targets one destination port so DAQIRI can map channels
    // to distinct receive queues.
    uhd::stream_args_t stream_args("sc16", "sc16");
    stream_args.channels = {channel};
    stream_args.args = uhd::device_addr_t(make_stream_args(dest_addr_.get(),
                                                           dest_ports[index],
                                                           adapter_.get(),
                                                           dest_mac_addr_.get(),
                                                           keep_hdr_.get(),
                                                           mtu_.get()));
    rx_streamers_.push_back(usrp_->get_rx_stream(stream_args));
  }

  uhd::time_spec_t start_time;
  // Multi-channel starts are scheduled slightly in the future so all channels
  // begin together rather than sequentially.
  const bool use_future_start = channels.size() > 1 && start_delay_seconds > 0.0;
  if (use_future_start) {
    start_time = uhd::time_spec_t(
        usrp_->get_time_now().get_real_secs() + start_delay_seconds);
  }

  try {
    for (auto& rx_streamer : rx_streamers_) {
      uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
      stream_cmd.stream_now = !use_future_start;
      if (use_future_start) {
        stream_cmd.time_spec = start_time;
      }
      rx_streamer->issue_stream_cmd(stream_cmd);
    }
  } catch (...) {
    const auto start_error = std::current_exception();
    // A command may have reached the radio before a later start failed. Stop
    // every created streamer so no partially started session is left running.
    for (auto& rx_streamer : rx_streamers_) {
      if (rx_streamer) {
        try {
          uhd::stream_cmd_t stop_cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
          rx_streamer->issue_stream_cmd(stop_cmd);
        } catch (const std::exception& error) {
          HOLOSCAN_LOG_ERROR("Failed to stop USRP streamer after start error: {}", error.what());
        } catch (...) {
          HOLOSCAN_LOG_ERROR("Failed to stop USRP streamer after start error");
        }
      }
    }
    rx_streamers_.clear();
    usrp_.reset();
    std::rethrow_exception(start_error);
  }

  started_ = true;
  HOLOSCAN_LOG_INFO("UsrpRxOp started {} channel(s) to {}", channels.size(), dest_addr_.get());
  for (const auto channel_value : channels) {
    const size_t channel = static_cast<size_t>(channel_value);
    HOLOSCAN_LOG_INFO("Channel {} configured: rate={} freq={} gain={} spp={}",
                      channel,
                      usrp_->get_rx_rate(channel),
                      usrp_->get_rx_freq(channel),
                      usrp_->get_rx_gain(channel),
                      spp_.get());
  }
}

// Streaming control is handled in start()/stop(); no per-tick compute work is needed.
void UsrpRxOp::compute(InputContext&, OutputContext&, ExecutionContext&) {}

void UsrpRxOp::stop() {
  if (started_) {
    // Tell every streamer to stop before releasing the UHD device handle.
    for (auto& rx_streamer : rx_streamers_) {
      if (rx_streamer) {
        try {
          uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
          rx_streamer->issue_stream_cmd(stream_cmd);
        } catch (const std::exception& error) {
          HOLOSCAN_LOG_ERROR("Failed to stop USRP streamer: {}", error.what());
        } catch (...) {
          HOLOSCAN_LOG_ERROR("Failed to stop USRP streamer");
        }
      }
    }
    HOLOSCAN_LOG_INFO("UsrpRxOp stopped USRP streaming");
  }

  rx_streamers_.clear();
  usrp_.reset();
  started_ = false;
}

}  // namespace holoscan::ops
