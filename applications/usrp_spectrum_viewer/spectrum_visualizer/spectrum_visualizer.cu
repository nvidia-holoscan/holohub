// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "spectrum_visualizer.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <gxf/std/tensor.hpp>

using namespace matx;

using in_t = std::tuple<tensor_t<float, 1>, cudaStream_t>;

namespace holoscan::ops {

namespace {

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(
        fmt::format("{} failed: {}", operation, cudaGetErrorString(result)));
  }
}

void log_cuda_error(cudaError_t result, const char* operation) noexcept {
  if (result != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("{} failed: {}", operation, cudaGetErrorString(result));
  }
}

// Ensure async input storage cannot be released on its producer stream until
// every read queued on the managed stream has completed, including on errors.
class InputLifetimeGuard {
 public:
  InputLifetimeGuard(cudaStream_t input_stream,
                     cudaStream_t operator_stream,
                     cudaEvent_t consumed_event)
      : input_stream_(input_stream),
        operator_stream_(operator_stream),
        consumed_event_(consumed_event) {}

  ~InputLifetimeGuard() {
    if (!active_) {
      return;
    }
    auto result = cudaEventRecord(consumed_event_, operator_stream_);
    if (result == cudaSuccess) {
      result = cudaStreamWaitEvent(input_stream_, consumed_event_, 0);
    }
    if (result != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Failed to preserve spectrum input lifetime: {}",
                         cudaGetErrorString(result));
    }
  }

  void complete() {
    check_cuda(cudaEventRecord(consumed_event_, operator_stream_), "cudaEventRecord");
    check_cuda(cudaStreamWaitEvent(input_stream_, consumed_event_, 0), "cudaStreamWaitEvent");
    active_ = false;
  }

 private:
  cudaStream_t input_stream_;
  cudaStream_t operator_stream_;
  cudaEvent_t consumed_event_;
  bool active_ = true;
};

}  // namespace

void SpectrumVisualizerOp::setup(OperatorSpec& spec) {
  spec.input<in_t>("in");
  spec.output<gxf::Entity>("outputs");
  spec.output<std::vector<HolovizOp::InputSpec>>("output_specs");

  spec.param(num_bins_,
      "num_bins",
      "Number of bins",
      "Number of frequency bins in the spectrum (length of the input tensor)");
  spec.param(num_channels_,
      "num_channels",
      "Number of channels",
      "Number of RF channels rendered as separate line strips");
  spec.param(db_min_,
      "db_min",
      "Minimum dB",
      "dB value mapped to the bottom of the plot");
  spec.param(db_max_,
      "db_max",
      "Maximum dB",
      "dB value mapped to the top of the plot");
  spec.param(ref_center_hz_,
      "ref_center_hz",
      "Reference center frequency (Hz)",
      "Radio's true center frequency; defines the absolute-frequency mapping of "
      "the captured band",
      1000.0e6F);
  spec.param(ref_bandwidth_hz_,
      "ref_bandwidth_hz",
      "Reference bandwidth (Hz)",
      "Radio's true captured bandwidth; defines the absolute-frequency span of "
      "the captured band",
      500.0e6F);
}

void SpectrumVisualizerOp::initialize() {
  // Allocator for GXF tensors emitted to Holoviz each frame.
  allocator_ = fragment()->make_resource<UnboundedAllocator>("pool");
  add_arg(allocator_);

  holoscan::Operator::initialize();

  if (num_bins_.get() == 0 || num_channels_.get() == 0) {
    throw std::runtime_error("spectrum_viz.num_bins and num_channels must both be > 0");
  }
  if (!std::isfinite(db_min_.get()) || !std::isfinite(db_max_.get())
      || db_max_.get() <= db_min_.get()) {
    throw std::runtime_error(fmt::format(
        "spectrum_viz.db_max ({}) must be finite and greater than db_min ({})",
        db_max_.get(), db_min_.get()));
  }
  if (!std::isfinite(ref_center_hz_.get())) {
    throw std::runtime_error("spectrum_viz.ref_center_hz must be finite");
  }
  if (!std::isfinite(ref_bandwidth_hz_.get()) || ref_bandwidth_hz_.get() <= 0.0F) {
    throw std::runtime_error(fmt::format(
        "spectrum_viz.ref_bandwidth_hz must be finite and > 0, got {}",
        ref_bandwidth_hz_.get()));
  }

  const index_t num_bins = static_cast<index_t>(num_bins_.get());
  const index_t num_channels = static_cast<index_t>(num_channels_.get());

  make_tensor(coords_, {num_channels, num_bins, 2}, MATX_DEVICE_MEMORY);
  make_tensor(minima_, {num_bins}, MATX_DEVICE_MEMORY);
  make_tensor(maxima_, {num_bins}, MATX_DEVICE_MEMORY);
  make_tensor(frac_, {num_bins}, MATX_DEVICE_MEMORY);

  // Per-channel host buffers and events for non-blocking peak detection.
  ch_host_db_.assign(static_cast<size_t>(num_channels), nullptr);
  peak_events_.assign(static_cast<size_t>(num_channels), nullptr);
  peak_copy_pending_.assign(static_cast<size_t>(num_channels), 0);
  for (size_t channel = 0; channel < static_cast<size_t>(num_channels); ++channel) {
    check_cuda(cudaMallocHost(reinterpret_cast<void**>(&ch_host_db_[channel]),
                              static_cast<size_t>(num_bins) * sizeof(float)),
               "cudaMallocHost");
    check_cuda(cudaEventCreateWithFlags(&peak_events_[channel], cudaEventDisableTiming),
               "cudaEventCreateWithFlags");
  }

  // Start with a flat line at the bottom; fill constant clamp bounds.
  (coords_ = 1.0F).run();
  (minima_ = 0.0F).run();
  (maxima_ = 1.0F).run();

  // FFT bin k represents k/N of the sample rate; the last bin is (N-1)/N.
  const float last_fraction = static_cast<float>(num_bins - 1) / static_cast<float>(num_bins);
  (frac_ = linspace(0.0F, last_fraction, num_bins)).run();
  for (index_t ch = 0; ch < num_channels; ++ch) {
    auto x_col = slice<1>(coords_, {ch, 0, 0}, {matxDropDim, matxEnd, matxDropDim});
    (x_col = frac_).run();
  }

  check_cuda(cudaEventCreateWithFlags(&input_ready_event_, cudaEventDisableTiming),
             "cudaEventCreateWithFlags");
  check_cuda(cudaEventCreateWithFlags(&input_consumed_event_, cudaEventDisableTiming),
             "cudaEventCreateWithFlags");

  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

SpectrumVisualizerOp::~SpectrumVisualizerOp() {
  if (input_ready_event_ != nullptr) {
    log_cuda_error(cudaEventDestroy(input_ready_event_), "cudaEventDestroy");
  }
  if (input_consumed_event_ != nullptr) {
    log_cuda_error(cudaEventDestroy(input_consumed_event_), "cudaEventDestroy");
  }
  for (size_t channel = 0; channel < peak_events_.size(); ++channel) {
    auto& peak_event = peak_events_[channel];
    if (peak_event != nullptr) {
      if (peak_copy_pending_[channel] != 0) {
        log_cuda_error(cudaEventSynchronize(peak_event), "cudaEventSynchronize");
      }
      log_cuda_error(cudaEventDestroy(peak_event), "cudaEventDestroy");
    }
    if (ch_host_db_[channel] != nullptr) {
      log_cuda_error(cudaFreeHost(ch_host_db_[channel]), "cudaFreeHost");
    }
  }
}

void SpectrumVisualizerOp::compute(InputContext& op_input,
                                 OutputContext& op_output,
                                 ExecutionContext& context) {
  auto input = op_input.receive<in_t>("in").value();
  auto in_db = std::get<0>(input);
  auto in_stream = std::get<1>(input);

  if (static_cast<index_t>(in_db.Size(0)) != static_cast<index_t>(num_bins_.get())) {
    throw std::runtime_error(fmt::format(
        "spectrum_viz.num_bins ({}) must match the upstream spectrum length ({})",
        num_bins_.get(), in_db.Size(0)));
  }

  auto meta = metadata();
  auto channel_num = meta->get<uint16_t>("channel_number", 0);
  if (channel_num >= num_channels_.get()) {
    throw std::runtime_error(fmt::format(
        "spectrum_viz.num_channels ({}) is too small for upstream channel {}",
        num_channels_.get(), channel_num));
  }

  // All GPU work runs on one managed stream; make it wait for the upstream
  cudaStream_t op_stream = op_input.receive_cuda_stream("in");
  check_cuda(cudaEventRecord(input_ready_event_, in_stream), "cudaEventRecord");
  check_cuda(cudaStreamWaitEvent(op_stream, input_ready_event_, 0), "cudaStreamWaitEvent");
  InputLifetimeGuard input_lifetime(in_stream, op_stream, input_consumed_event_);

  const float db_min = db_min_.get();
  const float db_max = db_max_.get();
  const float range = db_max - db_min;
  // With db_min = -120 and db_max = 0, the normalized value maps db_min -> 0
  // (bottom of the plot band) and db_max -> 1 (top of the plot band).
  const float inv_range = 1.0F / range;

  // Normalize dB to local viewport coordinates and flip because y grows down.
  auto y_col = slice<1>(coords_,
      {static_cast<index_t>(channel_num), 0, 1},
      {matxDropDim, matxEnd, matxDropDim});
  (y_col = 1.0F - min(max((in_db - db_min) * inv_range, minima_), maxima_)).run(op_stream);

  // Recompute x positions for the current pan/zoom view:
  //   x_i = (f_i - fc_disp_mhz)/bw_disp_mhz + 0.5
  const float fc_ref_mhz = ref_center_hz_.get() * 1.0e-6F;
  const float bw_ref_mhz = ref_bandwidth_hz_.get() * 1.0e-6F;
  float fc_disp_mhz = fc_ref_mhz;
  float bw_disp_mhz = bw_ref_mhz;
  if (view_state_) {
    view_state_->load_view(fc_disp_mhz, bw_disp_mhz);
  }
  if (!std::isfinite(fc_disp_mhz)) {
    fc_disp_mhz = fc_ref_mhz;
  }
  if (!std::isfinite(bw_disp_mhz) || bw_disp_mhz <= 0.0F) {
    bw_disp_mhz = bw_ref_mhz;
  }
  const float u_scale = bw_ref_mhz / bw_disp_mhz;
  const float u_offset =
      (fc_ref_mhz - HALF_BANDWIDTH_FRACTION * bw_ref_mhz - fc_disp_mhz) / bw_disp_mhz
      + PLOT_CENTER_U;
  const float scale_b = u_scale;
  const float offset_a = u_offset;
  // Refresh every channel's x ramp with the latest view parameters.
  for (uint16_t ch = 0; ch < num_channels_.get(); ++ch) {
    auto x_col = slice<1>(coords_,
        {static_cast<index_t>(ch), 0, 0},
        {matxDropDim, matxEnd, matxDropDim});
    (x_col = offset_a + scale_b * frac_).run(op_stream);
  }

  // Pack one (num_bins, 2) tensor and a LINE_STRIP spec per channel into a GXF entity.
  auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      context.context(), allocator_->gxf_cid());
  auto entity = gxf::Entity::New(&context);
  auto specs = std::vector<HolovizOp::InputSpec>();

  const auto num_bins = static_cast<int32_t>(num_bins_.get());
  const size_t channel_bytes = static_cast<size_t>(num_bins) * 2 * sizeof(float);

  for (uint16_t ch = 0; ch < num_channels_.get(); ++ch) {
    const std::string name = "spectrum_" + std::to_string(ch);

    auto tensor = static_cast<nvidia::gxf::Entity&>(entity)
                      .add<nvidia::gxf::Tensor>(name.c_str())
                      .value();
    tensor->reshape<float>(nvidia::gxf::Shape({num_bins, 2}),
        nvidia::gxf::MemoryStorageType::kDevice,
        allocator.value());

    auto block = slice<2>(coords_,
        {static_cast<index_t>(ch), 0, 0},
        {matxDropDim, matxEnd, matxEnd});
    check_cuda(cudaMemcpyAsync(tensor->pointer(),
                               block.Data(),
                               channel_bytes,
                               cudaMemcpyDeviceToDevice,
                               op_stream),
               "cudaMemcpyAsync");

    auto& spec = specs.emplace_back(
        HolovizOp::InputSpec(name, HolovizOp::InputType::LINE_STRIP));
    const auto& color = CHANNEL_PALETTE[ch % CHANNEL_PALETTE.size()];
    spec.line_width_ = 2.0F;
    spec.color_ = {color[0], color[1], color[2], color[3]};
    HolovizOp::InputSpec::View plot_view;
    plot_view.offset_x_ = PLOT_LEFT;
    plot_view.offset_y_ = PLOT_TOP;
    plot_view.width_ = PLOT_WIDTH;
    plot_view.height_ = PLOT_HEIGHT;
    spec.views_.push_back(plot_view);
  }

  // Peak detection without stalling: read last frame's D2H copy, then start the next.
  const bool want_peak =
      view_state_ && view_state_->show_peak.load(std::memory_order_relaxed);
  const auto num_bins_sz = static_cast<size_t>(num_bins_.get());
  if (want_peak && num_bins_sz > 0) {
    auto* host = ch_host_db_[channel_num];
    auto& peak_event = peak_events_[channel_num];
    auto& copy_pending = peak_copy_pending_[channel_num];

    // A previous frame's copy is ready: scan it on the CPU and publish.
    const auto query_result = copy_pending != 0
        ? cudaEventQuery(peak_event)
        : cudaErrorNotReady;
    if (copy_pending != 0 && query_result != cudaSuccess
        && query_result != cudaErrorNotReady) {
      check_cuda(query_result, "cudaEventQuery");
    }
    if (copy_pending != 0 && query_result == cudaSuccess) {
      size_t peak_idx = 0;
      float peak_val = host[0];
      for (size_t i = 1; i < num_bins_sz; ++i) {
        if (host[i] > peak_val) {
          peak_val = host[i];
          peak_idx = i;
        }
      }
      const float frac = static_cast<float>(peak_idx) / static_cast<float>(num_bins_sz);
      const float peak_frequency_mhz =
          fc_ref_mhz - HALF_BANDWIDTH_FRACTION * bw_ref_mhz + frac * bw_ref_mhz;

      // Publish the frequency and power as one atomic pair.
      if (channel_num < MAX_CHANNELS) {
        view_state_->store_peak(channel_num, peak_frequency_mhz, peak_val);
      }
      copy_pending = 0;
    }

    // Queue the next async D2H copy (ordered after this frame's GPU work).
    if (copy_pending == 0) {
      check_cuda(cudaMemcpyAsync(host,
                                 in_db.Data(),
                                 num_bins_sz * sizeof(float),
                                 cudaMemcpyDeviceToHost,
                                 op_stream),
                 "cudaMemcpyAsync");
      check_cuda(cudaEventRecord(peak_event, op_stream), "cudaEventRecord");
      copy_pending = 1;
    }
  }

  input_lifetime.complete();

  op_output.emit(entity, "outputs");
  op_output.emit(std::move(specs), "output_specs");
}

}  // namespace holoscan::ops
