// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "holoscan/holoscan.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"
#include "matx.h"

#include "spectrum_view_state.hpp"

namespace holoscan::ops {

/**
 * Bridge operator that converts a 1-D dB spectrum from SpectrumMagnitudeOp into
 * Holoviz line geometry: one interleaved (num_bins, 2) tensor per channel plus a
 * matching LINE_STRIP InputSpec, emitted in a single gxf::Entity.
 */
class SpectrumVisualizerOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SpectrumVisualizerOp)

  SpectrumVisualizerOp() = default;
  ~SpectrumVisualizerOp() override;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input,
               OutputContext& op_output,
               ExecutionContext& context) override;

  // Connect the live UI view parameters used to pan/zoom the spectrum.
  void set_view_state(std::shared_ptr<SpectrumViewState> state) {
    view_state_ = std::move(state);
  }

 private:
  // Allocator used to back the gxf tensors emitted to Holoviz.
  std::shared_ptr<UnboundedAllocator> allocator_;

  // Bridges the upstream tuple stream to this operator's managed output stream,
  // then keeps the input alive until all managed-stream reads have completed.
  cudaEvent_t input_ready_event_{nullptr};
  cudaEvent_t input_consumed_event_{nullptr};

  // Per-channel line geometry (num_channels, num_bins, 2): column 0 = x ramp,
  // column 1 = normalized y (dB).
  matx::tensor_t<float, 3> coords_;

  // Constant clamp bounds (0.0 / 1.0) applied to the normalized y value.
  matx::tensor_t<float, 1> minima_;
  matx::tensor_t<float, 1> maxima_;

  // Base 0..1 ramp (one per bin) rescaled to the current view each frame.
  matx::tensor_t<float, 1> frac_;

  // Live view parameters written by the UI overlay.
  std::shared_ptr<SpectrumViewState> view_state_;

  Parameter<uint32_t> num_bins_;
  Parameter<uint16_t> num_channels_;
  Parameter<float> db_min_;
  Parameter<float> db_max_;

  // Radio's true tuning, defining the absolute-frequency mapping of the
  // captured band.
  Parameter<float> ref_center_hz_;
  Parameter<float> ref_bandwidth_hz_;

  // Used to report the peak (highest dB bin) and its absolute frequency back to
  // the UI overlay. Peak detection is non-blocking: each channel's dB bins are
  // copied D2H asynchronously and scanned a frame later once the copy's event
  // has completed, so compute() never stalls the render pipeline.
  std::vector<float*> ch_host_db_;
  std::vector<cudaEvent_t> peak_events_;
  std::vector<uint8_t> peak_copy_pending_;
};

}  // namespace holoscan::ops
