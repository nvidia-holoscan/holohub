/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "process.h"

namespace holoscan::ops {

// ----- PulseCompressionOp ---------------------------------------------------
void PulseCompressionOp::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<RFArray>>("rf_in");
  spec.output<std::shared_ptr<ThreePulseCancellerData>>("pc_out");
  spec.param(num_pulses,
              "num_pulses",
              "Number of pulses",
              "Number of pulses per channel", {});
  spec.param(num_channels,
              "num_channels",
              "Number of channels",
              "Number of channels", {});
  spec.param(waveform_length,
              "waveform_length",
              "NWaveform length",
              "Length of waveform", {});
  spec.param(num_samples,
              "num_samples",
              "Number of samples",
              "Number of samples per channel", {});
}

void PulseCompressionOp::initialize() {
  HOLOSCAN_LOG_INFO("PulseCompressionOp::initialize()");
  holoscan::Operator::initialize();

  num_samples_rnd = 1;
  while (num_samples_rnd < num_samples.get()) {
    num_samples_rnd *= 2;
  }

  make_tensor(waveformView, {num_samples_rnd});
  cudaMemset(waveformView.Data(), 0, num_samples_rnd * sizeof(complex_t));

  make_tensor(zeroPaddedInput, {num_channels.get(), num_pulses.get(), num_samples_rnd});

  // Precondition FFT of waveform for matched filtering (assuming waveform is the
  // same for every pulse) this allows us to precompute waveform in frequency domain
  auto waveformPart = slice(waveformView, {0}, {waveform_length.get()});
  auto waveformFull = slice(waveformView, {0}, {num_samples_rnd});

  // Apply a Hamming window to the waveform to suppress sidelobes. Other
  // windows could be used as well (e.g., Taylor windows). Ultimately, it is
  // just an element-wise weighting by a pre-computed window function.
  (waveformPart = waveformPart * hamming<0>({waveform_length.get()})).run();

  // Normalize by L2 norm
  make_tensor(norms);
  (norms = sum(norm(waveformPart))).run();
  (norms = sqrt(norms)).run();
  (waveformPart = waveformPart / norms).run();

  // Do FFT
  (waveformFull = fft(waveformPart, num_samples_rnd)).run();
  (waveformFull = conj(waveformFull)).run();

  HOLOSCAN_LOG_INFO("PulseCompressionOp::initialize() done");
}

/**
 * @brief Stage 1 - Pulse compression - convolution via FFTs
 *
 * Pulse compression achieves high range resolution by applying intra-pulse
 * modulation during transmit followed by applying a matched filter after
 * reception. References:
 *    Richards, M. A., Scheer, J. A., Holm, W. A., "Principles of Modern
 *    Radar: Basic Principles", SciTech Publishing, Inc., 2010.  Chapter 20.
 *    Also, http://en.wikipedia.org/wiki/Pulse_compression
 */
void PulseCompressionOp::compute(InputContext& op_input,
                                 OutputContext& op_output,
                                 ExecutionContext&) {
  HOLOSCAN_LOG_INFO("PulseCompressionOp::compute() called");
  auto in = op_input.receive<std::shared_ptr<RFArray>>("rf_in").value();
  cudaStream_t stream = in->stream;

  auto waveformFFT = clone<3>(waveformView, {num_channels.get(), num_pulses.get(), matxKeepDim});

  HOLOSCAN_LOG_INFO("Dim: {}, {}, {}", in->data.Size(0), in->data.Size(1), in->data.Size(2));

  // Zero out the pad portion of the zero-padded input and copy the data portion
  auto zp = slice<3>(zeroPaddedInput, {0, 0, num_samples.get()},
                                      {matxEnd, matxEnd, num_samples_rnd});
  auto data = slice<3>(zeroPaddedInput, {0, 0, 0}, {matxEnd, matxEnd, num_samples.get()});
  (zp = 0).run(stream);
  matx::copy(data, in->data, stream);

  (zeroPaddedInput = fft(zeroPaddedInput)).run(stream);
  (zeroPaddedInput = zeroPaddedInput * waveformFFT).run(stream);
  (zeroPaddedInput = ifft(zeroPaddedInput)).run(stream);

  auto params = std::make_shared<ThreePulseCancellerData>(zeroPaddedInput, stream);
  op_output.emit(params, "pc_out");
}

// ----- ThreePulseCancellerOp ------------------------------------------------
void ThreePulseCancellerOp::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<ThreePulseCancellerData>>("tpc_in");
  spec.output<std::shared_ptr<DopplerData>>("tpc_out");
  spec.param(num_pulses,
              "num_pulses",
              "Number of pulses",
              "Number of pulses per channel", {});
  spec.param(num_channels,
              "num_channels",
              "Number of channels",
              "Number of channels", {});
  spec.param(waveform_length,
              "waveform_length",
              "NWaveform length",
              "Length of waveform", {});
  spec.param(num_samples,
              "num_samples",
              "Number of samples",
              "Number of samples per channel", {});
}

void ThreePulseCancellerOp::initialize() {
  HOLOSCAN_LOG_INFO("ThreePulseCancellerOp::initialize()");
  holoscan::Operator::initialize();

  num_pulses_rnd = 1;
  while (num_pulses_rnd <= num_pulses.get()) {
    num_pulses_rnd *= 2;
  }

  numCompressedSamples = num_samples.get() - waveform_length.get() + 1;
  make_tensor(tpcView, {num_channels.get(), num_pulses_rnd, numCompressedSamples});
  make_tensor(cancelMask, {3});
  cancelMask.SetVals({1, -2, 1});

  cudaMemset(tpcView.Data(), 0, tpcView.TotalSize() * sizeof(complex_t));

  tpcView.PrefetchDevice(0);
  cancelMask.PrefetchDevice(0);
  HOLOSCAN_LOG_INFO("ThreePulseCancellerOp::initialize() done");
}

/**
 * @brief Stage 2 - Three-pulse canceller - 1D convolution
 *
 * The three-pulse canceller is a simple high-pass filter designed to suppress
 * background, or "clutter", such as the ground and other non-moving objects.
 * The three-pulse canceller is a pair of two-pulse cancellers implemented in
 * a single stage. A two-pulse canceller just computes the difference between
 * two subsequent pulses at each range bin. Thus, the two pulse canceller is
 * equivalent to convolution in the pulse dimension with [1 -1] and the
 * three-pulse canceller is convolution in the pulse dimension with [1 -2 1]
 * ([1 -2 1] is just the convolution of [1 -1] with [1 -1], so it is
 * effectively a sequence of two two-pulse cancellers).
 * References:
 *   Richards, M. A., Scheer, J. A., Holm, W. A., "Principles of Modern Radar:
 *   Basic Principles",
 *       SciTech Publishing, Inc., 2010.  Section 17.4.
 */
void ThreePulseCancellerOp::compute(InputContext& op_input,
                                    OutputContext& op_output,
                                    ExecutionContext&) {
  HOLOSCAN_LOG_INFO("Three pulse canceller compute() called");
  auto tpc_data = op_input.receive<std::shared_ptr<ThreePulseCancellerData>>("tpc_in").value();

  auto x = tpc_data->inputView.Permute({0, 2, 1}).Slice(
      {0, 0, 0}, {num_channels.get(), numCompressedSamples, num_pulses.get()});
  auto xo = tpcView.Permute({0, 2, 1}).Slice(
      {0, 0, 0}, {num_channels.get(), numCompressedSamples, num_pulses.get()});
  (xo = conv1d(x, cancelMask, matxConvCorrMode_t::MATX_C_MODE_SAME)).run(tpc_data->stream);

  auto params = std::make_shared<DopplerData>(tpcView, cancelMask, tpc_data->stream);
  op_output.emit(params, "tpc_out");
}

// ----- DopplerOp ------------------------------------------------------------
void DopplerOp::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<DopplerData>>("dop_in");
  spec.output<std::shared_ptr<CFARData>>("dop_out");
  spec.param(num_pulses,
              "num_pulses",
              "Number of pulses",
              "Number of pulses per channel", {});
  spec.param(num_channels,
              "num_channels",
              "Number of channels",
              "Number of channels", {});
  spec.param(waveform_length,
              "waveform_length",
              "NWaveform length",
              "Length of waveform", {});
  spec.param(num_samples,
              "num_samples",
              "Number of samples",
              "Number of samples per channel", {});
}

void DopplerOp::initialize() {
  HOLOSCAN_LOG_INFO("DopplerOp::initialize()");
  holoscan::Operator::initialize();

  numCompressedSamples = num_samples.get() - waveform_length.get() + 1;
  HOLOSCAN_LOG_INFO("DopplerOp::initialize() done");
}

/**
 * @brief Stage 3 - Doppler Processing - FFTs in pulse
 *
 * Doppler processing converts the range-pulse data to range-Doppler data via
 * an FFT in the Doppler dimension. Explicit spectral analysis can then be
 * performed, such as the detector that will follow as stage 4.
 * References:
 *   Richards, M. A., Scheer, J. A., Holm, W. A., "Principles of Modern Radar:
 *   Basic Principles",
 *       SciTech Publishing, Inc., 2010.  Section 17.5.
 *
 * Apply a window in pulse to suppress sidelobes. Using a Hamming window for
 * simplicity, but others would work. repmat().
 */
void DopplerOp::compute(InputContext& op_input,
                        OutputContext& op_output,
                        ExecutionContext&) {
  HOLOSCAN_LOG_INFO("Doppler compute() called");
  auto dop_data = op_input.receive<std::shared_ptr<DopplerData>>("dop_in").value();

  const index_t cpulses = num_pulses.get() - (dop_data->cancelMask.Size(0) - 1);

  auto xc = dop_data->tpcView.Slice({0, 0, 0},
                                      {num_channels.get(), cpulses, numCompressedSamples});
  auto xf = dop_data->tpcView.Permute({0, 2, 1});

  (xc = xc * hamming<1>({num_channels.get(), num_pulses.get() - (dop_data->cancelMask.Size(0) - 1),
                        numCompressedSamples})).run(dop_data->stream);
  (xf = fft(xf)).run(dop_data->stream);

  auto params = std::make_shared<CFARData>(dop_data->tpcView, dop_data->stream);
  op_output.emit(params, "dop_out");
}

// ----- CFAROp ---------------------------------------------------------------
void CFAROp::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<CFARData>>("cfar_in");
  spec.param(num_transmits, "num_transmits",
              "Number of waveform transmissions",
              "Number of waveform transmissions to simulate", {});
  spec.param(num_pulses,
              "num_pulses",
              "Number of pulses",
              "Number of pulses per channel", {});
  spec.param(num_channels,
              "num_channels",
              "Number of channels",
              "Number of channels", {});
  spec.param(waveform_length,
              "waveform_length",
              "NWaveform length",
              "Length of waveform", {});
  spec.param(num_samples,
              "num_samples",
              "Number of samples",
              "Number of samples per channel", {});
}

void CFAROp::initialize() {
  HOLOSCAN_LOG_INFO("CFAROp::initialize()");
  holoscan::Operator::initialize();

  transmits = 0;

  num_pulses_rnd = 1;
  while (num_pulses_rnd <= num_pulses.get()) {
    num_pulses_rnd *= 2;
  }

  numCompressedSamples = num_samples.get() - waveform_length.get() + 1;

  make_tensor(normT, {num_channels.get(), num_pulses_rnd + cfarMaskY - 1,
              numCompressedSamples + cfarMaskX - 1});
  make_tensor(ba, {num_channels.get(), num_pulses_rnd + cfarMaskY - 1,
              numCompressedSamples + cfarMaskX - 1});
  make_tensor(dets, {num_channels.get(), num_pulses_rnd, numCompressedSamples});
  make_tensor(xPow, {num_channels.get(), num_pulses_rnd, numCompressedSamples});
  make_tensor(cfarMaskView, {cfarMaskY, cfarMaskX});

  // Mask for cfar detection
  // G == guard, R == reference, C == CUT
  // mask = [
  //    R R R R R ;
  //    R R R R R ;
  //    R R R R R ;
  //    R R R R R ;
  //    R R R R R ;
  //    R G G G R ;
  //    R G C G R ;
  //    R G G G R ;
  //    R R R R R ;
  //    R R R R R ;
  //    R R R R R ;
  //    R R R R R ;
  //    R R R R R ];
  //  }
  cfarMaskView.SetVals({{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                          {1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1},
                          {1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1},
                          {1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1},
                          {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});

  // Pre-process CFAR convolution
  (normT = conv2d(ones({num_channels.get(), num_pulses_rnd, numCompressedSamples}), cfarMaskView,
           matxConvCorrMode_t::MATX_C_MODE_FULL)).run();

  ba.PrefetchDevice(0);
  normT.PrefetchDevice(0);
  cfarMaskView.PrefetchDevice(0);
  dets.PrefetchDevice(0);
  xPow.PrefetchDevice(0);

  HOLOSCAN_LOG_INFO("CFAROp::initialize() done");
}

/**
 * @brief Stage 4 - Constant False Alarm Rate (CFAR) Detector - averaging or median
 *
 * filter CFAR detectors in general are designed to provide constant false
 * alarm rates by dynamically adjusting detection thresholds based on certain
 * statistical assumptions and interference estimates made from the data.
 * References:
 *   Richards, M. A., Scheer, J. A., Holm, W. A., "Principles of Modern Radar:
 *   Basic Principles",
 *       SciTech Publishing, Inc., 2010.  Section 16.4.
 *   Richards, M. A., "Fundamentals of Radar Signal Processing", McGraw-Hill,
 *   2005.
 *       Chapter 7. alpha below corresponds to equation (7.17)
 *   Also, http://en.wikipedia.org/wiki/Constant_false_alarm_rate

  * CFAR works by using a training window to average cells "near" a cell
  * under test (CUT) to estimate the background power for that cell. It is
  * an assumption that the average of the nearby cells represents a
  * reasonable background estimate. In general, there are guard cells (G)
  * and reference cells (R) around the CUT. The guard cells prevent
  * contributions of a potential target in the CUT from corrupting the
  * background estimate. More reference cells are preferred to better
  * estimate the background average. As implemented below, the CUT and
  * guard cells form a hole within the training window, but CA-CFAR is
  * largely just an averaging filter otherwise with a threshold check
  * at each pixel after applying the filter.
  * Currently, the window below is defined statically because it is then
  * easy to visualize, but more typically the number of guard and
  * reference cells would be given as input and the window would be
  * constructed; we could update to such an approach, but I'm keeping
  * it simple for now.

  * We apply CFAR to the power of X; X is still complex until this point
  * Xpow = abs(X).^2;
  */
void CFAROp::compute(InputContext& op_input,
                     OutputContext&,
                     ExecutionContext& context) {
  HOLOSCAN_LOG_INFO("CFAR compute() called");
  auto cfar_data = op_input.receive<std::shared_ptr<CFARData>>("cfar_in").value();

  (xPow = norm(cfar_data->tpcView)).run(cfar_data->stream);

  // Estimate the background average power in each cell
  // background_averages = conv2(Xpow, mask, 'same') ./ norm;
  (ba = conv2d(xPow, cfarMaskView, matxConvCorrMode_t::MATX_C_MODE_FULL)).run(cfar_data->stream);

  // Computing number of cells contributing to each cell.
  // This can be done with a convolution of the cfarMask with
  // ones.
  // norm = conv2(ones(size(X)), mask, 'same');
  auto normTrim = normT.Slice({0, cfarMaskY / 2, cfarMaskX / 2},
                              {num_channels.get(), num_pulses_rnd + cfarMaskY / 2,
                               numCompressedSamples + cfarMaskX / 2});

  auto baTrim = ba.Slice({0, cfarMaskY / 2, cfarMaskX / 2},
                         {num_channels.get(), num_pulses_rnd + cfarMaskY / 2,
                          numCompressedSamples + cfarMaskX / 2});
  (baTrim = baTrim / normTrim).run(cfar_data->stream);

  // The scalar alpha is used as a multiplier on the background averages
  // to achieve a constant false alarm rate (under certain assumptions);
  // it is based upon the desired probability of false alarm (Pfa) and
  // number of reference cells used to estimate the background for the
  // CUT. For the purposes of computation, it can be assumed as a given
  // constant, although it does vary at the edges due to the different
  // training windows.
  // Declare a detection if the power exceeds the background estimate
  // times alpha for a particular cell.
  // dets(find(Xpow > alpha.*background_averages)) = 1;

  // These 2 branches are functionally equivalent.  A custom op is more
  // efficient as it can avoid repeated loads.
  calcDets(dets, xPow, baTrim, normTrim, pfa).run(cfar_data->stream);

  // Interrupt if we're done
  transmits++;
  if (transmits == num_transmits.get()) {
    HOLOSCAN_LOG_INFO("Received {} of {} transmits, exiting...", transmits, num_transmits.get());
    GxfGraphInterrupt(context.context());
  }
}

}  // namespace holoscan::ops
