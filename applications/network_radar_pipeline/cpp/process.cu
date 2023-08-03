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
  spec.param(numPulses,
              "numPulses",
              "Number of pulses",
              "Number of pulses per channel", {});
  spec.param(numChannels,
              "numChannels",
              "Number of channels",
              "Number of channels", {});
  spec.param(waveformLength,
              "waveformLength",
              "NWaveform length",
              "Length of waveform", {});
  spec.param(numSamples,
              "numSamples",
              "Number of samples",
              "Number of samples per channel", {});
}

void PulseCompressionOp::initialize() {
  HOLOSCAN_LOG_INFO("PulseCompressionOp::initialize()");
  holoscan::Operator::initialize();

  numSamplesRnd = 1;
  while (numSamplesRnd < numSamples.get()) {
    numSamplesRnd *= 2;
  }

  waveformView = new tensor_t<complex_t, 1>({numSamplesRnd});
  cudaMemset(waveformView->Data(), 0, numSamplesRnd * sizeof(complex_t));

  // Precondition FFT of waveform for matched filtering (assuming waveform is the
  // same for every pulse) this allows us to precompute waveform in frequency domain
  auto waveformPart = waveformView->Slice({0}, {waveformLength.get()});
  auto waveformFull = waveformView->Slice({0}, {numSamplesRnd});

  // Apply a Hamming window to the waveform to suppress sidelobes. Other
  // windows could be used as well (e.g., Taylor windows). Ultimately, it is
  // just an element-wise weighting by a pre-computed window function.
  (waveformPart = waveformPart * hamming<0>({waveformLength.get()})).run();

  // Normalize by L2 norm
  norms = new tensor_t<float_t, 0>();
  sum(*norms, norm(waveformPart));
  (*norms = sqrt(*norms)).run();
  (waveformPart = waveformPart / *norms).run();

  // Do FFT
  fft(waveformFull, waveformPart, 0);
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
  auto x = in->data;
  cudaStream_t stream = in->stream;

  auto waveformFFT = waveformView->template Clone<3>(
    {numChannels.get(), numPulses.get(), matxKeepDim});

  HOLOSCAN_LOG_INFO("Dim: {}, {}, {}", x.Size(0), x.Size(1), x.Size(2));

  fft(x, x, 0, stream);
  (x = x * waveformFFT).run(stream);
  ifft(x, x, 0, stream);

  auto params = std::make_shared<ThreePulseCancellerData>(in->data, stream);
  op_output.emit(params, "pc_out");
}

// ----- ThreePulseCancellerOp ------------------------------------------------
void ThreePulseCancellerOp::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<ThreePulseCancellerData>>("tpc_in");
  spec.output<std::shared_ptr<DopplerData>>("tpc_out");
  spec.param(numPulses,
              "numPulses",
              "Number of pulses",
              "Number of pulses per channel", {});
  spec.param(numChannels,
              "numChannels",
              "Number of channels",
              "Number of channels", {});
  spec.param(waveformLength,
              "waveformLength",
              "NWaveform length",
              "Length of waveform", {});
  spec.param(numSamples,
              "numSamples",
              "Number of samples",
              "Number of samples per channel", {});
}

void ThreePulseCancellerOp::initialize() {
  HOLOSCAN_LOG_INFO("ThreePulseCancellerOp::initialize()");
  holoscan::Operator::initialize();

  numPulsesRnd = 1;
  while (numPulsesRnd <= numPulses.get()) {
    numPulsesRnd *= 2;
  }

  numCompressedSamples = numSamples.get() - waveformLength.get() + 1;
  tpcView = new tensor_t<complex_t, 3>(
      {numChannels.get(), numPulsesRnd, numCompressedSamples});
  cancelMask = new tensor_t<float_t, 1>({3});
  cancelMask->SetVals({1, -2, 1});

  cudaMemset(tpcView->Data(), 0, tpcView->TotalSize() * sizeof(complex_t));

  tpcView->PrefetchDevice(0);
  cancelMask->PrefetchDevice(0);
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
      {0, 0, 0}, {numChannels.get(), numCompressedSamples, numPulses.get()});
  auto xo = tpcView->Permute({0, 2, 1}).Slice(
      {0, 0, 0}, {numChannels.get(), numCompressedSamples, numPulses.get()});
  conv1d(xo, x, *cancelMask, matxConvCorrMode_t::MATX_C_MODE_SAME, tpc_data->stream);

  auto params = std::make_shared<DopplerData>(tpcView, cancelMask, tpc_data->stream);
  op_output.emit(params, "tpc_out");
}

// ----- DopplerOp ------------------------------------------------------------
void DopplerOp::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<DopplerData>>("dop_in");
  spec.output<std::shared_ptr<CFARData>>("dop_out");
  spec.param(numPulses,
              "numPulses",
              "Number of pulses",
              "Number of pulses per channel", {});
  spec.param(numChannels,
              "numChannels",
              "Number of channels",
              "Number of channels", {});
  spec.param(waveformLength,
              "waveformLength",
              "NWaveform length",
              "Length of waveform", {});
  spec.param(numSamples,
              "numSamples",
              "Number of samples",
              "Number of samples per channel", {});
}

void DopplerOp::initialize() {
  HOLOSCAN_LOG_INFO("DopplerOp::initialize()");
  holoscan::Operator::initialize();

  numCompressedSamples = numSamples.get() - waveformLength.get() + 1;
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

  const index_t cpulses = numPulses.get() - (dop_data->cancelMask->Size(0) - 1);

  auto xc = dop_data->tpcView->Slice({0, 0, 0},
                                      {numChannels.get(), cpulses, numCompressedSamples});
  auto xf = dop_data->tpcView->Permute({0, 2, 1});

  (xc = xc * hamming<1>({numChannels.get(), numPulses.get() - (dop_data->cancelMask->Size(0) - 1),
                        numCompressedSamples})).run(dop_data->stream);
  fft(xf, xf, 0, dop_data->stream);

  auto params = std::make_shared<CFARData>(dop_data->tpcView, dop_data->stream);
  op_output.emit(params, "dop_out");
}

// ----- CFAROp ---------------------------------------------------------------
void CFAROp::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<CFARData>>("cfar_in");
  spec.param(numTransmits, "numTransmits",
              "Number of waveform transmissions",
              "Number of waveform transmissions to simulate", {});
  spec.param(numPulses,
              "numPulses",
              "Number of pulses",
              "Number of pulses per channel", {});
  spec.param(numChannels,
              "numChannels",
              "Number of channels",
              "Number of channels", {});
  spec.param(waveformLength,
              "waveformLength",
              "NWaveform length",
              "Length of waveform", {});
  spec.param(numSamples,
              "numSamples",
              "Number of samples",
              "Number of samples per channel", {});
}

void CFAROp::initialize() {
  HOLOSCAN_LOG_INFO("CFAROp::initialize()");
  holoscan::Operator::initialize();

  transmits = 0;

  numPulsesRnd = 1;
  while (numPulsesRnd <= numPulses.get()) {
    numPulsesRnd *= 2;
  }

  numCompressedSamples = numSamples.get() - waveformLength.get() + 1;

  normT = new tensor_t<float_t, 3>(
      {numChannels.get(), numPulsesRnd + cfarMaskY - 1,
        numCompressedSamples + cfarMaskX - 1});
  ba = new tensor_t<float_t, 3>(
      {numChannels.get(), numPulsesRnd + cfarMaskY - 1,
        numCompressedSamples + cfarMaskX - 1});
  dets = new tensor_t<int, 3>(
      {numChannels.get(), numPulsesRnd, numCompressedSamples});
  xPow = new tensor_t<float_t, 3>(
      {numChannels.get(), numPulsesRnd, numCompressedSamples});
  cfarMaskView = new tensor_t<float_t, 2>(
      {cfarMaskY, cfarMaskX});

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
  cfarMaskView->SetVals({{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                          {1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1},
                          {1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1},
                          {1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1},
                          {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});

  // Pre-process CFAR convolution
  conv2d(*normT, ones({numChannels.get(), numPulsesRnd, numCompressedSamples}),
          *cfarMaskView, matxConvCorrMode_t::MATX_C_MODE_FULL, 0);

  ba->PrefetchDevice(0);
  normT->PrefetchDevice(0);
  cfarMaskView->PrefetchDevice(0);
  dets->PrefetchDevice(0);
  xPow->PrefetchDevice(0);

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

  (*xPow = norm(*cfar_data->tpcView)).run(cfar_data->stream);

  // Estimate the background average power in each cell
  // background_averages = conv2(Xpow, mask, 'same') ./ norm;
  conv2d(*ba, *xPow, *cfarMaskView, matxConvCorrMode_t::MATX_C_MODE_FULL, cfar_data->stream);

  // Computing number of cells contributing to each cell.
  // This can be done with a convolution of the cfarMask with
  // ones.
  // norm = conv2(ones(size(X)), mask, 'same');
  auto normTrim = normT->Slice({0, cfarMaskY / 2, cfarMaskX / 2},
                                {numChannels.get(), numPulsesRnd + cfarMaskY / 2,
                                numCompressedSamples + cfarMaskX / 2});

  auto baTrim = ba->Slice({0, cfarMaskY / 2, cfarMaskX / 2},
                          {numChannels.get(), numPulsesRnd + cfarMaskY / 2,
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
  calcDets(*dets, *xPow, baTrim, normTrim, pfa).run(cfar_data->stream);

  // Interrupt if we're done
  transmits++;
  if (transmits == numTransmits.get()) {
    HOLOSCAN_LOG_INFO("Received {} of {} transmits, exiting...", transmits, numTransmits.get());
    GxfGraphInterrupt(context.context());
  }
}

}  // namespace holoscan::ops