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

#include "holoscan/holoscan.hpp"
#include <cuda/std/complex>
#include "matx.h"

using namespace matx;
using ftype = float;
using ComplexType = cuda::std::complex<ftype>;

/* Structures for passing parameters between operators */
struct PulseCompressionData {
  PulseCompressionData(tensor_t<ComplexType, 1> _waveformView,
                        tensor_t<ComplexType, 3> _inputView,
                        cudaStream_t _stream)
    : waveformView(_waveformView), inputView(_inputView), stream(_stream)  {}
  tensor_t<ComplexType, 1> waveformView;
  tensor_t<ComplexType, 3> inputView;
  cudaStream_t stream;
};

struct ThreePulseCancellerData {
  ThreePulseCancellerData(tensor_t<ComplexType, 3> _inputView,
                          cudaStream_t _stream)
    : inputView(_inputView), stream(_stream)  {}
  tensor_t<ComplexType, 3> inputView;
  cudaStream_t stream;
};

struct DopplerData {
  DopplerData(tensor_t<ComplexType, 3> _tpcView,
              tensor_t<ftype, 1> _cancelMask,
              cudaStream_t _stream)
    : tpcView(_tpcView), cancelMask(_cancelMask), stream(_stream)  {}
  tensor_t<ComplexType, 3> tpcView;
  tensor_t<ftype, 1> cancelMask;
  cudaStream_t stream;
};

struct CFARData {
  CFARData(tensor_t<ComplexType, 3> _tpcView,
            cudaStream_t _stream)
    : tpcView(_tpcView), stream(_stream)  {}
  tensor_t<ComplexType, 3> tpcView;
  cudaStream_t stream;
};

/* Custom MatX operator for calculation detections in CFAR step. */
template <class O, class I1, class I2, class I3, class I4>
class calcDets : public BaseOp<calcDets<O, I1, I2, I3, I4>> {
 private:
    O out_;
    I1 xpow_;
    I2 ba_;
    I3 norm_;
    I4 pfa_;

 public:
    calcDets(O out, I1 xpow, I2 ba, I3 norm, I4 pfa)
        : out_(out), xpow_(xpow), ba_(ba), norm_(norm), pfa_(pfa)  { }

    __device__ inline void operator()(index_t idz, index_t idy, index_t idx) {
      typename I1::type xpow = xpow_(idz, idy, idx);
      typename I2::type ba = ba_(idz, idy, idx);
      typename I2::type norm = norm_(idz, idy, idx);
      typename I2::type alpha = norm * (cuda::std::powf(pfa_, -1.0f / norm) - 1.f);
      out_(idz, idy, idx) = (xpow > alpha * ba) ? 1 : 0;
    }

    __host__ __device__ inline index_t Size(uint32_t i) const {
      return out_.Size(i);
    }

    static inline constexpr __host__ __device__ int32_t Rank() {
      return O::Rank();
    }
};



namespace holoscan::ops {

class PulseCompressionOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PulseCompressionOp)

  PulseCompressionOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<ThreePulseCancellerData>("pc_out");
    spec.param(numPulses, "numPulses", "Number of pulses", "Number of pulses per channel", {});
    spec.param(numChannels, "numChannels", "Number of channels", "Number of channels", {});
    spec.param(waveformLength, "waveformLength", "NWaveform length", "Length of waveform", {});
    spec.param(numSamples, "numSamples", "Number of samples", "Number of samples per channel", {});
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("PulseCompressionOp::initialize()");
    holoscan::Operator::initialize();

    cudaStreamCreate(&stream);

    numSamplesRnd = 1;
    while (numSamplesRnd < numSamples.get()) {
      numSamplesRnd *= 2;
    }

    make_tensor(waveformView, {numSamplesRnd});
    make_tensor(norms);
    make_tensor(inputView, {numChannels.get(), numPulses.get(), numSamplesRnd});

    cudaMemset(waveformView.Data(), 0, numSamplesRnd * sizeof(ComplexType));
    cudaMemset(inputView.Data(), 0, inputView.TotalSize() * sizeof(ComplexType));

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
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    HOLOSCAN_LOG_DEBUG("Pulse compression compute() called");

    // Reshape waveform to be waveformLength
    auto waveformPart = slice(waveformView, {0}, {waveformLength.get()});
    auto waveformT    = clone<3>(waveformView,
                {numChannels.get(), numPulses.get(), matxKeepDim});
    auto waveformFull = slice(waveformView, {0}, {numSamplesRnd});

    auto x = inputView;

    // create waveform (assuming waveform is the same for every pulse)
    // this allows us to precompute waveform in frequency domain
    // Apply a Hamming window to the waveform to suppress sidelobes. Other
    // windows could be used as well (e.g., Taylor windows). Ultimately, it is
    // just an element-wise weighting by a pre-computed window function.
    (waveformPart = waveformPart * hamming<0>({waveformLength.get()})).run(stream);

    // compute L2 norm
    (norms = vector_norm(waveformPart, NormOrder::L2)).run(stream);

    (waveformPart = waveformPart / norms).run(stream);
    (waveformFull = fft(waveformPart, waveformFull.Size(waveformFull.Rank() - 1))).run(stream);
    (waveformFull = conj(waveformFull)).run(stream);

    (x = fft(x)).run(stream);
    (x = x * waveformT).run(stream);
    (x = ifft(x)).run(stream);

    auto params = ThreePulseCancellerData{inputView, stream};
    op_output.emit(params, "pc_out");
  };

 private:
  cudaStream_t stream;
  Parameter<int64_t> numPulses;
  Parameter<int64_t> numSamples;
  Parameter<int64_t> waveformLength;
  Parameter<int64_t> numChannels;
  index_t numSamplesRnd;

  tensor_t<ComplexType, 1> waveformView;
  tensor_t<ComplexType, 3> inputView;
  tensor_t<ftype, 0> norms;
};

class ThreePulseCancellerOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ThreePulseCancellerOp)

  ThreePulseCancellerOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<ThreePulseCancellerData>("tpc_in");
    spec.output<DopplerData>("tpc_out");
    spec.param(numPulses, "numPulses", "Number of pulses", "Number of pulses per channel", {});
    spec.param(numChannels, "numChannels", "Number of channels", "Number of channels", {});
    spec.param(waveformLength, "waveformLength", "NWaveform length", "Length of waveform", {});
    spec.param(numSamples, "numSamples", "Number of samples", "Number of samples per channel", {});
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("ThreePulseCancellerOp::initialize()");
    holoscan::Operator::initialize();

    numPulsesRnd = 1;
    while (numPulsesRnd <= numPulses.get()) {
      numPulsesRnd *= 2;
    }

    numCompressedSamples = numSamples.get() - waveformLength.get() + 1;
    make_tensor(tpcView,
        {numChannels.get(), numPulsesRnd, numCompressedSamples});
    make_tensor(cancelMask, {3});

    cancelMask.SetVals({1, -2, 1});

    cudaMemset(tpcView.Data(), 0, tpcView.TotalSize() * sizeof(ComplexType));

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
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    HOLOSCAN_LOG_DEBUG("Three pulse canceller compute() called");
    auto tpc_data = op_input.receive<ThreePulseCancellerData>("tpc_in");
    if (!tpc_data) {
      HOLOSCAN_LOG_CRITICAL("Failed to get tpc_data structure");
      return;
    }

    auto tpc_data_val = tpc_data.value();

    auto x = slice(permute(tpc_data_val.inputView, {0, 2, 1}),
        {0, 0, 0}, {numChannels.get(), numCompressedSamples, numPulses.get()});
    auto xo = slice(permute(tpcView, {0, 2, 1}),
        {0, 0, 0}, {numChannels.get(), numCompressedSamples, numPulses.get()});
    (xo = conv1d(x, cancelMask, matxConvCorrMode_t::MATX_C_MODE_SAME)).run(tpc_data_val.stream);

    auto params = DopplerData{tpcView, cancelMask, tpc_data_val.stream};
    op_output.emit(params, "tpc_out");
  };

 private:
  Parameter<int64_t> numPulses;
  Parameter<int64_t> numSamples;
  Parameter<int64_t> waveformLength;
  Parameter<int64_t> numChannels;
  index_t numCompressedSamples;
  index_t numPulsesRnd;

  tensor_t<ftype, 1> cancelMask;
  tensor_t<ComplexType, 3> tpcView;
};

class DopplerOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DopplerOp)

  DopplerOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<DopplerData>("dop_in");
    spec.output<CFARData>("dop_out");
    spec.param(numPulses, "numPulses", "Number of pulses", "Number of pulses per channel", {});
    spec.param(numChannels, "numChannels", "Number of channels", "Number of channels", {});
    spec.param(waveformLength, "waveformLength", "NWaveform length", "Length of waveform", {});
    spec.param(numSamples, "numSamples", "Number of samples", "Number of samples per channel", {});
  }

  void initialize() override {
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
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    HOLOSCAN_LOG_DEBUG("Doppler compute() called");
    auto dop_data = op_input.receive<DopplerData>("dop_in");

    if (!dop_data) {
      HOLOSCAN_LOG_CRITICAL("Failed to get dop_data structure");
      return;
    }

    auto dop_data_val = dop_data.value();

    const index_t cpulses = numPulses.get() - (dop_data_val.cancelMask.Size(0) - 1);

    auto xc = slice(dop_data->tpcView, {0, 0, 0},
          {numChannels.get(), cpulses, numCompressedSamples});
    auto xf = permute(dop_data_val.tpcView, {0, 2, 1});

    (xc = xc * hamming<1>({numChannels.get(),
                          numPulses.get() - (dop_data_val.cancelMask.Size(0) - 1),
                          numCompressedSamples})).run(dop_data_val.stream);
    (xf = fft(xf)).run(dop_data_val.stream);

    auto params = CFARData{dop_data_val.tpcView, dop_data_val.stream};
    op_output.emit(params, "dop_out");
  };

 private:
  Parameter<int64_t> numPulses;
  Parameter<int64_t> numSamples;
  Parameter<int64_t> waveformLength;
  Parameter<int64_t> numChannels;
  index_t numCompressedSamples;
};

class CFAROp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CFAROp)

  CFAROp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<CFARData>("cfar_in");
    spec.param(numPulses, "numPulses", "Number of pulses", "Number of pulses per channel", {});
    spec.param(numChannels, "numChannels", "Number of channels", "Number of channels", {});
    spec.param(waveformLength, "waveformLength", "NWaveform length", "Length of waveform", {});
    spec.param(numSamples, "numSamples", "Number of samples", "Number of samples per channel", {});
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("CFAROp::initialize()");
    holoscan::Operator::initialize();

    numPulsesRnd = 1;
    while (numPulsesRnd <= numPulses.get()) {
      numPulsesRnd *= 2;
    }

    numCompressedSamples = numSamples.get() - waveformLength.get() + 1;

    make_tensor(normT,
        {numChannels.get(), numPulsesRnd + cfarMaskY - 1,
         numCompressedSamples + cfarMaskX - 1});
    make_tensor(ba,
        {numChannels.get(), numPulsesRnd + cfarMaskY - 1,
         numCompressedSamples + cfarMaskX - 1});
    make_tensor(dets,
        {numChannels.get(), numPulsesRnd, numCompressedSamples});
    make_tensor(xPow,
        {numChannels.get(), numPulsesRnd, numCompressedSamples});
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
    (normT = conv2d(ones({numChannels.get(), numPulsesRnd, numCompressedSamples}),
           cfarMaskView, matxConvCorrMode_t::MATX_C_MODE_FULL)).run();

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
  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    HOLOSCAN_LOG_DEBUG("CFAR compute() called");
    auto cfar_data = op_input.receive<CFARData>("cfar_in");

    if (!cfar_data) {
      HOLOSCAN_LOG_CRITICAL("Failed to get cfar_data structure");
      return;
    }

    auto cfar_data_val = cfar_data.value();

    (xPow = abs2(cfar_data_val.tpcView)).run(cfar_data_val.stream);

    // Estimate the background average power in each cell
    // background_averages = conv2(Xpow, mask, 'same') ./ norm;
    (ba =
      conv2d(xPow, cfarMaskView, matxConvCorrMode_t::MATX_C_MODE_FULL)).run(cfar_data_val.stream);

    // Computing number of cells contributing to each cell.
    // This can be done with a convolution of the cfarMask with
    // ones.
    // norm = conv2(ones(size(X)), mask, 'same');
    auto normTrim = slice(normT, {0, cfarMaskY / 2, cfarMaskX / 2},
                                 {numChannels.get(), numPulsesRnd + cfarMaskY / 2,
                                  numCompressedSamples + cfarMaskX / 2});

    auto baTrim = slice(ba, {0, cfarMaskY / 2, cfarMaskX / 2},
                            {numChannels.get(), numPulsesRnd + cfarMaskY / 2,
                             numCompressedSamples + cfarMaskX / 2});
    (baTrim = baTrim / normTrim).run(cfar_data_val.stream);

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
    calcDets(dets, xPow, baTrim, normTrim, pfa).run(cfar_data_val.stream);
  };

 private:
  Parameter<int64_t> numPulses;
  Parameter<int64_t> numSamples;
  Parameter<int64_t> waveformLength;
  Parameter<int64_t> numChannels;
  index_t numCompressedSamples;
  index_t numPulsesRnd;
  const index_t cfarMaskX = 13;
  const index_t cfarMaskY = 5;
  static const constexpr float pfa = 1e-5f;

  tensor_t<ftype, 3> normT;
  tensor_t<ftype, 3> ba;
  tensor_t<int, 3> dets;
  tensor_t<ftype, 3> xPow;
  tensor_t<ftype, 2> cfarMaskView;
};



}  // namespace holoscan::ops

static constexpr int num_iterations = 100;

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto pc   = make_operator<ops::PulseCompressionOp>("pulse_compression",
          from_config("radar_pipeline"), make_condition<CountCondition>(num_iterations));
    auto tpc  = make_operator<ops::ThreePulseCancellerOp>("three_pulse_canceller",
          from_config("radar_pipeline"));
    auto dop  = make_operator<ops::DopplerOp>("doppler", from_config("radar_pipeline"));
    auto cfar = make_operator<ops::CFAROp>("cfar", from_config("radar_pipeline"));

    channels   = from_config("radar_pipeline.numChannels").as<int>();
    num_pulses = from_config("radar_pipeline.numPulses").as<int>();

    add_flow(pc, tpc,   {{"pc_out", "tpc_in"}});
    add_flow(tpc, dop,  {{"tpc_out", "dop_in"}});
    add_flow(dop, cfar, {{"dop_out", "cfar_in"}});
  }

  int GetChannels() const {
    return channels;
  }

  int GetPulses() const {
    return num_pulses;
  }

 private:
  int channels;
  int num_pulses;
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Get the configuration
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/simple_radar_pipeline.yaml";
  app->config(config_path);

  const auto start = std::chrono::steady_clock::now();
  app->run();
  const auto stop = std::chrono::steady_clock::now();
  HOLOSCAN_LOG_INFO("{} pulses/sec", static_cast<uint64_t>(
      static_cast<double>(num_iterations * app->GetPulses() * app->GetChannels()) /
    std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() * 1000.f));

  return 0;
}
