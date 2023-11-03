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
#pragma once

#include "common.h"

// ---------- Structures ----------
struct PulseCompressionData {
  PulseCompressionData(tensor_t<complex_t, 1> *_waveformView,
                       tensor_t<complex_t, 3> *_inputView,
                       cudaStream_t _stream)
    : waveformView(_waveformView), inputView(_inputView), stream(_stream)  {}
  tensor_t<complex_t, 1> *waveformView;
  tensor_t<complex_t, 3> *inputView;
  cudaStream_t stream;
};

struct ThreePulseCancellerData {
  ThreePulseCancellerData(tensor_t<complex_t, 3> _inputView,
                          cudaStream_t _stream)
    : inputView(_inputView), stream(_stream)  {}
  tensor_t<complex_t, 3> inputView;
  cudaStream_t stream;
};

struct DopplerData {
  DopplerData(tensor_t<complex_t, 3> _tpcView,
              tensor_t<float_t, 1> _cancelMask,
              cudaStream_t _stream)
    : tpcView(_tpcView), cancelMask(_cancelMask), stream(_stream)  {}
  tensor_t<complex_t, 3> tpcView;
  tensor_t<float_t, 1> cancelMask;
  cudaStream_t stream;
};

struct CFARData {
  CFARData(tensor_t<complex_t, 3> _tpcView,
           cudaStream_t _stream)
    : tpcView(_tpcView), stream(_stream)  {}
  tensor_t<complex_t, 3> tpcView;
  cudaStream_t stream;
};

// ---------- Operators ----------
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

  void setup(OperatorSpec& spec) override;
  void initialize() override;

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
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override;

 private:
  Parameter<int64_t> num_pulses;
  Parameter<int64_t> num_samples;
  Parameter<int64_t> waveform_length;
  Parameter<int64_t> num_channels;
  index_t num_samples_rnd;

  tensor_t<complex_t, 1> waveformView;
  tensor_t<complex_t, 0> norms;
  tensor_t<complex_t, 3> zeroPaddedInput;
};  // PulseCompressionOp

class ThreePulseCancellerOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ThreePulseCancellerOp)

  ThreePulseCancellerOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;

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
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override;

 private:
  Parameter<int64_t> num_pulses;
  Parameter<int64_t> num_samples;
  Parameter<int64_t> waveform_length;
  Parameter<int64_t> num_channels;
  index_t numCompressedSamples;
  index_t num_pulses_rnd;

  tensor_t<float_t, 1> cancelMask;
  tensor_t<complex_t, 3> tpcView;
};  // ThreePulseCancellerOp

class DopplerOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DopplerOp)

  DopplerOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;

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
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override;

 private:
  Parameter<int64_t> num_pulses;
  Parameter<int64_t> num_samples;
  Parameter<int64_t> waveform_length;
  Parameter<int64_t> num_channels;
  index_t numCompressedSamples;
};  // DopplerOp

class CFAROp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CFAROp)

  CFAROp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;

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
  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override;

 private:
  Parameter<uint16_t> num_transmits;
  Parameter<int64_t> num_pulses;
  Parameter<int64_t> num_samples;
  Parameter<int64_t> waveform_length;
  Parameter<int64_t> num_channels;
  index_t numCompressedSamples;
  index_t num_pulses_rnd;
  const index_t cfarMaskX = 13;
  const index_t cfarMaskY = 5;
  static const constexpr float pfa = 1e-5f;
  size_t transmits;

  tensor_t<float_t, 3> normT;
  tensor_t<float_t, 3> ba;
  tensor_t<int, 3> dets;
  tensor_t<float_t, 3> xPow;
  tensor_t<float_t, 2> cfarMaskView;
};  // CFAROp

}  // namespace holoscan::ops
