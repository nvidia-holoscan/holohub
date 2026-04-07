# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time

try:
    import numpy as np
except ImportError:
    raise ImportError("This demo requires numpy, but it could not be imported.")

try:
    import cupy as cp
except ImportError:
    raise ImportError("This demo requires cupy, but it could not be imported.")

import cupyx.scipy.signal as cusignal
from cupy.cuda.cufft import CUFFT_C2C, CUFFT_Z2Z, Plan1d
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

# Radar Settings
num_channels = 16
num_pulses = 128
num_uncompressed_range_bins = 9000
waveform_length = 1000
num_compressed_range_bins = num_uncompressed_range_bins - waveform_length + 1
NDfft = 256
Pfa = 1e-5
iterations = 100

# Window Settings
window = cusignal.windows.hamming(waveform_length)
# The -2 is a hack here to account for a 3 tap MTI filter
range_doppler_window = cp.transpose(
    cp.repeat(
        cp.expand_dims(cusignal.windows.hamming(num_pulses - 2), 0),
        num_compressed_range_bins,
        axis=0,
    )
)
Nfft = 2 ** math.ceil(math.log2(np.max([num_uncompressed_range_bins, waveform_length])))

# CFAR Settings
mask = cp.transpose(
    cp.asarray(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
)


class SignalGeneratorOp(Operator):
    def __init__(self, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("x")
        spec.output("waveform")

    def compute(self, op_input, op_output, context):
        x = cp.random.randn(
            num_pulses, num_uncompressed_range_bins, dtype=cp.float32
        ) + 1j * cp.random.randn(num_pulses, num_uncompressed_range_bins, dtype=cp.float32)
        waveform = cp.random.randn(waveform_length, dtype=cp.float32) + 1j * cp.random.randn(
            waveform_length, dtype=cp.float32
        )

        op_output.emit(x, "x")
        op_output.emit(waveform, "waveform")


def _cufft_type(dtype):
    return CUFFT_C2C if dtype == cp.complex64 else CUFFT_Z2Z


class PulseCompressionOp(Operator):
    def __init__(self, *args, **kwargs):
        # Need to call the base class constructor last
        self._plan_1d = None
        self._plan_batched = None
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("x")
        spec.input("waveform")
        spec.output("X")

    def compute(self, op_input, op_output, context):
        x = op_input.receive("x")
        waveform = op_input.receive("waveform")

        waveform_windowed = waveform * window
        waveform_windowed_norm = (waveform_windowed / cp.linalg.norm(waveform_windowed)).astype(
            x.dtype
        )

        if self._plan_1d is None:
            self._plan_1d = Plan1d(Nfft, _cufft_type(x.dtype), 1)
            self._plan_batched = Plan1d(Nfft, _cufft_type(x.dtype), num_pulses)

        with self._plan_1d:
            W = cp.conj(cp.fft.fft(waveform_windowed_norm, Nfft))
        with self._plan_batched:
            X = cp.fft.fft(x, Nfft, 1)
            Y = cp.fft.ifft(cp.multiply(X, W), Nfft, 1)

        x_compressed = Y[:, 0:num_compressed_range_bins]

        x_compressed_stack = cp.stack([x_compressed] * num_channels)

        op_output.emit(x_compressed_stack, "X")


class MTIFilterOp(Operator):
    def __init__(self, *args, **kwargs):
        # Need to call the base class constructor last
        self.index = 0
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("x")
        spec.output("X")

    def compute(self, op_input, op_output, context):
        x = op_input.receive("x")
        for channel in range(num_channels):
            x_conv2 = cusignal.convolve2d(
                x[channel, :, :], cp.array([[1], [-2], [-1]], dtype=cp.complex64), mode="valid"
            )
            x_conv2_stack = cp.stack([x_conv2] * num_channels)

        op_output.emit(x_conv2_stack, "X")


class RangeDopplerOp(Operator):
    def __init__(self, *args, **kwargs):
        # Need to call the base class constructor last
        self._plan_rd = None
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("x")
        spec.output("X")

    def compute(self, op_input, op_output, context):
        x = op_input.receive("x")
        for channel in range(num_channels):
            inp = cp.multiply(x[channel, :, :], range_doppler_window)
            if self._plan_rd is None:
                # inp shape: (num_pulses-2, num_compressed_range_bins); fft along axis 0
                self._plan_rd = Plan1d(NDfft, _cufft_type(inp.dtype), inp.shape[1])
            with self._plan_rd:
                x_window = cp.fft.fft(inp, NDfft, 0)
            x_window_stack = cp.stack([x_window] * num_channels)

        op_output.emit(x_window_stack, "X")


class CFAROp(Operator):
    def __init__(self, *args, **kwargs):
        # Need to call the base class constructor last
        self.index = 0
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("x")
        spec.output("X")

    def compute(self, op_input, op_output, context):
        x = op_input.receive("x")

        norm = cusignal.convolve2d(cp.ones(x.shape[1::]), mask, "same")

        for channel in range(num_channels):
            background_averages = cp.divide(
                cusignal.convolve2d(cp.abs(x[channel, :, :]) ** 2, mask, "same"), norm
            )
            alpha = cp.multiply(norm, cp.power(Pfa, cp.divide(-1.0, norm)) - 1)
            dets = cp.zeros(x[channel, :, :].shape)
            dets[cp.where(cp.abs(x[channel, :, :]) ** 2 > cp.multiply(alpha, background_averages))]

            dets_stacked = cp.stack([dets] * num_channels)

        op_output.emit(dets_stacked, "X")


class SinkOp(Operator):
    def __init__(self, *args, shape=(512, 512), **kwargs):
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("X")

    def compute(self, op_input, op_output, context):
        op_input.receive("X")


class BasicRadarFlow(Application):
    def __init__(self):
        super().__init__()

    def compose(self):
        src = SignalGeneratorOp(self, CountCondition(self, iterations), name="src")
        pulseCompression = PulseCompressionOp(self, name="pulse-compression")
        mtiFilter = MTIFilterOp(self, name="mti-filter")
        rangeDoppler = RangeDopplerOp(self, name="range-doppler")
        cfar = CFAROp(self, name="cfar")
        sink = SinkOp(self, name="sink")

        self.add_flow(src, pulseCompression, {("x", "x"), ("waveform", "waveform")})
        self.add_flow(pulseCompression, mtiFilter)
        self.add_flow(mtiFilter, rangeDoppler)
        self.add_flow(rangeDoppler, cfar)
        self.add_flow(cfar, sink)


def main():
    app = BasicRadarFlow()
    app.config("")

    tstart = time.time()
    app.run()
    tstop = time.time()

    duration = (iterations * num_pulses * num_channels) / (tstop - tstart)

    print(f"{duration:0.3f} pulses/sec")


if __name__ == "__main__":
    main()
