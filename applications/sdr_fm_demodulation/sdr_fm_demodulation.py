# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import queue
import time

try:
    import cupy as cp
except ImportError:
    raise ImportError("This demo requires cupy, but it could not be imported.")

try:
    import cusignal
except ImportError:
    raise ImportError("This demo requires cusignal, but it could not be imported.")

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX
except ImportError:
    raise ImportError("This demo requires SoapySDR, but it could not be imported.")

try:
    import pyaudio
except ImportError:
    raise ImportError("This demo requires PyAudio, but it could not be imported.")

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

# Demodulation and Radio Settings
fm_freq = 88500000
sdr_fs = int(250e3)
audio_fs = int(48e3)
buffer_size = 1024 * (sdr_fs // audio_fs)

try:
    args = dict(driver="rtlsdr")
except ImportError:
    raise ImportError("Ensure SDR is connected and appropriate drivers are installed.")

# SoapySDR Config
sdr = SoapySDR.Device(args)
sdr.setSampleRate(SOAPY_SDR_RX, 0, sdr_fs)
sdr.setFrequency(SOAPY_SDR_RX, 0, fm_freq)
rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)

# Start streams and allocate buffers
buffer = cusignal.get_shared_mem(buffer_size, dtype=cp.complex64)
p = pyaudio.PyAudio()
sdr.activateStream(rx)
que = queue.Queue()


# PyAudio Config
def pyaudio_callback(in_data, frame_count, time_info, status):
    print("in pyaudio callback")
    return (que.get(), pyaudio.paContinue)


stream = p.open(
    format=pyaudio.paFloat32, channels=1, rate=48000, output=True, stream_callback=pyaudio_callback
)

stream.start_stream()


class SignalGeneratorOp(Operator):
    def __init__(self, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("rx_sig")

    def compute(self, op_input, op_output, context):
        # sig = cp.random.randn(5120) + 1j*cp.random.randn(5120)
        sdr.readStream(rx, [buffer], buffer_size, timeoutUs=int(8e12))
        op_output.emit(cp.asarray(buffer.astype(cp.complex64)), "rx_sig")


class DemodulateOp(Operator):
    def __init__(self, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("rx_sig")
        spec.output("sig_out")

    def compute(self, op_input, op_output, context):
        sig = op_input.receive("rx_sig")
        print("In Demodulate - Got: ", sig[0:10])
        op_output.emit(cusignal.fm_demod(sig), "sig_out")


class ResampleOp(Operator):
    def __init__(self, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("rx_sig")
        spec.output("sig_out")

    def compute(self, op_input, op_output, context):
        sig = op_input.receive("rx_sig")
        print("In Resample - Got: ", sig[0:10])
        op_output.emit(
            cusignal.resample_poly(sig, 1, sdr_fs // audio_fs, window="hamm").astype(cp.float32),
            "sig_out",
        )


class SDRSinkOp(Operator):
    def __init__(self, *args, shape=(512, 512), **kwargs):
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("rx_sig")

    def compute(self, op_input, op_output, context):
        sig = op_input.receive("rx_sig")
        que.put(cp.asnumpy(sig))
        print("In SDR Sink - Got: ", sig[0:10])


class FMDemod(Application):
    def __init__(self):
        super().__init__()

    def compose(self):
        src = SignalGeneratorOp(self, CountCondition(self, 500), name="src")
        demodulate = DemodulateOp(self, name="demodulate")
        resample = ResampleOp(self, name="resample")
        sink = SDRSinkOp(self, name="sink")

        self.add_flow(src, demodulate)
        self.add_flow(demodulate, resample)
        self.add_flow(resample, sink)


if __name__ == "__main__":
    app = FMDemod()
    app.config("")

    tstart = time.time()
    app.run()
    duration = time.time() - tstart

    print(f"{duration:0.3f}")
