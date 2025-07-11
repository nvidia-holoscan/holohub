# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections import deque

import cupy as cp
import cusignal
from holoscan.core import Operator, OperatorSpec

pyaudio_available = False
p_audio = None
try:
    import pyaudio

    pyaudio_available = True
    p_audio = pyaudio.PyAudio()
except ImportError:
    print("PyAudio not available. Continuing")


class PlayAudioOp(Operator):
    """
    Operator to play back the incoming broadcast.
    Sound card limitations, w.r.t. sample rate, on some systems may require this separate
    operator/resampling step.
    """

    def __init__(self, *args, **kwargs):
        global_params = kwargs["params"]
        local_params = global_params[self.__class__.__name__]
        self.soundcard_fs = int(48e3)

        src_fs = global_params["RtlSdrGeneratorOp"]["sample_rate"]
        self.src_sample_rate = int(src_fs)
        self.play_audio = local_params["play_audio"]
        self.audio_buff: deque = kwargs["buffer"]
        del kwargs["params"]
        del kwargs["buffer"]
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("rx_sig")

    def compute(self, op_input, op_output, context):
        sig = op_input.receive("rx_sig")
        if self.play_audio and pyaudio_available:
            resamp_sig = (
                cusignal.resample_poly(
                    sig, 1, self.src_sample_rate // self.soundcard_fs, window="hamm"
                ).astype(cp.float32)
                / 8
            )  # Division by a constant controls output volume
            cpu_sig = cp.asnumpy(resamp_sig).tobytes()
            self.audio_buff.append(cpu_sig)
