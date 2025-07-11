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

import threading
import time
from collections import deque

import numpy as np
import nvtx
import pyaudio


class AudioHandler(threading.Thread):
    def __init__(self, buffer) -> None:
        threading.Thread.__init__(self)
        self.p_audio = pyaudio.PyAudio()
        self.buffer: deque = buffer
        self._kill = False
        self.audio_stream = self.p_audio.open(
            format=pyaudio.paFloat32, channels=1, rate=int(48e3), output=True
        )

    def run(self):
        while not self._kill:
            if len(self.buffer) > 0:
                self.play_audio(self.buffer.popleft())
            else:
                time.sleep(0.1)

    def kill(self):
        self._kill = True

    @nvtx.annotate("audio_handler_play", color="purple")
    def play_audio(self, clip: np.array):
        if clip is not None:
            self.audio_stream.write(clip)

    def clean_up(self):
        self.audio_stream.close()
        self.p_audio.terminate()
