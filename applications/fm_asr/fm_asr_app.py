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
import sys
import time
from collections import deque

from holoscan.conditions import CountCondition
from holoscan.core import Application
from operators.audio_handler import AudioHandler
from operators.demodulate_op import DemodulateOp
from operators.pcm_op import PCMOp
from operators.play_audio_op import PlayAudioOp
from operators.realtime_asr_op import RealtimeAsrOp
from operators.resample_op import ResampleOp
from operators.rtl_sdr_generator_op import RtlSdrGeneratorOp
from operators.transcript_handler import TranscriptHandler
from operators.transcript_sink_op import TranscriptSinkOp
from utils import load_params, run_time_to_iterations

params = load_params(*sys.argv[1:])

# Shared buffers for multithreaded handlers
transcript_buffer = deque([])
audio_buffer = deque([])


class FmAsr(Application):
    """
    Pipeline architecture:
    SignalGenerator -> Demodulation -> Resampling -> PCM -> ASR -> Sink -> End
                                    |
                                    -> PlayAudio -> End
    """

    def __init__(self):
        super().__init__()

    def compose(self):
        run_time = params["run_time"]
        buffer_size = 2**17

        fs = params["RtlSdrGeneratorOp"]["sample_rate"]
        num_iterations = run_time_to_iterations(run_time, fs, buffer_size)
        src = RtlSdrGeneratorOp(
            self,
            CountCondition(self, num_iterations),
            name="src",
            params=params,
            buffer_size=buffer_size,
        )

        demodulate = DemodulateOp(self, name="demodulate")
        resample = ResampleOp(self, name="resample", params=params)
        if params["PlayAudioOp"]["play_audio"]:
            audio = PlayAudioOp(self, name="audio", params=params, buffer=audio_buffer)
        pcm = PCMOp(self, name="pcm")
        asr = RealtimeAsrOp(self, name="asr_op", params=params)
        sink = TranscriptSinkOp(self, name="sink", buffer=transcript_buffer)

        self.add_flow(src, demodulate)
        self.add_flow(demodulate, resample)
        if params["PlayAudioOp"]["play_audio"]:
            self.add_flow(demodulate, audio)
        self.add_flow(resample, pcm)
        self.add_flow(pcm, asr)
        self.add_flow(asr, sink)


def main():
    app = FmAsr()
    app.config("")

    # Kick off handler threads
    t_handler = TranscriptHandler(transcript_buffer, params["TranscriptSinkOp"]["output_file"])
    t_handler.start()
    if params["PlayAudioOp"]["play_audio"]:
        a_handler = AudioHandler(audio_buffer)
        a_handler.start()

    # Launch main application
    tstart = time.time()
    app.run()
    duration = time.time() - tstart
    print(f"{duration:0.3f}")

    # Clean up handlers
    t_handler.kill()
    t_handler.join()
    if params["PlayAudioOp"]["play_audio"]:
        a_handler.kill()
        a_handler.join()
        a_handler.clean_up()


if __name__ == "__main__":
    main()
