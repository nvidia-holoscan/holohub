# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from copy import deepcopy

import numpy as np
import nvtx
import riva.client
import riva.client.proto.riva_asr_pb2 as rasr
from holoscan.core import Operator, OperatorSpec


class RealtimeAsrOp(Operator):
    """
    Execute Riva ASR requests.
    """

    def __init__(self, *args, **kwargs):
        global_params = kwargs["params"]
        riva_params = global_params["RivaAsrOp"]
        self.riva_recog_interval = riva_params["recognize_interval"]
        self.sample_rate = 16e3

        auth = riva.client.Auth(uri=riva_params["uri"])
        self.asr_service = riva.client.ASRService(auth)

        self.rasr_config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=riva_params["sample_rate"],
            language_code=riva_params["language_code"],
            max_alternatives=riva_params["max_alternatives"],
            enable_automatic_punctuation=riva_params["automatic_punctuation"],
            enable_word_time_offsets=riva_params["word_time_offsets"],
            verbatim_transcripts=not riva_params["no_verbatim_transcripts"],
        )
        self.streaming_rasr_config = riva.client.StreamingRecognitionConfig(
            config=deepcopy(self.rasr_config), interim_results=riva_params["interim_transcriptions"]
        )

        # variables to control frequency of gRPC calls
        self.batch = deque([])
        self.batch_len = 0
        self.time_inc = 0.1
        self.start_t = 0
        self.curr_t = 0

        del kwargs["params"]
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("rx_sig")
        spec.output("asr_responses")

    @nvtx.annotate("asr_compute", color="green")
    def compute(self, op_input, op_output, context):
        signal = op_input.receive("rx_sig")
        rasr_req = rasr.StreamingRecognizeRequest(audio_content=signal)
        self.batch.append(rasr_req)
        self.batch_len += len(self.batch[-1].audio_content)

        asr_responses = None
        self.curr_t = self._get_audio_length()
        buff_cleared = False
        if (
            self.curr_t - self.start_t
        ) > self.time_inc:  # need to prevent huge amount of RPC requests
            self.start_t = self.curr_t
            with nvtx.annotate("asr_rpc", color="black"):
                # asynchronous call. latency not realized unless access is attempted.
                asr_responses = self.asr_service.stub.StreamingRecognize(
                    self._req_generator(), metadata=self.asr_service.auth.get_auth_metadata()
                )
        if self._get_audio_length() >= self.riva_recog_interval:
            self.batch.clear()
            self.batch_len = 0
            self.start_t = 0
            buff_cleared = True
        out = (asr_responses, buff_cleared)
        op_output.emit(out, "asr_responses")

    def _req_generator(self):
        yield rasr.StreamingRecognizeRequest(streaming_config=self.streaming_rasr_config)
        for b in self.batch:
            yield b

    def _get_audio_length(self):
        pcm_dtype = np.int16
        pcm_bytes = np.dtype(pcm_dtype).itemsize
        num_samples = self.batch_len / pcm_bytes
        audio_time = num_samples / self.sample_rate
        return audio_time
