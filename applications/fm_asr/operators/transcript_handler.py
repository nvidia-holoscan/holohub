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

import os
import threading
import time
from collections import deque

import nvtx


class TranscriptHandler(threading.Thread):
    def __init__(self, buffer, output_file) -> None:
        threading.Thread.__init__(self)
        self.buffer: deque = buffer

        self.output_file = output_file
        self._kill = False
        self.transcript = ""
        print("Transcript will be saved in", self.output_file)
        if os.path.exists(self.output_file):  # if file exists, clear it.
            f = open(self.output_file, "w")
            f.write("")
            f.close()

    def run(self):
        while not self._kill:
            if len(self.buffer) > 0:
                self._write_to_file(self.buffer.popleft())
            else:
                time.sleep(0.1)

    def kill(self):
        self._kill = True

    @nvtx.annotate("transcript_handler_write_file", color="magenta")
    def _write_to_file(self, inp_tuple):
        responses, buff_cleared = inp_tuple
        if responses is not None:
            with open(self.output_file, "a") as f:
                try:
                    for response in responses:
                        if not response.results:
                            continue
                        partial_transcript = ""
                        for result in response.results:
                            if result.is_final:
                                for index, alternative in enumerate(result.alternatives):
                                    print(alternative.transcript, end="\r")
                                    self.transcript = alternative.transcript
                            else:
                                transcript = result.alternatives[0].transcript
                                partial_transcript += transcript
                except Exception as e:
                    print("Exception:", e)
        if buff_cleared:
            print("\n")
            with open(self.output_file, "a") as f:
                f.write("Transcript: %s\n" % (self.transcript))
                self.transcript = ""
