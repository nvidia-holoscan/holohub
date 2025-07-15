# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvtx
from holoscan.core import Operator, OperatorSpec


class TranscriptSinkOp(Operator):
    """
    Pipeline termination. Writes received responses to file
    """

    def __init__(self, *args, **kwargs):
        self.buffer: deque = kwargs["buffer"]
        kwargs["buffer"] = None

        del kwargs["buffer"]
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("asr_responses")

    @nvtx.annotate("sink_compute", color="yellow")
    def compute(self, op_input, op_output, context):
        responses = op_input.receive("asr_responses")
        if responses is not None:
            self.buffer.append(responses)
