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

import cupy as cp
import nvtx
from holoscan.core import Operator, OperatorSpec
from utils import float_to_pcm


class PCMOp(Operator):
    """
    Converts signal from float to PCM16 format, and moves it to host.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("rx_sig")
        spec.output("sig_out")

    @nvtx.annotate("pcm_compute", color="orange")
    def compute(self, op_input, op_output, context):
        sig = op_input.receive("rx_sig")
        # print('PCM Block')
        pcm_sig = cp.asnumpy(float_to_pcm(sig, cp.int16)).tobytes()
        op_output.emit(pcm_sig, "sig_out")
