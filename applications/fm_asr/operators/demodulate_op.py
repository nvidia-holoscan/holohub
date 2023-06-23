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
import cusignal
import nvtx
from holoscan.core import Operator, OperatorSpec


class DemodulateOp(Operator):
    def __init__(self, *args, **kwargs):
        # opt tests
        self.buffer_size = int(1024 * (250e3 // 16e3))
        init_sig = cp.ones(self.buffer_size, dtype=cp.complex64)
        cusignal.fm_demod(init_sig)  # prime/compile the demod kernel

        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("rx_sig")
        spec.output("sig_out")

    @nvtx.annotate("demodulate_compute", color="blue")
    def compute(self, op_input, op_output, context):
        sig = op_input.receive("rx_sig")
        op_output.emit(cusignal.fm_demod(sig), "sig_out")
