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


class ResampleOp(Operator):
    def __init__(self, *args, **kwargs):
        global_params = kwargs["params"]
        src_fs = global_params["RtlSdrGeneratorOp"]["sample_rate"]
        self.src_fs = int(src_fs)
        self.riva_sample_rate = int(16e3)

        # prime/compile the kernel
        t_sig = cp.ones([int(1024 * 250e3 // 16e3)], dtype=cp.float32)
        cusignal.resample_poly(t_sig, 1, self.src_fs // self.riva_sample_rate, window="hamm")

        # Need to call the base class constructor last
        del kwargs["params"]
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("rx_sig")
        spec.output("sig_out")

    @nvtx.annotate("resample_compute", color="red")
    def compute(self, op_input, op_output, context):
        sig = op_input.receive("rx_sig")
        op_output.emit(
            cusignal.resample_poly(
                sig, 1, self.src_fs // self.riva_sample_rate, window="hamm"
            ).astype(cp.float32),
            "sig_out",
        )
