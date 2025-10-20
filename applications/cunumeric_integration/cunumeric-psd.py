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

import cunumeric as np
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec


class SourceOp(Operator):
    def __init__(self, *args, **kwargs):
        self.fs = 10e3  # sample rate
        self.N = 1e7  # number of samples
        self.t = np.arange(self.N) / float(self.fs)  # array of segment times
        self.signal = np.sin(2 * np.pi * self.t)

        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("signal_out")

    def compute(self, op_input, op_output, context):
        op_output.emit(self.signal, "signal_out")


class PSDOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("signal_in")
        spec.output("psd_out")

    def compute(self, op_input, op_output, context):
        sig = op_input.receive("signal_in")
        op_output.emit(np.abs(np.fft.fft(sig)), "psd_out")


class SinkOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("psd_in")

    def compute(self, op_input, op_output, context):
        sig = op_input.receive("psd_in")
        print(sig[0:5])


class PSDApp(Application):
    def compose(self):
        src = SourceOp(self, CountCondition(self, 10), name="src_op")
        psd = PSDOp(self, name="psd_op")
        sink = SinkOp(self, name="sink_op")

        # Connect the operators into the workflow:  src -> psd -> sink
        self.add_flow(src, psd)
        self.add_flow(psd, sink)


def main():
    app = PSDApp()
    app.config("")
    app.run()


if __name__ == "__main__":
    main()
