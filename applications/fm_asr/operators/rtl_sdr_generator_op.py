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

import cupy as cp
import cusignal
import nvtx
import SoapySDR
from holoscan.core import Operator, OperatorSpec
from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX


class RtlSdrGeneratorOp(Operator):
    def __init__(self, *args, **kwargs):
        global_params = kwargs["params"]
        local_params = global_params[self.__class__.__name__]
        self.fm_freq = local_params["tune_frequency"]
        self.sdr_fs = int(local_params["sample_rate"])
        self.riva_sample_rate = global_params["RivaAsrOp"]["sample_rate"]
        self.buffer_size = kwargs["buffer_size"]
        # Start streams and allocate buffers
        self.buffer = cusignal.get_shared_mem(self.buffer_size, dtype=cp.complex64)

        sdr_args = dict(driver="rtlsdr")
        self.sdr = SoapySDR.Device(sdr_args)
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sdr_fs)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.fm_freq)
        self.sdr.setGain(SOAPY_SDR_RX, 0, local_params["gain"])
        self.rx = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rx)
        del kwargs["params"]
        del kwargs["buffer_size"]
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("rx_sig")

    @nvtx.annotate("src_compute", color="purple")
    def compute(self, op_input, op_output, context):
        self.sdr.readStream(self.rx, [self.buffer], self.buffer_size, timeoutUs=int(8e12))
        op_output.emit(cp.asarray(self.buffer.astype(cp.complex64)), "rx_sig")
