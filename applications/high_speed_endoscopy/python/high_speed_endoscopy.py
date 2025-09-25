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

import os
from argparse import ArgumentParser

from holoscan.core import Application
from holoscan.operators import BayerDemosaicOp, HolovizOp
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType


class HighSpeedEndoscopyApp(Application):
    def __init__(self):
        super().__init__()

        # set name
        self.name = "High speed endoscopy app"

    def compose(self):
        try:
            from holohub.emergent_source import EmergentSourceOp
        except ImportError:
            raise ImportError(
                "Could not import EmergentSourceOp. This application requires that the library "
                "was built with Emergent SDK support."
            )

        source = EmergentSourceOp(self, name="emergent", **self.kwargs("emergent"))

        cuda_stream_pool = CudaStreamPool(
            self,
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        pool = BlockMemoryPool(
            self,
            name="pool",
            storage_type=MemoryStorageType.DEVICE,
            block_size=72576000,
            num_blocks=2,
        )

        bayer_demosaic = BayerDemosaicOp(
            self,
            name="bayer_demosaic",
            pool=pool,
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("demosaic"),
        )

        viz = HolovizOp(self, name="holoviz", **self.kwargs("holoviz"))

        self.add_flow(source, bayer_demosaic, {("signal", "receiver")})
        self.add_flow(bayer_demosaic, viz, {("transmitter", "receivers")})


def main():
    parser = ArgumentParser(description="High-speed endoscopy demo application.")
    parser.parse_args()

    config_file = os.path.join(os.path.dirname(__file__), "high_speed_endoscopy.yaml")

    app = HighSpeedEndoscopyApp()
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    main()
