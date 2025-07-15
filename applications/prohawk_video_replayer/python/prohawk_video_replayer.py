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

import argparse
import os

from holoscan.core import Application
from holoscan.operators import HolovizOp, VideoStreamReplayerOp

from holohub.prohawk_video_processing import ProhawkOp


class ProhawkVideoProcessingApp(Application):
    def __init__(self, datapath):
        self.datapath = datapath
        super().__init__()

    def compose(self):
        replayer = VideoStreamReplayerOp(
            self, name="replayer", directory=self.datapath, **self.kwargs("replayer")
        )
        prohawk_op = ProhawkOp(self, name="input")
        visualizer = HolovizOp(self, name="holoviz", **self.kwargs("holoviz"))

        self.add_flow(replayer, prohawk_op, {("output", "input")})
        self.add_flow(prohawk_op, visualizer, {("output1", "receivers")})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prohawk Video Processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/workspace/holohub/data/endoscopy",
        help="The path to the data to be used",
    )
    parser.add_argument("--config", type=str, help="The path to the Yaml config")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    app = ProhawkVideoProcessingApp(args.data)

    if args.config:
        app.config(args.config)
    else:
        config_path = os.path.join(os.path.dirname(__file__), "prohawk_video_replayer.yaml")
        app.config(config_path)

    app.run()


if __name__ == "__main__":
    main()
