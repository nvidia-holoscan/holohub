"""
SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

import os

from holoscan.conditions import CountCondition
from holoscan.core import Application
from holoscan.operators import HolovizOp

from holohub.aja_source import AJASourceOp


class AJACaptureApp(Application):
    """
    Example of an application that uses the following operators:

    - AJASourceOp
    - HolovizOp

    The AJASourceOp reads frames from an AJA input device and sends it to the HolovizOp.
    The HolovizOp displays the frames.
    """

    def compose(self):
        args_aja = self.kwargs("aja")

        count = args_aja["count"]
        args_aja.pop("count")

        source = AJASourceOp(self, CountCondition(self, count), name="aja", **args_aja)

        visualizer = HolovizOp(self, name="holoviz", **self.kwargs("holoviz"))

        self.add_flow(source, visualizer, {("video_buffer_output", "receivers")})


def main(config_file):
    app = AJACaptureApp()
    # if the --config command line argument was provided, it will override this config_file
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "aja_capture.yaml")
    main(config_file=config_file)
