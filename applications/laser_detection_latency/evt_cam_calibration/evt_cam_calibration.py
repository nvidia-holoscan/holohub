# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os
from argparse import ArgumentParser

import cv2
import holoscan
import numpy as np
from holoscan.conditions import CountCondition
from holoscan.core import Application
from holoscan.operators import BayerDemosaicOp, HolovizOp
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType
from skimage.io import imread

from holohub.apriltag_detector import ApriltagDetectorOp


def perspective_transform_evt(corners, width, height):
    top_l, top_r, bottom_r, bottom_l = corners
    dimensions = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32"
    )

    # Convert to Numpy format
    corners = np.array(corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(corners, dimensions)
    np.save("evt-cali.npy", matrix)
    print(f"matrix: {matrix}")

    # Return
    return


class AddBackgroundViewOperator(holoscan.core.Operator):
    def __init__(self, *args, width, height, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_count = 0
        self.is_done = 0
        self.width = width
        self.height = height

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")
        spec.output("outputs")
        spec.output("output_specs")

    def start(self):
        self.background = imread("apriltag-calibration.png")

    def stop(self):
        pass

    def compute(self, op_input, op_output, context):
        # Get input message
        input_corners = op_input.receive("input")
        corners_list = np.zeros((4, 2), dtype=np.float32)
        for tag in input_corners:
            corners_list[tag.id][0] = tag.corners[tag.id][0]
            corners_list[tag.id][1] = tag.corners[tag.id][1]

        if not np.all(corners_list == 0) and self.is_done != 1:
            # perspective_transform_evt will write the calibration file to disk.
            perspective_transform_evt(corners_list, self.width, self.height)
            self.is_done = 1

        self.frame_count += 1
        out_message = {
            "image": self.background,
        }

        op_output.emit(out_message, "outputs")

        specs = []
        spec = HolovizOp.InputSpec("image", HolovizOp.InputType.COLOR)
        view = HolovizOp.InputSpec.View()
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.width = 1.0
        view.height = 1.0
        spec.views = [view]
        specs.append(spec)
        op_output.emit(specs, "output_specs")


class EvtCamCalibrationApp(Application):
    def __init__(self):
        super().__init__()

        # set name
        self.name = "EVT camera calibration app"

    def compose(self):
        try:
            from holohub.emergent_source import EmergentSourceOp
        except ImportError:
            raise ImportError(
                "Could not import EmergentSourceOp. This application requires that the library "
                "was built with Emergent SDK support."
            )

        source = EmergentSourceOp(self, name="emergent", **self.kwargs("emergent"))
        source.add_arg(CountCondition(self, count=350))
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

        apriltag = ApriltagDetectorOp(
            self,
            name="apriltag",
            **self.kwargs("apriltag"),
        )

        back_view = AddBackgroundViewOperator(
            self,
            name="back_view",
            **self.kwargs("back_view"),
        )

        viz = HolovizOp(self, name="holoviz", **self.kwargs("holoviz"))

        self.add_flow(source, bayer_demosaic, {("signal", "receiver")})
        self.add_flow(bayer_demosaic, apriltag, {("transmitter", "input")})
        self.add_flow(apriltag, back_view, {("output", "input")})
        self.add_flow(back_view, viz, {("outputs", "receivers")})
        self.add_flow(back_view, viz, {("output_specs", "input_specs")})


def main():
    parser = ArgumentParser(description="Calibrating Emergent Camera.")
    parser.parse_args()

    config_file = os.path.join(os.path.dirname(__file__), "evt_cam_calibration.yaml")

    app = EvtCamCalibrationApp()
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    main()
