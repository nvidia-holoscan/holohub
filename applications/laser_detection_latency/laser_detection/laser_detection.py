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
from enum import Enum

import cupy as cp
import cvcuda
import holoscan
import numpy as np
import nvcv
from holoscan.core import Application, ConditionType
from holoscan.operators import BayerDemosaicOp, FormatConverterOp, HolovizOp
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType
from holoscan.schedulers import MultiThreadScheduler
from skimage.io import imread


class CameraSource(Enum):
    EVT = 1
    USB = 2


class CalCoordsOperator(holoscan.core.Operator):
    def __init__(
        self, *args, width=1920, height=1080, threshold=24, mode=CameraSource.USB, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.travel_x = 0
        self.travel_y = 0
        self.width = width
        self.height = height
        self.threshold = threshold
        self.mode = mode

    def setup(self, spec):
        logging.info("setup")
        spec.input("input").condition(ConditionType.NONE)
        spec.output("output")

    def start(self):
        self.travel_x = 0
        self.travel_y = 0

        # CPU
        self.travel = np.array([0, 0])
        self.detected_circles = self.previous_point = np.array([0, 0])
        self.constant_zero = np.array([0, 0])

        self.frameCount = 0

        current_path = os.getcwd()
        if self.mode == CameraSource.EVT:
            target_path = "evt_cam_calibration/applications/laser_detection_app/evt_cam_calibration"
            calibration_fn = "evt-cali.npy"
            self.dead_zone = 1
        elif self.mode == CameraSource.USB:
            target_path = "usb_cam_calibration/applications/laser_detection_app/usb_cam_calibration"
            calibration_fn = "usb-cali.npy"
            self.dead_zone = 4
        else:
            target_path = " "
            calibration_fn = ""
            self.dead_zone = 1

        target_directory = "build"
        path_parts = current_path.split(os.sep)
        if target_directory in path_parts:
            index = path_parts.index(target_directory)
            truncated_path = os.sep.join(path_parts[: index + 1])
        else:
            truncated_path = current_path
        new_path = os.path.join(truncated_path, target_path, calibration_fn)
        self.matrix = np.load(new_path)

        self.persp_border_value = np.array([0]).astype(np.float32)
        self.stream = cvcuda.Stream()
        with self.stream:
            self.thresh_val = cvcuda.as_tensor(
                cp.asarray(np.array([255 * (self.threshold / 100)], dtype=np.float64), cp.float64),
                "N",
            )
            self.max_val = cvcuda.as_tensor(cp.asarray(np.array([255]), cp.float64), "N")

    def stop(self):
        pass

    def detect_laser_cvcuda_cupy(self, input_tensor):
        cp_frame = cp.asarray(input_tensor, cp.uint8)
        gpu_tensor = cvcuda.as_tensor(cp_frame, "HWC")

        # RGB(A) to Gray
        if self.mode == CameraSource.EVT:
            gray = cvcuda.cvtcolor(gpu_tensor, cvcuda.ColorConversion.RGB2GRAY)
        elif self.mode == CameraSource.USB:
            rgb = cvcuda.cvtcolor(gpu_tensor, cvcuda.ColorConversion.RGBA2RGB)
            gray = cvcuda.cvtcolor(rgb, cvcuda.ColorConversion.RGB2GRAY)

        # threshold
        threshold = cvcuda.threshold(
            gray, self.thresh_val, self.max_val, cvcuda.ThresholdType.TOZERO
        )

        # warp perspective
        persp = cvcuda.warp_perspective(
            threshold,
            self.matrix,
            cvcuda.Interp.LINEAR,
            border_mode=cvcuda.Border.CONSTANT,
            border_value=self.persp_border_value,
        )

        # Find the min/max locs
        outs = cvcuda.max_loc(persp, 1)
        max_val, max_loc, num_max = outs

        # CPU code
        max_val = cp.asnumpy(cp.asarray(max_val.cuda(), cp.float32))
        max_loc = cp.asnumpy(cp.asarray(max_loc.cuda(), cp.int32))

        # CPU code
        val = max_val[0][0]
        pt = max_loc[0][0]
        pt = np.array([pt[0], pt[1]])
        if val == 0:
            pt = self.previous_point

        if not np.array_equal(pt, self.constant_zero):
            self.detected_circles = pt
        if np.linalg.norm(self.detected_circles - self.previous_point) < self.dead_zone:
            self.detected_circles = self.previous_point
        if np.linalg.norm(self.detected_circles - self.travel) > self.dead_zone:
            curr_coord = self.detected_circles
            self.previous_point = self.detected_circles
            self.travel = curr_coord
        else:
            curr_coord = self.travel
        return curr_coord

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("input")

        if in_message is None:
            # do nothing
            self.travel_x = self.travel_x
            self.travel_y = self.travel_y
        elif in_message is not None:
            if self.mode == CameraSource.USB:
                input_tensor = in_message.get("preprocessed")
            else:
                input_tensor = in_message.get("")

            with self.stream:
                curr_coord = self.detect_laser_cvcuda_cupy(input_tensor)
            self.travel_x = curr_coord[0]
            self.travel_y = curr_coord[1]

        # Send out_message
        if self.mode == CameraSource.EVT:
            out_message = {"coords_1": (self.travel_x / self.width, self.travel_y / self.height)}
        elif self.mode == CameraSource.USB:
            out_message = {"coords_2": (self.travel_x / self.width, self.travel_y / self.height)}
        op_output.emit(out_message, "output")


class AddViewOperator(holoscan.core.Operator):
    def __init__(self, *args, display_width=1920, display_height=1080, **kwargs):
        super().__init__(*args, **kwargs)
        self.display_width = 1920
        self.display_height = 1080

    def setup(self, spec):
        logging.info("setup")
        spec.input("input1")
        spec.input("input2")
        spec.output("outputs")
        spec.output("output_specs")

    def start(self):
        image = imread("left-asset.png")
        self.left = cp.asarray(image)
        self.left_h, self.left_w, c = self.left.shape

        image = imread("right-asset.png")
        self.right = cp.asarray(image)
        self.right_h, self.right_w, c = self.right.shape
        number_of_components = 3
        self.background = cp.zeros(
            (self.display_height, self.display_width, number_of_components), cp.uint8
        )

    def stop(self):
        pass

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message_1 = op_input.receive("input1")
        X1, Y1 = in_message_1.get("coords_1")

        in_message_2 = op_input.receive("input2")
        X2, Y2 = in_message_2.get("coords_2")

        out_message = {
            "video": self.background,
            "image": self.left,
            "image_2": self.right,
        }
        op_output.emit(out_message, "outputs")

        #
        specs = []
        #
        spec = HolovizOp.InputSpec("video", HolovizOp.InputType.COLOR)
        view = HolovizOp.InputSpec.View()
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.width = 1.0
        view.height = 1.0
        spec.views = [view]
        specs.append(spec)

        #
        spec = HolovizOp.InputSpec("image", HolovizOp.InputType.COLOR)
        view = HolovizOp.InputSpec.View()
        view.offset_x = X2 - (self.left_w * 0.4) / self.display_width
        view.offset_y = Y2 - (self.left_h * 0.5) / self.display_height
        view.width = self.left_w / self.display_width
        view.height = self.left_h / self.display_height
        spec.views = [view]
        specs.append(spec)

        spec = HolovizOp.InputSpec("image_2", HolovizOp.InputType.COLOR)
        view = HolovizOp.InputSpec.View()
        view.offset_x = X1 - (0.4 * self.right_w) / self.display_width
        view.offset_y = Y1 - (0.5 * self.right_h) / self.display_height
        view.width = self.right_w / self.display_width
        view.height = self.right_h / self.display_height
        spec.views = [view]
        specs.append(spec)

        # emit the output specs
        op_output.emit(specs, "output_specs")


class LaserDetectionApp(Application):
    def __init__(self, threshold_usb, threshold_evt):
        super().__init__()

        # set name
        self.name = "Laser Detection App"
        self._threshold_usb = threshold_usb
        self._threshold_evt = threshold_evt

    def compose(self):
        try:
            from holohub.emergent_source import EmergentSourceOp
        except ImportError:
            raise ImportError(
                "Could not import EmergentSourceOp. This application requires that the library "
                "was built with Emergent SDK support."
            )
        # EVT source
        e_source = EmergentSourceOp(self, name="emergent", **self.kwargs("emergent"))
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
            num_blocks=3,
        )
        bayer_demosaic = BayerDemosaicOp(
            self,
            name="bayer_demosaic",
            pool=pool,
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("demosaic"),
        )
        cal_evt_coords = CalCoordsOperator(
            self,
            name="cal_evt_coords",
            threshold=self._threshold_evt,
            mode=CameraSource.EVT,
            **self.kwargs("cal_evt_coords"),
        )

        # USB pipeline
        u_source_args = self.kwargs("usb_source")
        if "width" in u_source_args and "height" in u_source_args:
            # width and height given, use BlockMemoryPool (better latency)
            width = u_source_args["width"]
            height = u_source_args["height"]
        else:
            width = 1920
            height = 1080

        n_channels = 4
        block_size = width * height * n_channels
        allocator = holoscan.resources.BlockMemoryPool(
            self, name="pool", storage_type=0, block_size=block_size, num_blocks=5
        )
        allocator2 = holoscan.resources.BlockMemoryPool(
            self, name="pool", storage_type=1, block_size=block_size, num_blocks=4
        )

        u_source = holoscan.operators.V4L2VideoCaptureOp(
            self,
            name="u_source",
            allocator=allocator,
            **u_source_args,
        )
        preprocessor = FormatConverterOp(
            self,
            name="preprocessor",
            out_tensor_name="preprocessed",
            out_dtype="uint8",
            pool=allocator2,
            # **preprocessor_args,
        )
        cal_usb_coords = CalCoordsOperator(
            self,
            name="cal_usb_coords",
            threshold=self._threshold_usb,
            mode=CameraSource.USB,
            **self.kwargs("cal_usb_coords"),
        )

        # common pipeline
        view_operator = AddViewOperator(self, name="add_view", **self.kwargs("add_view"))
        visualizer = HolovizOp(
            self,
            name="holoviz",
            **self.kwargs("holoviz"),
        )

        # EVT camera pipeline
        self.add_flow(e_source, bayer_demosaic, {("signal", "receiver")})
        self.add_flow(bayer_demosaic, cal_evt_coords, {("transmitter", "input")})
        self.add_flow(cal_evt_coords, view_operator, {("output", "input1")})

        # USB camera pipeline
        self.add_flow(u_source, preprocessor)
        self.add_flow(preprocessor, cal_usb_coords, {("", "input")})
        self.add_flow(cal_usb_coords, view_operator, {("output", "input2")})

        # viz
        self.add_flow(view_operator, visualizer, {("outputs", "receivers")})
        self.add_flow(view_operator, visualizer, {("output_specs", "input_specs")})


def main():
    parser = ArgumentParser(description="EVT and USB cameras detecting laser pointer")
    parser.add_argument(
        "--threshold-usb",
        type=int,
        default=24,
        help="Threshold percentage applied to usb cam",
    )
    parser.add_argument(
        "--threshold-evt",
        type=int,
        default=60,
        help="Threshold percentage applied to EVT cam",
    )
    args = parser.parse_args()

    config_file = os.path.join(os.path.dirname(__file__), "laser_detection.yaml")

    app = LaserDetectionApp(
        args.threshold_usb,
        args.threshold_evt,
    )
    app.config(config_file)
    # multithreaded scheduler
    scheduler = MultiThreadScheduler(
        app,
        worker_thread_number=5,
        check_recession_period_ms=0.0,
        stop_on_deadlock=True,
        stop_on_deadlock_timeout=500,
        name="multithread_scheduler",
    )
    app.scheduler(scheduler)
    app.run()
    nvcv.clear_cache()


if __name__ == "__main__":
    main()
