"""
SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""

import argparse
import os

import cupy as cp
import kvikio
import yaml
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec, Tracker

parser = argparse.ArgumentParser(
    prog="holoscan_gds.py",
    description="Example of GPU Direct Storage on IGX",
    epilog="See <Example on github> for more information",
)

parser.add_argument("--out_path", help="Path for file output", default="/mnt/nvme/data/test-file")
args = parser.parse_args()
out_path = args.out_path


# HDF5 reader operator
# currently just reads first 100 frames from hdf5 file
class ReaderOperator(Operator):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.data = cp.empty_like(cp.zeros((1, 576, 576)))
        self.file_path = self.kwargs.get("file_path", out_path)
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("frames_out")

    def compute(self, op_input, op_output, context):
        with kvikio.CuFile(self.file_path, "r") as file:
            file.read(self.data)
        op_output.emit(self.data, "frames_out")


# Threshold operator
class ThresholdSumOperator(Operator):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.threshold = self.kwargs.get("threshold", 27)
        super().__init__(*args, **kwargs)

    def setup(self, spec: Operator):
        spec.input("frames_in")
        spec.output("summed_frames_out")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("frames_in")
        thresholded_sum = cp.zeros(data.shape[1:])
        for frame in data:
            frame = cp.where(frame < 27, 0, frame)
            thresholded_sum += frame
        op_output.emit(thresholded_sum, "summed_frames_out")


# saving operator (saves images to folder)
class PlottingOperator(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("data_in")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("data_in")
        f = kvikio.CuFile("/mnt/nvme/data/test-file-complete", "w")
        # Write whole array to file
        f.write(data)
        f.close()


class HDF5ProcessingApp(Application):
    def compose(self):

        # defaults
        default_config = {
            "hdf5_reader": {"file_path": out_path},
            "threshold_sum": {"threshold": 27},
        }
        config_path = "config.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                config = yaml.safe_load(config_file)
                print("config file loaded")
        else:
            config = default_config
            print("using defaults")

        reader = ReaderOperator(
            self,
            CountCondition(self, 100),
            file_path=str(config["hdf5_reader"]["file_path"]),
            name="hdf5_reader",
        )
        threshold_sum = ThresholdSumOperator(
            self, threshold=config["threshold_sum"]["threshold"], name="threshold_sum"
        )
        plotter = PlottingOperator(self, name="plotter")

        # add flows
        self.add_flow(reader, threshold_sum, {("frames_out", "frames_in")})
        self.add_flow(threshold_sum, plotter, {("summed_frames_out", "data_in")})


def main():
    if kvikio.defaults.compat_mode():
        raise Exception("Enable compat_mode for Kvikio to ensure GDS is working correctly.")
    app = HDF5ProcessingApp()
    app.config("config.yaml")
    with Tracker(app, filename="logger.log") as tracker:
        app.run()
        tracker.print()


if __name__ == "__main__":
    main()
