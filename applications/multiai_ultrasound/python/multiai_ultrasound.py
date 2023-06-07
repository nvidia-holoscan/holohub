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

import os
from argparse import ArgumentParser

from holoscan.core import Application
from holoscan.operators import (
    AJASourceOp,
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    InferenceProcessorOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import UnboundedAllocator

from holohub.visualizer_icardio import VisualizerICardioOp
from holohub.qcap_source import QCAPSourceOp


class MultiAIICardio(Application):
    def __init__(self, data, source="replayer"):
        super().__init__()

        if data == "none":
            data = os.environ.get("HOLOSCAN_DATA_PATH", "../data")

        # set name
        self.name = "Ultrasound App"

        # Optional parameters affecting the graph created by compose.
        source = source.lower()
        if source not in ["replayer", "aja", "qcap"]:
            raise ValueError(f"unsupported source: {source}. Please use 'replayer', 'aja', or 'qcap'.")
        self.source = source

        self.sample_data_path = data

    def compose(self):
        is_aja = self.source.lower() == "aja"
        is_qcap = self.source.lower() == "qcap"
        SourceClass = AJASourceOp if is_aja else VideoStreamReplayerOp
        SourceClass = QCAPSourceOp if is_qcap else SourceClass
        source_kwargs = self.kwargs(self.source)
        if self.source == "replayer":
            video_dir = self.sample_data_path
            if not os.path.exists(video_dir):
                raise ValueError(f"Could not find video data: {video_dir=}")
            source_kwargs["directory"] = video_dir
        source = SourceClass(self, name=self.source, **source_kwargs)

        in_dtype = "rgba8888" if is_aja else "rgb888"
        pool = UnboundedAllocator(self, name="pool")
        plax_cham_resized = FormatConverterOp(
            self,
            name="plax_cham_resized",
            pool=pool,
            in_dtype=in_dtype,
            **self.kwargs("plax_cham_resized"),
        )
        plax_cham_pre = FormatConverterOp(
            self,
            name="plax_cham_pre",
            pool=pool,
            in_dtype=in_dtype,
            **self.kwargs("plax_cham_pre"),
        )
        aortic_ste_pre = FormatConverterOp(
            self,
            name="aortic_ste_pre",
            pool=pool,
            in_dtype=in_dtype,
            **self.kwargs("aortic_ste_pre"),
        )
        b_mode_pers_pre = FormatConverterOp(
            self,
            name="b_mode_pers_pre",
            pool=pool,
            in_dtype=in_dtype,
            **self.kwargs("b_mode_pers_pre"),
        )

        model_path_map = {
            "plax_chamber": os.path.join(self.sample_data_path, "plax_chamber.onnx"),
            "aortic_stenosis": os.path.join(self.sample_data_path, "aortic_stenosis.onnx"),
            "bmode_perspective": os.path.join(self.sample_data_path, "bmode_perspective.onnx"),
        }
        for k, v in model_path_map.items():
            if not os.path.exists(v):
                raise RuntimeError(f"Could not find model file: {v}")
        inference_kwargs = self.kwargs("multiai_inference")
        inference_kwargs["model_path_map"] = model_path_map

        device_map = dict()
        if "device_map" in inference_kwargs.keys():
            device_map = inference_kwargs["device_map"]
            for k, v in device_map.items():
                device_map[k] = str(v)
            inference_kwargs["device_map"] = device_map

        multiai_inference = InferenceOp(
            self,
            name="multiai_inference",
            allocator=pool,
            **inference_kwargs,
        )

        multiai_postprocessor = InferenceProcessorOp(
            self,
            allocator=pool,
            **self.kwargs("multiai_postprocessor"),
        )
        visualizer_icardio = VisualizerICardioOp(
            self, allocator=pool, **self.kwargs("visualizer_icardio")
        )
        holoviz = HolovizOp(self, allocator=pool, name="holoviz", **self.kwargs("holoviz"))

        # connect the input to the resizer and each pre-processor
        for op in [plax_cham_resized, plax_cham_pre, aortic_ste_pre, b_mode_pers_pre]:
            if is_aja or is_qcap:
                self.add_flow(source, op, {("video_buffer_output", "")})
            else:
                self.add_flow(source, op)

        # connect the resized source video to the visualizer
        self.add_flow(plax_cham_resized, holoviz, {("", "receivers")})

        # connect all pre-processor outputs to the inference operator
        for op in [plax_cham_pre, aortic_ste_pre, b_mode_pers_pre]:
            self.add_flow(op, multiai_inference, {("", "receivers")})

        # connect the inference output to the postprocessor
        self.add_flow(multiai_inference, multiai_postprocessor, {("transmitter", "receivers")})

        # prepare postprocessed output for visualization with holoviz
        self.add_flow(multiai_postprocessor, visualizer_icardio, {("transmitter", "receivers")})

        # connect the overlays to holoviz
        visualizer_inputs = (
            "keypoints",
            "keyarea_1",
            "keyarea_2",
            "keyarea_3",
            "keyarea_4",
            "keyarea_5",
            "lines",
            "logo",
        )
        for src in visualizer_inputs:
            self.add_flow(visualizer_icardio, holoviz, {(src, "receivers")})


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-AI demo application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer", "aja", "qcap"],
        default="replayer",
        help=(
            "If 'replayer', replay a prerecorded video. If 'aja' use an AJA "
            "capture card as the source. If 'qcap' use an QCAP capture card "
            "as the source (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )
    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "multiai_ultrasound.yaml")
    else:
        config_file = args.config

    app = MultiAIICardio(source=args.source, data=args.data)
    app.config(config_file)
    app.run()
