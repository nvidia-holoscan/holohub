# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan import __version__ as holoscan_version
from holoscan.core import Application
from holoscan.operators import (
    AJASourceOp,
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    InferenceProcessorOp,
    VideoStreamRecorderOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType
from packaging.version import Version

from holohub.visualizer_icardio import VisualizerICardioOp


class MultiAIICardio(Application):
    def __init__(self, data, source="replayer", record_type=None):
        super().__init__()

        if data == "none":
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")

        # set name
        self.name = "Ultrasound App"

        # Optional parameters affecting the graph created by compose.
        source = source.lower()
        if source not in ["replayer", "aja"]:
            raise ValueError(f"unsupported source: {source}. Please use 'replayer' or 'aja'.")
        self.source = source

        self.record_type = record_type
        if record_type is not None:
            if record_type not in ("input", "visualizer"):
                raise ValueError("record_type must be either ('input' or 'visualizer')")

        self.sample_data_path = data

    def compose(self):
        cuda_stream_pool = CudaStreamPool(self, name="cuda_stream")

        record_type = self.record_type
        is_aja = self.source.lower() == "aja"

        SourceClass = AJASourceOp if is_aja else VideoStreamReplayerOp
        source_kwargs = self.kwargs(self.source)
        if self.source == "replayer":
            video_dir = self.sample_data_path
            if not os.path.exists(video_dir):
                raise ValueError(f"Could not find video data: {video_dir=}")
            source_kwargs["directory"] = video_dir
            # the RMMAllocator supported since v2.6 is much faster than the default UnboundAllocator
            try:
                from holoscan.resources import RMMAllocator

                source_kwargs["allocator"] = RMMAllocator(self, name="video_replayer_allocator")
            except Exception:
                pass
        source = SourceClass(self, name=self.source, **source_kwargs)

        in_dtype = "rgba8888" if is_aja else "rgb888"
        in_components = 4 if is_aja else 3
        # FormatConverterOp needs an temporary buffer if converting from RGBA
        format_convert_pool_blocks = 4 if in_components == 4 else 3
        bytes_per_float32 = 4
        plax_cham_pre = FormatConverterOp(
            self,
            name="plax_cham_pre",
            in_dtype=in_dtype,
            pool=BlockMemoryPool(
                self,
                name="plax_cham_pre_pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=320 * 320 * bytes_per_float32 * in_components,
                num_blocks=format_convert_pool_blocks,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("plax_cham_pre"),
        )
        aortic_ste_pre = FormatConverterOp(
            self,
            name="aortic_ste_pre",
            in_dtype=in_dtype,
            pool=BlockMemoryPool(
                self,
                name="aortic_ste_pre_pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=300 * 300 * bytes_per_float32 * in_components,
                num_blocks=format_convert_pool_blocks,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("aortic_ste_pre"),
        )
        b_mode_pers_pre = FormatConverterOp(
            self,
            name="b_mode_pers_pre",
            in_dtype=in_dtype,
            pool=BlockMemoryPool(
                self,
                name="b_mode_pers_pre_pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=320 * 240 * bytes_per_float32 * in_components,
                num_blocks=format_convert_pool_blocks,
            ),
            cuda_stream_pool=cuda_stream_pool,
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

        plax_chamber_output_size = 320 * 320 * bytes_per_float32 * 6
        aortic_stenosis_output_size = bytes_per_float32 * 2
        bmode_perspective_output_size = bytes_per_float32 * 1
        block_size = max(
            plax_chamber_output_size, aortic_stenosis_output_size, bmode_perspective_output_size
        )

        multiai_inference = InferenceOp(
            self,
            name="multiai_inference",
            allocator=BlockMemoryPool(
                self,
                name="multiai_inference_allocator",
                storage_type=MemoryStorageType.DEVICE,
                block_size=block_size,
                num_blocks=2 * 3,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **inference_kwargs,
        )

        # version 2.6 supports the CUDA version of `max_per_channel_scaled`
        try:
            supports_cuda_processing = Version(holoscan_version) >= Version("2.6")
        except Exception:
            supports_cuda_processing = False
        multiai_postprocessor = InferenceProcessorOp(
            self,
            input_on_cuda=supports_cuda_processing,
            output_on_cuda=supports_cuda_processing,
            allocator=BlockMemoryPool(
                self,
                name="multiai_postprocessor_allocator",
                storage_type=MemoryStorageType.DEVICE,
                block_size=2 * bytes_per_float32 * 6,
                num_blocks=1,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("multiai_postprocessor"),
        )

        visualizer_kwargs = self.kwargs("visualizer_icardio")
        visualizer_kwargs["data_dir"] = self.sample_data_path
        bytes_per_uint8 = 1
        visualizer_icardio = VisualizerICardioOp(
            self,
            allocator=BlockMemoryPool(
                self,
                name="visualizer_icardio_allocator",
                storage_type=MemoryStorageType.DEVICE,
                block_size=320 * 320 * 4 * bytes_per_uint8,
                num_blocks=1 * 8,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **visualizer_kwargs,
        )

        source_pool_kwargs = dict(
            storage_type=MemoryStorageType.DEVICE,
            block_size=block_size,
            num_blocks=1,
        )

        if record_type is not None:
            if ((record_type == "input") and (self.source != "replayer")) or (
                record_type == "visualizer"
            ):
                recorder_format_converter = FormatConverterOp(
                    self,
                    name="recorder_format_converter",
                    pool=BlockMemoryPool(self, name="pool", **source_pool_kwargs),
                    **self.kwargs("recorder_format_converter"),
                )
            recorder = VideoStreamRecorderOp(
                name="recorder", fragment=self, **self.kwargs("recorder")
            )

        if (record_type == "visualizer") and (self.source == "replayer"):
            visualizer_allocator = BlockMemoryPool(self, name="allocator", **source_pool_kwargs)
        else:
            visualizer_allocator = None

        holoviz = HolovizOp(
            self,
            name="holoviz",
            cuda_stream_pool=cuda_stream_pool,
            enable_render_buffer_output=record_type == "visualizer",
            allocator=visualizer_allocator,
            **self.kwargs("holoviz"),
        )

        # connect the input to the resizer and each pre-processor
        source_port_name = "video_buffer_output" if is_aja else ""
        for op in [plax_cham_pre, aortic_ste_pre, b_mode_pers_pre]:
            self.add_flow(source, op, {(source_port_name, "")})

        # connect the source video to the visualizer
        self.add_flow(source, holoviz, {(source_port_name, "receivers")})

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

        # Flow for the recorder
        if record_type == "input":
            if self.source != "replayer":
                self.add_flow(
                    source,
                    recorder_format_converter,
                    {("video_buffer_output", "source_video")},
                )
                self.add_flow(recorder_format_converter, recorder)
            else:
                self.add_flow(source, recorder)
        elif record_type == "visualizer":
            self.add_flow(
                holoviz,
                recorder_format_converter,
                {("render_buffer_output", "source_video")},
            )
            self.add_flow(recorder_format_converter, recorder)


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-AI demo application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer", "aja"],
        default="replayer",
        help=(
            "If 'replayer', replay a prerecorded video. If 'aja' use an AJA "
            "capture card as the source (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "-r",
        "--record_type",
        choices=["none", "input", "visualizer"],
        default="none",
        help="The video stream to record (default: %(default)s).",
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

    record_type = args.record_type
    if record_type == "none":
        record_type = None

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "multiai_ultrasound.yaml")
    else:
        config_file = args.config

    app = MultiAIICardio(record_type=record_type, source=args.source, data=args.data)
    app.config(config_file)
    app.run()
