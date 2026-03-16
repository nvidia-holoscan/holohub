# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Tracks2Endo4D: 3D point tracking and camera parameter estimation from video.

Uses overlapping forward runs with TapNextInferenceOp (C++) for robust long-range
point tracking combined with TracksTo4D for 3D reconstruction.
"""

import copy
import os
from argparse import ArgumentParser

from holoscan.conditions import BooleanCondition, PeriodicCondition, PeriodicConditionPolicy
from holoscan.core import Application
from holoscan.operators import FormatConverterOp, HolovizOp, InferenceOp, VideoStreamReplayerOp
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    RMMAllocator,
    UnboundedAllocator,
)
from postprocessing import PostprocessorOp, Visualize3DPostprocessorOp
from preprocessing import BatchSplitterVideoOp, OverlapWindowCoordinatorOp
from tracking import (
    BatchMergerOp,
    BatchMergerSchedulingOp,
    ReverseBatch,
    get_model_path,
)
from tracks_3d import Postprocess3DOp, Preprocessor3DOp, StitchPredictionsOp
from tracks_assembler import TracksAssemblerOp

import holohub.tracks2endo4d_viz as viz_cpp
from holohub.tapnext_inference import TapNextInferenceOp

bytes_per_float32 = 4


def get_bytes_tapnext():
    # Init and FWD are the same.
    size_float = max(
        1 * 1 * 225 * 2,
        1 * 1 * 225 * 512,
        1 * 1 * 225 * 1,
        1,
        12 * 1249 * 768,
        12 * 1249 * 3 * 768,
    )
    size = bytes_per_float32 * size_float
    return size


def get_bytes_tracks_4d():
    # Get the maximum size of the output bindings for Tracks4D.
    size_float = max(
        1,
        1 * 21 * 2 * 675,
        1 * 21 * 2 * 675,
        1 * 21 * 3 * 3,
        1 * 21 * 3,
        1 * 675 * 3 * 12,
        1 * 21 * 3 * 675,
        1 * 21 * 3 * 675,
        1 * 21 * 675,
        1 * 21 * 675,
        1,
        1 * 12 * 21,
        1,
        1,
        1 * 21 * 3 * 675,
        1 * 675,
    )
    size = bytes_per_float32 * size_float
    return size


class Tracks2Endo4DApp(Application):
    def __init__(self, data=None, model=None, viz_2d=False, count=None, source=None):
        super().__init__()

        self.name = "Tracks2Endo4D"

        data = os.environ.get("HOLOHUB_DATA_PATH", "../data") if data is None else data
        print(f"data path: {data}")
        print(f"Holohub data path: {os.environ.get('HOLOHUB_DATA_PATH')}")

        model = data if model is None else model

        self.data_path = os.path.join(data, "video")
        self.model_path = model
        self.viz_2d = viz_2d
        self.frame_count = count if (count is not None and count > 0) else 0
        self.source = source

    def compose(self):
        host_allocator = UnboundedAllocator(self, name="host_allocator")
        video_allocator = RMMAllocator(self, name="video_replayer_allocator")
        cuda_stream_pool = CudaStreamPool(self, name="cuda_stream_pool", max_size=6)

        if self.source == "replayer":
            video_dir = self.data_path
            if not os.path.exists(video_dir):
                raise ValueError(f"Could not find video data: {video_dir=}")
            replayer_kwargs = self.kwargs("replayer")
            if self.frame_count != 0:
                replayer_kwargs["count"] = self.frame_count

            # Input and formatting
            replayer = VideoStreamReplayerOp(
                self,
                name="replayer",
                directory=video_dir,
                allocator=video_allocator,
                **replayer_kwargs,
            )
        elif self.source == "aja":
            from holohub.aja_source import AJASourceOp

            source_dtype = "rgb888"

            aja = AJASourceOp(self, name="aja_source", **self.kwargs("aja"))
            # Convert VideoBuffer from AJA to Tensor (drop alpha channel for downstream compatibility)
            aja_format_converter = FormatConverterOp(
                self,
                name="aja_format_converter",
                pool=host_allocator,
                out_dtype=source_dtype,
                **self.kwargs("aja_format_converter"),
            )

        width_preprocessor = self.kwargs("preprocessor")["resize_width"]
        height_preprocessor = self.kwargs("preprocessor")["resize_height"]
        preprocessor_block_size = width_preprocessor * height_preprocessor * bytes_per_float32 * 3
        preprocessor = FormatConverterOp(
            self,
            name="preprocessor",
            pool=BlockMemoryPool(
                self,
                name="preprocessor_pool",
                block_size=preprocessor_block_size,
                num_blocks=3,
                storage_type=MemoryStorageType.DEVICE,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("preprocessor"),
        )

        window_args = copy.deepcopy(self.kwargs("window"))
        coordinator = OverlapWindowCoordinatorOp(
            self,
            name="coordinator",
            window_size=window_args["window_size"],
            overlap_size=window_args["overlap_size"],
        )

        if self.source == "replayer":
            self.add_flow(replayer, preprocessor, {("output", "source_video")})
        elif self.source == "aja":
            self.add_flow(aja, aja_format_converter, {("output", "source_video")})
            self.add_flow(aja_format_converter, preprocessor, {("", "source_video")})

        self.add_flow(preprocessor, coordinator, {("tensor", "in")})

        # Helper to resolve model path
        def resolve_model_path(path):
            return os.path.join(self.model_path, path) if not os.path.isabs(path) else path

        tapnext_kwargs = copy.deepcopy(self.kwargs("tapnext"))
        tapnext_kwargs["model_file_path_init"] = resolve_model_path(
            tapnext_kwargs["model_file_path_init"]
        )
        tapnext_kwargs["model_file_path_fwd"] = resolve_model_path(
            tapnext_kwargs["model_file_path_fwd"]
        )

        # --------------------------
        # Forward branch 0
        # --------------------------
        tapnext_fwd0 = TapNextInferenceOp(
            self,
            name="tapnext_fwd0",
            pool=BlockMemoryPool(
                self,
                name="tapnext_fwd0_pool",
                block_size=get_bytes_tapnext(),
                num_blocks=48,
                storage_type=MemoryStorageType.DEVICE,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **tapnext_kwargs,
        )

        merger_args_fwd = copy.deepcopy(window_args)
        batch_merger_fwd0 = BatchMergerSchedulingOp(
            self,
            name="batch_merger_fwd0",
            window_size=merger_args_fwd["window_size"],
            overlap_size=merger_args_fwd["overlap_size"],
            suffix="_fwd0",
            schedule_emission=lambda x: (
                x >= merger_args_fwd["window_size"] and x % merger_args_fwd["overlap_size"] == 0
            ),
        )

        self.add_flow(coordinator, tapnext_fwd0, {("fwd0_frame", "receivers")})
        self.add_flow(tapnext_fwd0, batch_merger_fwd0, {("transmitter", "predictions_in")})
        self.add_flow(coordinator, batch_merger_fwd0, {("fwd0_frame", "frame_in")})

        # --------------------------
        # Forward branch 1
        # --------------------------
        tapnext_fwd1 = TapNextInferenceOp(
            self,
            name="tapnext_fwd1",
            pool=BlockMemoryPool(
                self,
                name="tapnext_fwd1_pool",
                block_size=get_bytes_tapnext(),
                num_blocks=48,
                storage_type=MemoryStorageType.DEVICE,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **tapnext_kwargs,
        )

        batch_merger_fwd1 = BatchMergerSchedulingOp(
            self,
            name="batch_merger_fwd1",
            window_size=merger_args_fwd["window_size"],
            overlap_size=merger_args_fwd["overlap_size"],
            suffix="_fwd1",
            schedule_emission=lambda x: (
                x >= merger_args_fwd["overlap_size"] and x % merger_args_fwd["overlap_size"] == 0
            ),
        )

        self.add_flow(coordinator, tapnext_fwd1, {("fwd1_frame", "receivers")})
        self.add_flow(tapnext_fwd1, batch_merger_fwd1, {("transmitter", "predictions_in")})
        self.add_flow(coordinator, batch_merger_fwd1, {("fwd1_frame", "frame_in")})

        # --------------------------
        # Backward branch (per-window)
        # --------------------------
        reverse_frames_bwd = ReverseBatch(
            self, name="reverse_frames_bwd", do_backwards=True, axis=0
        )

        # Use BatchSplitterVideoOp to format for TapNextInferenceOp
        self.run_flag = BooleanCondition(self, name="run_flag", enable_tick=True)
        splitter_bwd = BatchSplitterVideoOp(
            self,
            self.run_flag,
            name="splitter_bwd",
            allocator=host_allocator,
            grid_query_frame=0,
            max_frames=self.frame_count,
        )

        tapnext_bwd = TapNextInferenceOp(
            self,
            name="tapnext_bwd",
            pool=BlockMemoryPool(
                self,
                name="tapnext_bwd_pool",
                block_size=get_bytes_tapnext(),
                num_blocks=48,
                storage_type=MemoryStorageType.DEVICE,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **tapnext_kwargs,
        )

        batch_merger_bwd = BatchMergerOp(
            self,
            name="batch_merger_bwd",
            window_size=merger_args_fwd["window_size"],
            suffix="_bwd",
        )
        reverse_tracks_bwd = ReverseBatch(
            self, name="reverse_tracks_bwd", do_backwards=True, axis=0
        )

        self.add_flow(coordinator, reverse_frames_bwd, {("batch_out", "in")})
        self.add_flow(reverse_frames_bwd, splitter_bwd, {("out", "in")})
        self.add_flow(splitter_bwd, tapnext_bwd, {("out", "receivers")})

        self.add_flow(tapnext_bwd, batch_merger_bwd, {("transmitter", "predictions_in")})
        self.add_flow(splitter_bwd, batch_merger_bwd, {("out", "frame_in")})

        self.add_flow(batch_merger_bwd, reverse_tracks_bwd, {("out", "in")})

        # --------------------------
        # Assemble (generic) and visualize
        # --------------------------
        assembler = TracksAssemblerOp(
            self,
            name="tracks_assembler_generic",
            window_size=window_args["window_size"],
            grid_size=window_args.get("grid_size", 15),
            overlap_size=window_args["overlap_size"],
        )

        self.add_flow(batch_merger_fwd0, assembler, {("out", "input0_in")})
        self.add_flow(batch_merger_fwd1, assembler, {("out", "input1_in")})
        self.add_flow(reverse_tracks_bwd, assembler, {("out", "backward_in")})

        if self.viz_2d:
            postprocessor = PostprocessorOp(
                self,
                name="visualize_3d_postprocessor",
                **self.kwargs("window"),
            )

            holoviz = HolovizOp(
                self,
                PeriodicCondition(
                    self,
                    name="postprocess_period",
                    recess_period=0.033,
                    policy=PeriodicConditionPolicy.NO_CATCH_UP_MISSED_TICKS,
                ),
                name="holoviz",
                **self.kwargs("holoviz_2d"),
            )

            self.add_flow(assembler, postprocessor, {("out", "inference_result")})
            self.add_flow(assembler, postprocessor, {("track_ids_out", "track_ids")})
            self.add_flow(postprocessor, holoviz, {("output", "receivers")})
        else:
            # ------------------------------------------------------------
            # 2D -> 3D Tracking
            # ------------------------------------------------------------

            preprocessor_3d = Preprocessor3DOp(
                self, name="preprocessor_3d", **self.kwargs("preprocessor_3d")
            )
            tracks_4d_args = get_model_path(
                self.kwargs("tracks_4d"), self.model_path, name="tracks_4d"
            )
            inference_tracks_4d = InferenceOp(
                self,
                name="inference_tracks_4d",
                allocator=BlockMemoryPool(
                    self,
                    name="inference_tracks_4d_pool",
                    block_size=get_bytes_tracks_4d(),
                    num_blocks=40,  # Output size + margin
                    storage_type=MemoryStorageType.DEVICE,
                ),
                cuda_stream_pool=cuda_stream_pool,
                **tracks_4d_args,
            )
            stitcher = StitchPredictionsOp(self, name="stitcher", **self.kwargs("window"))
            postprocess_3d = Postprocess3DOp(
                self,
                name="postprocess_3d",
                **self.kwargs("window"),
                **self.kwargs("preprocessor_3d"),
            )

            viz_processor = Visualize3DPostprocessorOp(
                self,
                PeriodicCondition(
                    self,
                    name="postprocess_period",
                    recess_period=0.04,
                    policy=PeriodicConditionPolicy.NO_CATCH_UP_MISSED_TICKS,
                ),
                name="visualize_3d_postprocessor",
                **self.kwargs("window"),
            )

            holoviz = viz_cpp.VizOp(
                self,
                name="viz",
                device_allocator=host_allocator,
                **self.kwargs("holoviz_3d"),
            )

            self.add_flow(assembler, preprocessor_3d, {("out", "in")})
            self.add_flow(preprocessor_3d, inference_tracks_4d, {("out", "receivers")})
            self.add_flow(inference_tracks_4d, stitcher, {("transmitter", "predictions")})
            self.add_flow(assembler, stitcher, {("track_ids_out", "track_ids")})

            self.add_flow(stitcher, postprocess_3d, {("out", "predictions")})
            self.add_flow(preprocessor_3d, postprocess_3d, {("out", "tracks")})
            self.add_flow(preprocessor_3d, postprocess_3d, {("out_frames", "frames")})

            self.add_flow(postprocess_3d, viz_processor, {("out", "in_results")})

            self.add_flow(viz_processor, holoviz, {("output", "receivers")})


if __name__ == "__main__":
    os.environ["HOLOSCAN_LOG_LEVEL"] = "INFO"

    parser = ArgumentParser(
        description="Tracks2Endo4D: 3D point tracking and camera estimation from video."
    )
    parser.add_argument(
        "--source",
        choices=["replayer", "aja"],
        default="replayer",
        help="Set the source type",
    )
    parser.add_argument(
        "-d",
        "--data",
        default=None,
        help="Set the data path",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Set the model path",
    )
    parser.add_argument(
        "--viz-2d",
        action="store_true",
        default=False,
        help="Enable 2D visualization",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Override the replayer frame count (0 = unlimited)",
    )
    args = parser.parse_args()

    # Use the tapnextcpp config
    config_file = os.path.join(os.path.dirname(__file__), "config.yaml")

    print(f"args: {args}")
    app = Tracks2Endo4DApp(
        data=args.data,
        model=args.model,
        viz_2d=args.viz_2d,
        count=args.count,
        source=args.source,
    )
    app.config(config_file)
    app.run()
