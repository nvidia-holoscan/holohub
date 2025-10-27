"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.core import Application
from holoscan.operators import (
    FormatConverterOp,
    InferenceOp,
    InferenceProcessorOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import UnboundedAllocator


class EndoscopyOutOfBodyDetectionApp(Application):
    """Endoscopy Out-of-Body Detection"""

    def __init__(self, data_dir, source="replayer", do_record=False, enable_analytics=False):
        super().__init__()
        self.data_dir = data_dir
        self.source = source.lower()
        if self.source not in ["replayer", "aja"]:
            raise ValueError(
                f"Unsupported source: {source}. Please choose either 'replayer' or 'aja'."
            )
        self.do_record = do_record
        self.enable_analytics = enable_analytics

    def compose(self):
        """Compose the Holoscan application pipeline."""

        # Select the input source
        is_aja = self.source == "aja"
        if is_aja:
            from holohub.aja_source import AJASourceOp

            source = AJASourceOp(self, name="aja_source", **self.kwargs("aja"))
        else:
            replayer_config = "analytics_replayer" if self.enable_analytics else "replayer"
            print("Replayer config: ", replayer_config)
            print("Replayer config: ", self.kwargs(replayer_config))
            source = VideoStreamReplayerOp(
                self, name="video_replayer", directory=self.data_dir, **self.kwargs(replayer_config)
            )

        # Memory allocator for some operators
        pool = UnboundedAllocator(self, name="pool")
        in_dtype = "rgba8888" if is_aja else "rgb888"

        # Format conversion (ensures correct format for inference)
        out_of_body_preprocessor = FormatConverterOp(
            self,
            name="out_of_body_preprocessor",
            pool=pool,
            in_dtype=in_dtype,
            **self.kwargs("out_of_body_preprocessor"),
        )

        # AI Model Inference (detects whether endoscope is inside or outside the body)
        inference_kwargs = self.kwargs("out_of_body_inference")
        for k, v in inference_kwargs["model_path_map"].items():
            inference_kwargs["model_path_map"][k] = os.path.join(self.data_dir, v)
        out_of_body_inference = InferenceOp(
            self, name="out_of_body_inference", allocator=pool, **inference_kwargs
        )

        # Post-processing for inference results
        postprocess_config = (
            "analytics_out_of_body_postprocessor"
            if self.enable_analytics
            else "out_of_body_postprocessor"
        )
        out_of_body_postprocessor = InferenceProcessorOp(
            self,
            name="out_of_body_postprocessor",
            allocator=pool,
            disable_transmitter=True,
            **self.kwargs(postprocess_config),
        )

        # Define the pipeline connections
        if is_aja:
            self.add_flow(source, out_of_body_preprocessor, {("video_buffer_output", "")})
        else:
            self.add_flow(source, out_of_body_preprocessor)
        self.add_flow(out_of_body_preprocessor, out_of_body_inference, {("", "receivers")})
        self.add_flow(
            out_of_body_inference, out_of_body_postprocessor, {("transmitter", "receivers")}
        )


def main(args):
    app = EndoscopyOutOfBodyDetectionApp(args.data, args.source, args.record, args.analytics)
    app.config(args.config)
    app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Endoscopy Out-of-Body Detection (Python)")
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=os.environ.get("HOLOHUB_DATA_PATH", "../data"),
        help="Path to the data directory",
    )
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer", "aja"],
        default="replayer",
        help="Input source (default: replayer)",
    )
    parser.add_argument("--record", action="store_true", help="Record the input video")
    parser.add_argument("--analytics", action="store_true", help="Enable analytics")
    args = parser.parse_args()
    print("ARGS:", args)
    main(args)
