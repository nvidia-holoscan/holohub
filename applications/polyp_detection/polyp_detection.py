# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import holoscan as hs
import torch
import torch.nn.functional as F
import torchvision
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.operators import FormatConverterOp, HolovizOp, InferenceOp, VideoStreamReplayerOp
from holoscan.resources import BlockMemoryPool, MemoryStorageType, UnboundedAllocator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolypDetPostprocessorOp(Operator):
    """Example of an operator post processing the tensor from inference component.

    This operator has:
        inputs:  "input_tensor"
        outputs: "output_tensor"
    """

    def __init__(self, *args, max_det=10, scores_threshold=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_det = max_det
        self.scores_threshold = scores_threshold

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def postprocess(self, outputs):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        logits = torch.as_tensor(logits, device=DEVICE)
        boxes = torch.as_tensor(boxes, device=DEVICE)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

        scores = F.sigmoid(logits)
        scores, index = torch.topk(scores.flatten(1), self.max_det, dim=-1)
        boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        return boxes, scores

    def compute(self, op_input, op_output, context):
        # Get input message which is a dictionary
        in_message = op_input.receive("in")
        # (B, N, 4), (B, N)
        pred_boxes, pred_scores = self.postprocess(in_message)
        ix = pred_scores.flatten() > self.scores_threshold
        if torch.all(ix == False):
            bboxes = torch.zeros([1, 2, 2], dtype=torch.float32)
        else:
            bboxes = pred_boxes[:, ix, :].view(-1, 2, 2)

        bboxes = bboxes.cpu().numpy()
        out_message = Entity(context)
        out_message.add(hs.as_tensor(bboxes), "rectangles")
        op_output.emit(out_message, "out")


class PolypDetectionApp(Application):
    def __init__(
        self,
        video_dir,
        data,
        source="replayer",
        video_size=(720, 576),
    ):
        """Initialize the colonoscopy detection application

        Parameters
        ----------
        source : {"replayer"}
            When set to "replayer" (the default), pre-recorded sample video data is
            used as the application input.
            capture card is used.
        video_dir: str
            Path to the input video data.
        data : str
            Path to the onnx model.
        video_size : tuple
            Size of the video input as (width, height).
        """

        super().__init__()

        # set name
        self.name = "Polyp Detection App"

        # Optional parameters affecting the graph created by compose.
        self.source = source

        # Optional parameters affecting the graph created by compose.
        if data == "none":
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")

        if video_dir == "none":
            video_dir = "./example_video"

        self.video_dir = video_dir
        self.video_size = video_size
        self.model_path = os.path.join(
            data, "rtdetrv2_timm_r50_nvimagenet_pretrained_neg_finetune_bhwc.onnx"
        )

    def compose(self):
        n_channels = 3
        bpp = 4  # bytes per pixel

        if self.source == "replayer":
            video_dir = self.video_dir
            if not os.path.exists(video_dir):
                raise ValueError(f"Could not find video data: {video_dir=}")
            source = VideoStreamReplayerOp(
                self, name="replayer", directory=video_dir, **self.kwargs("replayer")
            )
            width_preprocessor = self.video_size[0]
            height_preprocessor = self.video_size[1]

        else:
            raise ValueError(f"Unknown source type: {self.source}")

        preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp
        preprocessor_num_blocks = 3
        detection_preprocessor = FormatConverterOp(
            self,
            name="detection_preprocessor",
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=preprocessor_block_size,
                num_blocks=preprocessor_num_blocks,
            ),
            resize_width=640,
            resize_height=640,
            **self.kwargs("detection_preprocessor"),
        )

        detection_inference = InferenceOp(
            self,
            name="detection_inference",
            allocator=UnboundedAllocator(self, name="pool"),
            model_path_map={"polyp_det": self.model_path},
            **self.kwargs("detection_inference"),
        )

        detection_postprocessor = PolypDetPostprocessorOp(
            self,
            name="detection_postprocessor",
            allocator=UnboundedAllocator(self, name="allocator"),
            **self.kwargs("detection_postprocessor"),
        )

        detection_visualizer = HolovizOp(
            self,
            name="detection_visualizer",
            tensors=[
                dict(name="", type="color"),
                dict(
                    name="rectangles",
                    type="rectangles",
                    opacity=0.5,
                    line_width=4,
                    color=[1.0, 0.0, 0.0, 1.0],
                ),
            ],
            **self.kwargs("detection_visualizer"),
        )

        self.add_flow(source, detection_visualizer, {("", "receivers")})
        self.add_flow(source, detection_preprocessor)
        self.add_flow(detection_preprocessor, detection_inference, {("tensor", "receivers")})
        self.add_flow(detection_inference, detection_postprocessor, {("transmitter", "in")})
        self.add_flow(detection_postprocessor, detection_visualizer, {("out", "receivers")})


def main():
    # Parse args
    parser = ArgumentParser(description="Polyp Detection demo application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer"],
        default="replayer",
        help=("If 'replayer', replay a prerecorded video"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the model path"),
    )
    parser.add_argument(
        "-v",
        "--video_dir",
        default="none",
        help=("Set the video dir path"),
    )
    parser.add_argument(
        "--video_size",
        default="(720, 576)",
        help=("Set the video size"),
    )
    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )

    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "polyp_detection.yaml")
    else:
        config_file = args.config
    args.video_size = tuple(map(int, args.video_size.strip("()").split(",")))
    app = PolypDetectionApp(
        source=args.source,
        data=args.data,
        video_dir=args.video_dir,
        video_size=args.video_size,
    )
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    main()
