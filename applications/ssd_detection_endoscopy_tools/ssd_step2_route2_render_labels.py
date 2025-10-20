# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import FormatConverterOp, HolovizOp, InferenceOp, VideoStreamReplayerOp
from holoscan.resources import BlockMemoryPool, MemoryStorageType, UnboundedAllocator

from holohub.aja_source import AJASourceOp

try:
    import cupy as cp
except ImportError:
    raise ImportError(
        "CuPy must be installed to run this example. See "
        "https://docs.cupy.dev/en/stable/install.html"
    )
import csv


class DetectionPostprocessorOp(Operator):
    """Example of an operator post processing the tensor from inference component.
    Following the example of tensor_interop.py and ping.py4

    This operator has:
        inputs:  "input_tensor"
        outputs: "output_tensor"
    """

    def __init__(self, *args, label_dict={}, label_text_size=0.05, scores_threshold=0.3, **kwargs):
        self.label_text_size = label_text_size
        self.scores_threshold = scores_threshold
        self.label_dict = label_dict

        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def append_size_to_text_coord(self, text_coord, size):
        # text_coord should be of shape [1, -1, 2]
        # we want to add a third size number to each (x, y)
        # so the text_coord shape [1, -1, 3]
        # the size number determines the text display size in Holoviz
        text_size = np.ones((1, text_coord.shape[1], 1)) * size
        new_text_coord = np.append(text_coord, text_size, 2)
        return new_text_coord.astype(np.float32)

    def compute(self, op_input, op_output, context):
        # Get input message which is a dictionary
        in_message = op_input.receive("in")
        # Convert input to numpy array (using CuPy) via .get()
        output_bboxes = cp.asarray(in_message["inference_output_detection_boxes"]).get()
        output_scores = cp.asarray(in_message["inference_output_detection_scores"]).get()
        output_labels = cp.asarray(in_message["inference_output_detection_classes"]).get()
        # can check the data type of the incoming tensors here
        # print(output_num_det.dtype)

        # Threshold output_scores and prune boxes
        ix = output_scores.flatten() >= self.scores_threshold
        has_rect = ix.any()

        output_bboxes = output_bboxes[:, ix, :]  # output_bboxes is of size [1, num_bbox, 4]
        output_labels = output_labels[:, ix].flatten()  # labels is of size [ num_bbox]

        bbox_coords = np.zeros([1, 2, 2], dtype=np.float32)

        if len(self.label_dict) > 0:
            # the label file isn't empty, we want to colorize the bbox and text colors
            bbox_coords = {}
            text_coords = {}
            for label in self.label_dict:
                bbox_coords[label] = np.zeros([1, 2, 2], dtype=np.float32)
                # coords tensor for text to display in Holoviz can be of shape [1, n, 2] or [1, n, 3]
                # with each location having [x,y] coords or [x,y,s] coords where s = size of text to
                # display
                text_coords[label] = np.zeros([1, 1, 2], dtype=np.float32) - 1.0

            if has_rect:
                # there are bboxes and we want to colorize them as well as label text
                for label in self.label_dict:
                    curr_l_ix = output_labels == label

                    if curr_l_ix.any():
                        bbox_coords[label] = np.reshape(output_bboxes[0, curr_l_ix, :], (1, -1, 2))
                        text_coords[label] = self.append_size_to_text_coord(
                            np.reshape(output_bboxes[0, curr_l_ix, :2], (1, -1, 2)),
                            self.label_text_size,
                        )

        else:
            # the label file is empty, just display bboxes in one color
            if has_rect:
                bbox_coords = np.reshape(output_bboxes, (1, -1, 2))

        # Create output message
        out_message = {}
        if len(self.label_dict) > 0:
            # we have split bboxs and text labels into categories
            for label in self.label_dict:
                out_message["rectangles" + str(label)] = bbox_coords[label]
                out_message["label" + str(label)] = text_coords[label]
        else:
            # only transmit the bbox_coords
            out_message["rectangles"] = bbox_coords

        op_output.emit(out_message, "out")


class SSDDetectionApp(Application):
    def __init__(self, source="replayer", labelfile=""):
        """Initialize the ssd detection application

        Parameters
        ----------
        source : {"replayer", "aja"}
            When set to "replayer" (the default), pre-recorded sample video data is
            used as the application input. Otherwise, the video stream from an AJA
            capture card is used.
        """

        super().__init__()

        # set name
        self.name = "SSD Detection App"

        # Optional parameters affecting the graph created by compose.
        self.source = source
        self.labelfile = labelfile

    def compose(self):
        n_channels = 4  # RGBA
        bpp = 4  # bytes per pixel
        label_dict = {}
        if self.labelfile != "":
            assert os.path.isfile(self.labelfile)
            with open(self.labelfile, newline="") as labelcsv:
                csvreader = csv.reader(labelcsv)
                for row in csvreader:
                    # assume each row looks like: 1, "Grasper", 1.0, 0.0, 1.0
                    label_dict[int(row[0])] = {}
                    label_dict[int(row[0])]["text"] = str(row[1])
                    label_dict[int(row[0])]["color"] = [float(row[2]), float(row[3]), float(row[4])]

        is_aja = self.source.lower() == "aja"
        drop_alpha_block_size = 1920 * 1080 * n_channels * bpp
        drop_alpha_num_blocks = 2

        if is_aja:
            source = AJASourceOp(self, name="aja", **self.kwargs("aja"))
            drop_alpha_block_size = 1920 * 1080 * n_channels * bpp
            drop_alpha_num_blocks = 2
            drop_alpha_channel = FormatConverterOp(
                self,
                name="drop_alpha_channel",
                pool=BlockMemoryPool(
                    self,
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=drop_alpha_block_size,
                    num_blocks=drop_alpha_num_blocks,
                ),
                **self.kwargs("drop_alpha_channel"),
            )
        else:
            source = VideoStreamReplayerOp(self, name="replayer", **self.kwargs("replayer"))

        width_preprocessor = 1920
        height_preprocessor = 1080
        preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp
        preprocessor_num_blocks = 2
        detection_preprocessor = FormatConverterOp(
            self,
            name="detection_preprocessor",
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=preprocessor_block_size,
                num_blocks=preprocessor_num_blocks,
            ),
            **self.kwargs("detection_preprocessor"),
        )

        detection_inference = InferenceOp(
            self,
            name="detection_inference",
            allocator=UnboundedAllocator(self, name="pool"),
            **self.kwargs("detection_inference"),
        )

        detection_postprocessor = DetectionPostprocessorOp(
            # this is where we write our own post processor in the BYOM process
            self,
            name="detection_postprocessor",
            allocator=UnboundedAllocator(self, name="allocator"),
            label_dict=label_dict,
            **self.kwargs("detection_postprocessor"),
        )

        holoviz_tensors = [dict(name="", type="color")]
        if len(label_dict) > 0:
            for label in label_dict:
                color = label_dict[label]["color"]
                color.append(1.0)
                text = [label_dict[label]["text"]]
                holoviz_tensors.append(
                    dict(
                        name="rectangles" + str(label),
                        type="rectangles",
                        opacity=0.5,
                        line_width=4,
                        color=color,
                    )
                )
                holoviz_tensors.append(
                    dict(
                        name="label" + str(label), type="text", opacity=0.5, color=color, text=text
                    )
                )
        else:
            holoviz_tensors.append(
                dict(
                    name="rectangles",
                    type="rectangles",
                    opacity=0.5,
                    line_width=4,
                    color=[1.0, 0.0, 0.0, 1.0],
                )
            )
        detection_visualizer = HolovizOp(
            self,
            name="detection_visualizer",
            tensors=holoviz_tensors,
            **self.kwargs("detection_visualizer"),
        )

        if is_aja:
            self.add_flow(source, detection_visualizer, {("video_buffer_output", "receivers")})
            self.add_flow(source, drop_alpha_channel, {("video_buffer_output", "")})
            self.add_flow(drop_alpha_channel, detection_preprocessor)
        else:
            self.add_flow(source, detection_visualizer, {("", "receivers")})
            self.add_flow(source, detection_preprocessor)

        self.add_flow(detection_preprocessor, detection_inference, {("", "receivers")})
        self.add_flow(detection_inference, detection_postprocessor, {("transmitter", "in")})
        self.add_flow(detection_postprocessor, detection_visualizer, {("out", "receivers")})


def main():
    # Parse args
    parser = ArgumentParser(description="SSD Detection demo application.")
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
        "-l",
        "--labelfile",
        default="",
        help=(
            "Optional arg for a csv file path containing the class labels. There "
            "should be 5 columns: [label value, label text, red, green, blue] "
            "where label value = the model output value for each class to display and should be int, "
            "label text = the text to display for this label, "
            "red/green/blue should = values between 0 and 1 for the display color "
            "for this class."
        ),
    )
    args = parser.parse_args()

    config_file = os.path.join(os.path.dirname(__file__), "ssd_endo_model_with_NMS.yaml")

    app = SSDDetectionApp(source=args.source, labelfile=args.labelfile)
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    main()
