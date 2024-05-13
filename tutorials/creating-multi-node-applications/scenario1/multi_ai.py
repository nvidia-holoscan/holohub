# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import csv
import os
import sys
from argparse import ArgumentParser

import cupy as cp
import numpy as np
from holoscan.core import Application, Fragment, Operator, OperatorSpec
from holoscan.operators import (
    AJASourceOp,
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    SegmentationPostprocessorOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import UnboundedAllocator


class DetectionPostprocessorOp(Operator):
    """Example of an operator post processing the tensor from inference component.
    Following the example of tensor_interop.py and ping.py4

    This operator has:
        inputs:  "in"
        outputs: "out"
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
        # Convert input to numpy array (using CuPy)
        cp.asarray(in_message.get("inference_output_num_detections")).get()
        output_bboxes = cp.asarray(in_message["inference_output_detection_boxes"]).get()
        output_scores = cp.asarray(in_message["inference_output_detection_scores"]).get()
        output_labels = cp.asarray(in_message["inference_output_detection_classes"]).get()
        # can check the data type of the incoming tensors here
        # print(output_labels.dtype)

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
                # with each location having [x,y] coords or [x,y,s] coords where s = size of text
                # to display
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


class Fragment1(Fragment):
    def __init__(self, app, name, sample_data_path, source, label_dict):
        super().__init__(app, name)

        self.source = source
        self.label_dict = label_dict
        self.sample_data_path = sample_data_path

    def compose(self):
        # start constructing app
        is_aja = self.source.lower() == "aja"
        SourceClass = AJASourceOp if is_aja else VideoStreamReplayerOp
        source_kwargs = self.kwargs(self.source)
        if self.source == "replayer":
            video_dir = os.path.join(self.sample_data_path, "endoscopy")
            if not os.path.exists(video_dir):
                raise ValueError(f"Could not find video data: {video_dir=}")
            source_kwargs["directory"] = video_dir
        source = SourceClass(self, name=self.source, **source_kwargs)

        in_dtype = "rgba8888" if is_aja else "rgb888"
        pool = UnboundedAllocator(self, name="pool")

        detection_preprocessor = FormatConverterOp(
            self,
            name="detection_preprocessor",
            pool=pool,
            in_dtype=in_dtype,
            **self.kwargs("detection_preprocessor"),
        )
        segmentation_preprocessor = FormatConverterOp(
            self,
            name="segmentation_preprocessor",
            pool=pool,
            in_dtype=in_dtype,
            **self.kwargs("segmentation_preprocessor"),
        )

        model_path_map = {
            "ssd": os.path.join(self.sample_data_path, "ssd_model", "epoch24_nms.onnx"),
            "tool_seg": os.path.join(
                self.sample_data_path,
                "monai_tool_seg_model",
                "model_endoscopic_tool_seg_sanitized_nhwc_in_nchw_out.onnx",
            ),
        }
        for k, v in model_path_map.items():
            if not os.path.exists(v):
                raise RuntimeError(f"Could not find model file: {v}")
        inference_kwargs = self.kwargs("multi_ai_inference")
        inference_kwargs["model_path_map"] = model_path_map

        multi_ai_inference = InferenceOp(
            self,
            name="multi_ai_inference",
            allocator=pool,
            **inference_kwargs,
        )

        detection_postprocessor = DetectionPostprocessorOp(
            self,
            name="detection_postprocessor",
            label_dict=self.label_dict,
            allocator=UnboundedAllocator(self, name="allocator"),
            **self.kwargs("detection_postprocessor"),
        )

        segmentation_postprocessor = SegmentationPostprocessorOp(
            self,
            name="segmentation_postprocessor",
            allocator=pool,
            **self.kwargs("segmentation_postprocessor"),
        )

        # connect the input each pre-processor
        if is_aja:
            self.add_flow(source, detection_preprocessor, {("video_buffer_output", "")})
            self.add_flow(source, segmentation_preprocessor, {("video_buffer_output", "")})

        else:
            self.add_flow(source, detection_preprocessor)
            self.add_flow(source, segmentation_preprocessor)

        # connect all pre-processor outputs to the inference operator
        for op in [detection_preprocessor, segmentation_preprocessor]:
            self.add_flow(op, multi_ai_inference, {("", "receivers")})

        # connect the inference output to the postprocessors
        self.add_flow(multi_ai_inference, detection_postprocessor, {("transmitter", "in")})
        self.add_flow(multi_ai_inference, segmentation_postprocessor, {("transmitter", "")})


class Fragment2(Fragment):
    def __init__(self, app, name, label_dict):
        super().__init__(app, name)

        self.label_dict = label_dict

    def compose(self):
        # Holoviz
        holoviz_tensors = [dict(name="", type="color"), dict(name="out_tensor", type="color_lut")]
        if len(self.label_dict) > 0:
            for label in self.label_dict:
                color = self.label_dict[label]["color"]
                color.append(1.0)
                text = [self.label_dict[label]["text"]]
                holoviz_tensors.append(
                    dict(
                        name="rectangles" + str(label),
                        type="rectangles",
                        opacity=0.7,
                        line_width=4,
                        color=color,
                    )
                )
                holoviz_tensors.append(
                    dict(
                        name="label" + str(label), type="text", opacity=0.7, color=color, text=text
                    )
                )
        else:
            holoviz_tensors.append(
                dict(
                    name="rectangles",
                    type="rectangles",
                    opacity=0.7,
                    line_width=4,
                    color=[1.0, 0.0, 0.0, 1.0],
                )
            )
        pool = UnboundedAllocator(self, name="pool")
        holoviz = HolovizOp(
            self, allocator=pool, name="holoviz", tensors=holoviz_tensors, **self.kwargs("holoviz")
        )

        # add operators

        self.add_operator(holoviz)


class MultiAIDetectionSegmentation(Application):
    def __init__(self, data, source="replayer", labelfile=""):
        super().__init__()

        if data == "none":
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")

        # set name
        self.name = "Multi AI App"

        # Optional parameters affecting the graph created by compose.
        source = source.lower()
        if source not in ["replayer", "aja"]:
            raise ValueError(f"unsupported source: {source}. Please use 'replayer' or 'aja'.")
        self.source = source
        self.label_dict = self.get_label_dict(labelfile)
        self.sample_data_path = data

    def compose(self):
        fragment1 = Fragment1(
            self,
            name="fragment1",
            source=self.source,
            sample_data_path=self.sample_data_path,
            label_dict=self.label_dict,
        )
        fragment2 = Fragment2(self, name="fragment2", label_dict=self.label_dict)

        # Connect the two fragments
        # We can skip the "out" and "in" suffixes, as they are the default
        source_output = (
            self.source + ".video_buffer_output"
            if self.source.lower() == "aja"
            else self.source + ".output"
        )
        self.add_flow(fragment1, fragment2, {(source_output, "holoviz.receivers")})
        self.add_flow(fragment1, fragment2, {("detection_postprocessor.out", "holoviz.receivers")})
        self.add_flow(
            fragment1, fragment2, {("segmentation_postprocessor.out_tensor", "holoviz.receivers")}
        )

    def get_label_dict(self, labelfile):
        # construct the labels dictionary if the commandline arg for labelfile isn't empty
        label_dict = {}
        if labelfile != "":
            assert os.path.isfile(labelfile)
            with open(labelfile, newline="") as labelcsv:
                csvreader = csv.reader(labelcsv)
                for row in csvreader:
                    # assume each row looks like: 1, "Grasper", 1.0, 0.0, 1.0
                    label_dict[int(row[0])] = {}
                    label_dict[int(row[0])]["text"] = str(row[1])
                    label_dict[int(row[0])]["color"] = [float(row[2]), float(row[3]), float(row[4])]
        return label_dict


if __name__ == "__main__":

    parser = ArgumentParser(description="Multi-AI Detection Segmentation application.")
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
    parser.add_argument(
        "-l",
        "--labelfile",
        default="none",
        help=(
            "Optional arg for a csv file path containing the class labels. There "
            "should be 5 columns: [label value, label text, red, green, blue] "
            "where label value = the model output value for each class to display and should be int, "
            "label text = the text to display for this label, "
            "red/green/blue should = values between 0 and 1 for the display color "
            "for this class."
        ),
    )
    apps_argv = Application().argv
    args = parser.parse_args(apps_argv[1:])
    if args.config == "none":
        config_file = os.path.join(
            os.path.dirname(__file__),
            "../../../applications/multiai_endoscopy/python/multi_ai.yaml",
        )
    else:
        config_file = args.config

    if args.labelfile == "none":
        labelfile = os.path.join(
            os.path.dirname(__file__),
            "../../../applications/multiai_endoscopy/python/endo_ref_data_labels.csv",
        )
    else:
        labelfile = args.labelfile

    app = MultiAIDetectionSegmentation(data=args.data, source=args.source, labelfile=labelfile)
    app.config(config_file)
    print("sys.argv:", sys.argv)
    print("app.argv:", app.argv)
    app.run()
