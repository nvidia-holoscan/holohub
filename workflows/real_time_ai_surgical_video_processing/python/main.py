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

import csv
import os
from argparse import ArgumentParser
from typing import Dict

import cupy as cp
import numpy as np
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    SegmentationPostprocessorOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import UnboundedAllocator

from holohub.aja_source import AJASourceOp

# Constants
DEFAULT_LABEL_TEXT_SIZE = 0.05
DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_BBOX_LINE_WIDTH = 4
DEFAULT_OPACITY = 0.7


class HubOp(Operator):
    """
    This operator is used to conditionally pass the input to the output based on the condition.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        op_output.emit(in_message, "out")


class ConditionalOp(Operator):
    """
    This operator is used to conditionally pass the input to the output based on the condition.
    """

    def __init__(self, *args, **kwargs):
        self.decision = False
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.input("decision")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        decision_message = op_input.receive("decision")
        self.decision = decision_message["decision"]
        if self.decision:
            print("**** Inside Body ****")
        else:
            print("**** Outside Body ****")
        op_output.emit(in_message, "out")


class OutOfBodyPostprocessorOp(Operator):
    """
    This operator is used to postprocess the out of body inference output.
    """

    def __init__(self, *args, in_tensor_name: str = "out_of_body_inferred", out_tensor_name: str = "decision", **kwargs):
        self.in_tensor_name = in_tensor_name
        self.out_tensor_name = out_tensor_name
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        out_of_body_inferred = cp.array(in_message[self.in_tensor_name])
        is_out_of_body = cp.argmax(out_of_body_inferred).item() == 0
        out_message = {self.out_tensor_name: is_out_of_body}
        op_output.emit(out_message, "out")


class DeidentificationOp(Operator):
    """
    This operator is used to deidentify the input image.
    """

    def __init__(self, *args, block_size_h: int = 16, block_size_w: int = 16, **kwargs):
        self.block_size_h = block_size_h
        self.block_size_w = block_size_w
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        image = cp.asarray(in_message[""])

        # Pixelate the image by downsampling and upsampling
        h, w = image.shape[:2]
        small_h = h // self.block_size_h
        small_w = w // self.block_size_w

        # Reshape and mean across blocks to downsample
        reshaped = image.reshape(small_h, self.block_size_h, small_w, self.block_size_w, -1)
        downsampled = cp.mean(reshaped, axis=(1, 3))

        # Repeat each pixel to upsample back to original size
        upsampled = cp.repeat(cp.repeat(downsampled, self.block_size_h, axis=0), self.block_size_w, axis=1)

        # Ensure output matches input dimensions and type
        image = upsampled[:h, :w]
        image = image.astype(np.uint8)

        out_message = {"": image}
        op_output.emit(out_message, "out")


class DetectionPostprocessorOp(Operator):
    """Post-processes detection inference outputs to prepare visualization data.

    This operator processes bounding boxes, scores, and class labels from an object
    detection model and prepares them for visualization with Holoviz.

    Args:
        label_dict: Dictionary mapping class IDs to label information
        label_text_size: Size of the label text to display
        scores_threshold: Confidence threshold for filtering detections
    """

    def __init__(
        self,
        *args,
        label_dict: Dict = {},
        label_text_size: float = DEFAULT_LABEL_TEXT_SIZE,
        scores_threshold: float = DEFAULT_SCORE_THRESHOLD,
        **kwargs,
    ):
        self.label_text_size = label_text_size
        self.scores_threshold = scores_threshold
        self.label_dict = label_dict

        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def append_size_to_text_coord(self, text_coord: np.ndarray, size: float) -> np.ndarray:
        """Appends size information to text coordinates.

        Args:
            text_coord: Array of shape [1, N, 2] containing x,y coordinates
            size: Text size value to append

        Returns:
            Array of shape [1, N, 3] with size appended to each coordinate
        """
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


class Workflow(Application):
    def __init__(self, data: str, source: str = "replayer", labelfile: str = ""):
        super().__init__()

        # Validate inputs
        if data == "none":
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")
            if not os.path.exists(data):
                raise ValueError(f"Data path does not exist: {data}")

        # set name
        self.name = "Real-Time AI Surgical Video Processing"

        # Optional parameters affecting the graph created by compose.
        self.source = source.lower()
        if self.source not in ["replayer", "aja"]:
            raise ValueError(f"unsupported source: {self.source}. Please use 'replayer' or 'aja'.")
        self.labelfile = labelfile
        self.sample_data_path = data

    def _parse_label_file(self, labelfile: str) -> dict:
        """Parses the label CSV file into a dictionary.

        Args:
            labelfile: Path to CSV file containing label information

        Returns:
            Dictionary mapping label IDs to label information

        Raises:
            ValueError: If CSV file format is invalid
        """
        label_dict = {}
        with open(labelfile, newline="") as labelcsv:
            csvreader = csv.reader(labelcsv)
            for row in csvreader:
                try:
                    label_id = int(row[0])
                    label_dict[label_id] = {"text": str(row[1]), "color": [float(row[2]), float(row[3]), float(row[4])]}
                except (IndexError, ValueError):
                    raise ValueError(
                        "Label file must have 5 columns: label ID (int), "
                        "text (str), red (float), green (float), blue (float)"
                    )
        return label_dict

    def compose(self):
        # override source with config file
        self.source = self.kwargs("source")["source"]

        # Validate label file
        if self.labelfile:
            if not os.path.isfile(self.labelfile):
                raise FileNotFoundError(f"Label file not found: {self.labelfile}")

            try:
                label_dict = self._parse_label_file(self.labelfile)
            except (csv.Error, ValueError) as e:
                raise ValueError(f"Failed to parse label file: {e}")
        else:
            label_dict = {}

        # start constructing app
        # Configure video source (AJA capture card or video replay)
        is_aja = self.source == "aja"  # Already lowercase from __init__
        source_kwargs = self.kwargs(self.source)

        if is_aja:
            source = AJASourceOp(self, name="aja_source", **source_kwargs)
        else:
            # For replayer, validate and set video directory
            if self.source == "replayer":
                video_dir = os.path.join(self.sample_data_path, "endoscopy")
            elif self.source == "replayer2":
                video_dir = os.path.join(self.sample_data_path, "endoscopy_out_of_body_detection")
            else:
                raise ValueError(f"Unsupported source: {self.source}")
            if not os.path.exists(video_dir):
                raise ValueError(f"Video directory not found: {video_dir}")
            source_kwargs["directory"] = video_dir
            source = VideoStreamReplayerOp(self, name="video_replayer", **source_kwargs)

        # Memory allocator for some operators
        pool = UnboundedAllocator(self, name="pool")
        # input format for the preprocessors
        in_dtype = "rgba8888" if is_aja else "rgb888"

        # Preprocessors: ensures correct format for inference
        out_of_body_preprocessor = FormatConverterOp(
            self,
            name="out_of_body_preprocessor",
            pool=pool,
            in_dtype=in_dtype,
            **self.kwargs("out_of_body_preprocessor"),
        )
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

        # Inference: Out of body detection
        inference_kwargs = self.kwargs("out_of_body_inference")
        inference_kwargs["model_path_map"] = {
            "out_of_body": os.path.join(
                self.sample_data_path, "endoscopy_out_of_body_detection", "out_of_body_detection.onnx"
            )
        }
        out_of_body_inference = InferenceOp(self, name="out_of_body_inference", allocator=pool, **inference_kwargs)

        # Inference: Multi-AI
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

        # Post-processing
        out_of_body_postprocessor = OutOfBodyPostprocessorOp(
            self, name="out_of_body_postprocessor", **self.kwargs("out_of_body_postprocessor")
        )
        detection_postprocessor = DetectionPostprocessorOp(
            self,
            name="detection_postprocessor",
            label_dict=label_dict,
            allocator=pool,
            **self.kwargs("detection_postprocessor"),
        )
        segmentation_postprocessor = SegmentationPostprocessorOp(
            self,
            name="segmentation_postprocessor",
            allocator=pool,
            **self.kwargs("segmentation_postprocessor"),
        )

        # Holoviz tensors for visualization
        holoviz_tensors = [dict(name="", type="color"), dict(name="out_tensor", type="color_lut")]
        if len(label_dict) > 0:
            for label in label_dict:
                color = label_dict[label]["color"]
                color.append(1.0)
                text = [label_dict[label]["text"]]
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
                    dict(name="label" + str(label), type="text", opacity=0.7, color=color, text=text)
                )
        else:
            holoviz_tensors.append(
                dict(
                    name="rectangles",
                    type="rectangles",
                    opacity=DEFAULT_OPACITY,
                    line_width=DEFAULT_BBOX_LINE_WIDTH,
                    color=[1.0, 0.0, 0.0, 1.0],
                )
            )

        # Holoviz operator for visualization
        holoviz = HolovizOp(self, allocator=pool, name="holoviz", tensors=holoviz_tensors, **self.kwargs("holoviz"))
        condition = ConditionalOp(self, name="conditional_op")  # conditional operator
        hub = HubOp(self, name="hub_op")  # pass output of one operator to multiple operators
        deidentification = DeidentificationOp(self, name="deidentification_op", **self.kwargs("deidentification"))

        # ------------------------------------------------------------
        # Create the pipeline
        # ------------------------------------------------------------
        # Main Branch: out of body detection application
        self.add_flow(source, out_of_body_preprocessor)
        self.add_flow(out_of_body_preprocessor, out_of_body_inference, {("", "receivers")})
        self.add_flow(out_of_body_inference, out_of_body_postprocessor, {("transmitter", "in")})

        # Feed the image and decision to the conditional operator
        self.add_flow(out_of_body_postprocessor, condition, {("out", "decision")})
        self.add_flow(source, condition, {("", "in")})

        # Create dynamic flow condition based on conditional oprator
        self.add_flow(condition, deidentification, {("out", "in")})
        self.add_flow(condition, hub, {("out", "in")})

        def dynamic_flow_callback(op):
            if op.decision:
                op.add_dynamic_flow(deidentification)
            else:
                op.add_dynamic_flow(hub)
        self.set_dynamic_flows(condition, dynamic_flow_callback)

        # Branch 1: deidentification application
        self.add_flow(deidentification, holoviz, {("out", "receivers")})

        # # Branch 2: multi-ai detection and segmentation application
        self.add_flow(hub, detection_preprocessor, {("out", "")})
        self.add_flow(hub, segmentation_preprocessor, {("out", "")})
        self.add_flow(hub, holoviz, {("out", "receivers")})

        # connect all pre-processor outputs to the inference operator
        self.add_flow(detection_preprocessor, multi_ai_inference, {("", "receivers")})
        self.add_flow(segmentation_preprocessor, multi_ai_inference, {("", "receivers")})

        # connect the inference output to the postprocessors
        self.add_flow(multi_ai_inference, detection_postprocessor, {("transmitter", "in")})
        self.add_flow(multi_ai_inference, segmentation_postprocessor, {("transmitter", "")})

        # prepare postprocessed output for visualization with holoviz
        self.add_flow(detection_postprocessor, holoviz, {("out", "receivers")})
        self.add_flow(segmentation_postprocessor, holoviz, {("", "receivers")})


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-AI Detection Segmentation application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer1", "replayer2", "aja"],
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
    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "multi_ai.yaml")
    else:
        config_file = args.config

    if args.labelfile == "none":
        labelfile = os.path.join(os.path.dirname(__file__), "endo_ref_data_labels.csv")
    else:
        labelfile = args.labelfile

    app = Workflow(source=args.source, data=args.data, labelfile=labelfile)
    app.config(config_file)
    app.run()
