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
from typing import Dict, Optional

import cupy as cp
import numpy as np
from holoscan.core import Application, IOSpec, Operator, OperatorSpec
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    SegmentationPostprocessorOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import UnboundedAllocator

from holohub.aja_source import AJASourceOp


class AggregatorOp(Operator):
    """
    This operator is used to aggregate the messages form multiple operators into a single tensor map.
    """

    def setup(self, spec):
        spec.input("in", size=IOSpec.ANY_SIZE)
        spec.output("out")

    def compute(self, op_input, op_output, context):
        in_messages = op_input.receive("in")

        # Combine all tensors into a single dictionary with a dictionary comprehension
        out_message = {k: v for message in in_messages for k, v in message.items()}
        op_output.emit(out_message, "out")


class ForwardOp(Operator):
    """
    This operator is used to forward the input to the output operator(s).
    """

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        op_output.emit(in_message, "out")


class ConditionOp(Operator):
    """
    This operator set the dynamic flow condition based on the input decision and forward the input image.
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
        # receive the decision and update the decision attribute
        decision_message = op_input.receive("decision")
        self.decision = decision_message["decision"]
        if self.decision:
            print("**** Outside Body ****")
        else:
            print("**** Inside Body ****")
        # forward the input to the output
        op_output.emit(in_message, "out")


class OutOfBodyPostprocessorOp(Operator):
    """
    This operator is used to postprocess the out of body inference output.
    """

    def __init__(
        self, *args, in_tensor_name: str = "out_of_body_inferred", out_tensor_name: str = "decision", **kwargs
    ):
        self.in_tensor_name = in_tensor_name
        self.out_tensor_name = out_tensor_name
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        out_of_body_inferred = cp.array(in_message[self.in_tensor_name])
        is_out_of_body = cp.argmax(out_of_body_inferred).item() == 1
        out_message = {self.out_tensor_name: is_out_of_body}
        op_output.emit(out_message, "out")


class DeIdentificationOp(Operator):
    """
    This operator is used to deidentify the input image.
    """

    def __init__(
        self,
        *args,
        detection_labels=None,
        segmentation_shape=(1, 1, 1),
        block_size_h: int = 16,
        block_size_w: int = 16,
        **kwargs,
    ):
        self.block_size_h = block_size_h
        self.block_size_w = block_size_w
        self.detection_labels = detection_labels if detection_labels is not None else []
        self.segmentation_shape = segmentation_shape
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
        image = image.astype(cp.uint8)
        # add the holoviz tensors to the output message
        out_message = {}
        for label in self.detection_labels:
            out_message[ "rectangles" + str(label)] = cp.zeros([1, 2, 2], dtype=cp.float32)
            out_message["label" + str(label)] = -1.0 * cp.ones([1, 1, 2], dtype=cp.float32)
        out_message["out_tensor"] = cp.zeros(self.segmentation_shape, dtype=cp.uint8)
        out_message[""] = image
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
        label_text_size: float = 0.05,
        scores_threshold: float = 0.3,
        **kwargs,
    ):
        self.label_text_size = label_text_size
        self.scores_threshold = scores_threshold
        self.label_dict = label_dict

        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def append_size_to_text_coord(self, text_coord: cp.ndarray, size: float) -> cp.ndarray:
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
        text_size = cp.ones((1, text_coord.shape[1], 1)) * size
        new_text_coord = cp.append(text_coord, text_size, 2)
        return new_text_coord.astype(cp.float32)

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

        bbox_coords = cp.zeros([1, 2, 2], dtype=cp.float32)

        if len(self.label_dict) > 0:
            # the label file isn't empty, we want to colorize the bbox and text colors
            bbox_coords = {}
            text_coords = {}
            for label in self.label_dict:
                bbox_coords[label] = cp.zeros([1, 2, 2], dtype=cp.float32)
                # coords tensor for text to display in Holoviz can be of shape [1, n, 2] or [1, n, 3]
                # with each location having [x,y] coords or [x,y,s] coords where s = size of text
                # to display
                text_coords[label] = cp.zeros([1, 1, 2], dtype=cp.float32) - 1.0

            if has_rect:
                # there are bboxes and we want to colorize them as well as label text
                for label in self.label_dict:
                    curr_l_ix = output_labels == int(label)
                    if curr_l_ix.any():
                        bbox_coords[label] = cp.reshape(output_bboxes[0, curr_l_ix, :], (1, -1, 2))
                        text_coords[label] = self.append_size_to_text_coord(
                            cp.reshape(output_bboxes[0, curr_l_ix, :2], (1, -1, 2)),
                            self.label_text_size,
                        )

        else:
            # the label file is empty, just display bboxes in one color
            if has_rect:
                bbox_coords = cp.reshape(output_bboxes, (1, -1, 2))

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
    def __init__(self, data: Optional[str] = None, source: Optional[str] = None):
        super().__init__()
        # Set application name
        self.name = "Real-Time AI Surgical Video Processing"
        # Validate the path to the data directory
        self.data_dir = data if data is not None else os.environ.get("HOLOHUB_DATA_PATH", "../data")
        if not self.data_dir or not os.path.exists(self.data_dir):
            raise ValueError(f"Data path does not exist: {self.data_dir}")
        # Validate source
        self.source = source.lower() if source is not None else "replayer"
        self.supported_sources = ["replayer", "aja"]

    def compose(self):
        # ------------------------------------------------------------------------------------------
        # Configure video source (AJA capture card or video replay)
        # -----------------------------------------------------------------------------------------
        # override source with config file and validate
        if self.kwargs("source"):
            self.source = self.kwargs("source")["name"].lower()
        print(f"### {self.name} - source: {self.source}")
        # Set source operator
        source_kwargs = self.kwargs(self.source)
        if self.source == "aja":
            in_dtype = "rgba8888"
            source = AJASourceOp(self, name="aja_source", **source_kwargs)
        elif self.source.startswith("replayer"):
            in_dtype = "rgb888"
            # Prifix the video directory with the data directory and validate
            video_dir = os.path.join(self.data_dir, source_kwargs["directory"])
            if not os.path.exists(video_dir):
                raise ValueError(f"Video directory not found: {video_dir}")
            source_kwargs["directory"] = video_dir
            source = VideoStreamReplayerOp(self, name="video_replayer", **source_kwargs)
        else:
            raise ValueError(f"Unsupported source: {self.source}. Please use {' or '.join(self.supported_sources)}.")
        # ------------------------------------------------------------------------------------------
        # Memory allocator for some operators
        # ------------------------------------------------------------------------------------------
        pool = UnboundedAllocator(self, name="pool")

        # ------------------------------------------------------------------------------------------
        # Out of Body Detection
        # ------------------------------------------------------------------------------------------
        # Preprocessor: ensures correct format for inference
        out_of_body_preprocessor = FormatConverterOp(
            self,
            name="out_of_body_preprocessor",
            pool=pool,
            in_dtype=in_dtype,
            **self.kwargs("out_of_body_preprocessor"),
        )
        # Inference: Out of body detection
        inference_kwargs = self.kwargs("out_of_body_inference")
        inference_kwargs["model_path_map"] = {
            "out_of_body": os.path.join(self.data_dir, "endoscopy_out_of_body_detection", "out_of_body_detection.onnx")
        }
        out_of_body_inference = InferenceOp(self, name="out_of_body_inference", allocator=pool, **inference_kwargs)
        # Postprocessor: postprocesses the out of body inference output to a decision
        out_of_body_postprocessor = OutOfBodyPostprocessorOp(
            self, name="out_of_body_postprocessor", **self.kwargs("out_of_body_postprocessor")
        )
        # ------------------------------------------------------------------------------------------
        # Multi-AI Detection and Segmentation
        # ------------------------------------------------------------------------------------------
        # Preprocessor: ensures correct format for inference
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
        # Inference: Multi-AI Detection and Segmentation
        model_path_map = {
            "ssd": os.path.join(self.data_dir, "ssd_model", "epoch24_nms.onnx"),
            "tool_seg": os.path.join(
                self.data_dir,
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
        # Post-processors: detection and segmentation
        detection_postprocessor = DetectionPostprocessorOp(
            self,
            name="detection_postprocessor",
            allocator=pool,
            **self.kwargs("detection_postprocessor"),
        )
        segmentation_postprocessor = SegmentationPostprocessorOp(
            self,
            name="segmentation_postprocessor",
            allocator=pool,
            **self.kwargs("segmentation_postprocessor"),
        )
        # ------------------------------------------------------------------------------------------
        # Holoviz
        # ------------------------------------------------------------------------------------------
        # Holoviz tensors for visualization
        label_dict = detection_postprocessor.label_dict
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
                    opacity=0.7,
                    line_width=4,
                    color=[1.0, 0.0, 0.0, 1.0],
                )
            )
        # Holoviz operators for visualization
        holoviz_delegate = ForwardOp(self, name="holoviz_delegate_op")
        holoviz = HolovizOp(self, allocator=pool, name="holoviz", tensors=holoviz_tensors, **self.kwargs("holoviz"))
        # ------------------------------------------------------------------------------------------
        # Auxiliary operators
        # ------------------------------------------------------------------------------------------
        # Conditional operator
        condition = ConditionOp(self, name="condition_op")
        # Distributor operator
        distributor = ForwardOp(self, name="distributor_op")
        # Postprocessor aggregator operator
        postprocessor_aggregator = AggregatorOp(self, name="postprocessor_aggregator_op")

        # Dynamic flow callback
        def dynamic_flow_callback(op):
            if op.decision:
                op.add_dynamic_flow(deidentification)
            else:
                op.add_dynamic_flow(distributor)

        # ------------------------------------------------------------------------------------------
        # Deidentification
        # ------------------------------------------------------------------------------------------
        segmentation_shape = (
            self.kwargs("segmentation_preprocessor")["resize_height"],
            self.kwargs("segmentation_preprocessor")["resize_width"],
            1
        )
        deidentification = DeIdentificationOp(
            self,
            name="deidentification_op",
            detection_labels=list(label_dict.keys()),
            segmentation_shape=segmentation_shape,
            **self.kwargs("deidentification"),
        )
        # ------------------------------------------------------------------------------------------
        # Create the pipeline
        # -----------------------------------------------------------------------------------------
        # _______________________________________________
        # Main Branch: out of body detection application
        self.add_flow(source, out_of_body_preprocessor)
        self.add_flow(out_of_body_preprocessor, out_of_body_inference, {("", "receivers")})
        self.add_flow(out_of_body_inference, out_of_body_postprocessor, {("transmitter", "in")})
        # Feed the source and out of body detection decision to the conditional operator
        self.add_flow(out_of_body_postprocessor, condition, {("out", "decision")})
        self.add_flow(source, condition, {("", "in")})
        # ___________________________________________________________
        # Create dynamic flow condition based on conditional oprator
        self.add_flow(condition, deidentification, {("out", "in")})
        self.add_flow(condition, distributor, {("out", "in")})
        self.set_dynamic_flows(condition, dynamic_flow_callback)
        # _______________________________________________
        # Branch 1: rest of deidentification application
        # connect the deidentification output to the holoviz delegate
        self.add_flow(deidentification, holoviz_delegate)
        # __________________________________________________________________
        # Branch 2: rest of multi-ai detection and segmentation application
        self.add_flow(distributor, detection_preprocessor, {("out", "")})
        self.add_flow(distributor, segmentation_preprocessor, {("out", "")})
        self.add_flow(distributor, postprocessor_aggregator, {("out", "in")})
        # connect all pre-processor outputs to the inference operator
        self.add_flow(detection_preprocessor, multi_ai_inference, {("", "receivers")})
        self.add_flow(segmentation_preprocessor, multi_ai_inference, {("", "receivers")})
        # connect the inference output to the postprocessors
        self.add_flow(multi_ai_inference, detection_postprocessor, {("transmitter", "in")})
        self.add_flow(multi_ai_inference, segmentation_postprocessor, {("transmitter", "")})
        # connect postprocessed output to the postprocessor aggregator
        self.add_flow(detection_postprocessor, postprocessor_aggregator, {("out", "in")})
        self.add_flow(segmentation_postprocessor, postprocessor_aggregator, {("", "in")})
        # connect the postprocessor aggregator to the holoviz delegate
        self.add_flow(postprocessor_aggregator, holoviz_delegate)
        # ____________________________________________________________________
        # Branch 1&2: connect the holoviz delegate to the holoviz operator
        self.add_flow(holoviz_delegate, holoviz, {("out", "receivers")})


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-AI Detection Segmentation application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer", "aja"],
        default="replayer",
        help="If 'replayer', replay a prerecorded video. If 'aja' use an AJA capture card as the source.",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Set config path to override the default config file location",
    )
    parser.add_argument(
        "-d",
        "--data",
        help="Set the path the data directory. If not provided, use the HOLOHUB_DATA_PATH environment variable.",
    )
    args = parser.parse_args()

    if args.config:
        config_file = args.config
    else:
        config_file = os.path.join(os.path.dirname(__file__), "config.yaml")

    app = Workflow(source=args.source, data=args.data)
    app.config(config_file)
    app.run()
