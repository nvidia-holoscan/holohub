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

import numpy as np
from holoscan.core import Operator, OperatorSpec
from holoscan.operators import HolovizOp


class DetectionPostprocessorOp(Operator):
    def __init__(self, *args, label_text_size=0.1, **kwargs):
        """
        Initialize the DetectionPostprocessorOp.

        Args:
            *args: Additional arguments.
            label_text_size (float): The size of the label text.
            **kwargs: Additional keyword arguments.
        """
        self.label_text_size = label_text_size
        self.label_color_map = {}  # Map to store unique colors for labels
        self.caption_tasks = {"<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"}
        self.bbox_tasks = {
            "<OD>",
            "<OPEN_VOCABULARY_DETECTION>",
            "<REGION_PROPOSAL>",
            "<DENSE_REGION_CAPTION>",
            "<CAPTION_TO_PHRASE_GROUNDING>",
            "<OCR_WITH_REGION>",
        }
        self.bbox_map = {
            "<OD>": "bboxes",
            "<OPEN_VOCABULARY_DETECTION>": "bboxes",
            "<REGION_PROPOSAL>": "bboxes",
            "<DENSE_REGION_CAPTION>": "bboxes",
            "<CAPTION_TO_PHRASE_GROUNDING>": "bboxes",
            "<OCR_WITH_REGION>": "quad_boxes",
        }
        self.segmentation_task = {"<REFERRING_EXPRESSION_SEGMENTATION>"}
        self.labeling_tasks = {
            "<OD>",
            "<OPEN_VOCABULARY_DETECTION>",
            "<DENSE_REGION_CAPTION>",
            "<CAPTION_TO_PHRASE_GROUNDING>",
            "<OCR_WITH_REGION>",
        }
        self.labeling_map = {
            "<OD>": "labels",
            "<OPEN_VOCABULARY_DETECTION>": "bboxes_labels",
            "<DENSE_REGION_CAPTION>": "labels",
            "<CAPTION_TO_PHRASE_GROUNDING>": "labels",
            "<REGION_PROPOSAL>": "labels",
            "<OCR_WITH_REGION>": "labels",
        }
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """
        Define the operator's inputs and outputs.
        """
        spec.input("input")
        spec.input("video_frame")
        spec.output("output_specs")
        spec.output("outputs")

    def get_unique_color(self, label):
        """
        Get a unique color for a label.

        Args:
            label (str): Label to get a color for.

        Returns:
            list: Color as an RGBA list.
        """
        if label not in self.label_color_map:
            hue = len(self.label_color_map) * 0.618033988749895  # Golden ratio
            hue = hue % 1  # Ensure hue is in [0, 1]
            color = self.hsv_to_rgb(hue, 0.75, 0.75)
            self.label_color_map[label] = color
        return self.label_color_map[label]

    def create_multiline_text(self, text, max_line_length=100):
        """
        Convert a single line of text to multiple lines to fit within a specified length.
        """
        wrapped_caption = ""
        needs_newline = False
        for i, c in enumerate(text):
            if i % max_line_length == 0 and i != 0:
                needs_newline = True
            if c == " " and needs_newline:
                wrapped_caption += "\n"
                needs_newline = False
            wrapped_caption += c
        return wrapped_caption

    def hsv_to_rgb(self, h, s, v):
        """
        Convert HSV color to RGB color.

        Args:
            h (float): Hue.
            s (float): Saturation.
            v (float): Value.

        Returns:
            list: RGB color as a list.
        """
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        r, g, b = [
            (v, t, p),
            (q, v, p),
            (p, v, t),
            (p, q, v),
            (t, p, v),
            (v, p, q),
        ][i % 6]
        return [r, g, b, 1.0]  # Return as RGBA

    def compute(self, op_input, op_output, context):
        """
        Compute method to process the input data and generate outputs.
        """
        # Receive inputs
        input = op_input.receive("input")
        video_frame = op_input.receive("video_frame").get("")
        height, width = video_frame.shape[:2]
        data = input["output"]
        task = input["task"]

        # Initialize lists for detection data
        bboxes = []
        labels = []
        polygons = []
        caption = ""

        # Process the task-specific data
        if task in self.caption_tasks:
            caption = data[task]
        if task in self.bbox_tasks:
            bboxes = data[task][self.bbox_map[task]]
            labels = data[task][self.labeling_map[task]]
        if task in self.segmentation_task:
            polygons = data[task]["polygons"]
        if task == "<OCR>":
            caption = data[task]

        # Convert lists to numpy arrays for Holoviz
        output_bboxes = np.array(bboxes, dtype=np.float32)
        output_labels = np.array(labels)

        bbox_coords_map = {}
        text_coords_map = {}

        specs = []

        # This allows for multiple instances of the same label to be displayed
        identifiers = {}
        # This allows for the color of identical labels to be the same
        color_identifier = {}

        # Process each label and bounding box
        for label, bbox in zip(output_labels, output_bboxes):
            if label == "":
                label = "None"
            if label in identifiers:
                identifiers[label] += 1
            else:
                identifiers[label] = 0

            identifier = f"{label}{identifiers[label] if identifiers[label] > 0 else ''}"
            color_identifier[identifier] = label

            bbox_coords_map[identifier] = np.reshape(bbox, (1, -1, 2))
            if task in self.labeling_tasks:
                text_coords_map[identifier] = np.reshape(bbox, (1, -1, 2))

        # Create output message
        out_message = {}

        # Handle any text labels
        for label in text_coords_map:
            x, y = bbox_coords_map[label][0][0][0], bbox_coords_map[label][0][0][1]
            # Normalize to [0, 1]
            x /= width
            y /= height
            dynamic_text = []
            dynamic_text.append((x, y))
            out_message[label] = np.asarray(dynamic_text).astype(np.float32)

            # Dynamic specs for text labels
            spec = HolovizOp.InputSpec(label, "text")
            spec.text = [label]
            spec.color = self.get_unique_color(color_identifier[label])
            specs.append(spec)

        # Handle bbox outputs
        for label in bbox_coords_map:
            # Get top left and lower right of bbox as tuple
            top_left = (bbox_coords_map[label][0][0][0], bbox_coords_map[label][0][0][1])
            lower_right = (bbox_coords_map[label][0][1][0], bbox_coords_map[label][0][1][1])
            # Normalize to [0, 1]
            top_left = (top_left[0] / width, top_left[1] / height)
            lower_right = (lower_right[0] / width, lower_right[1] / height)
            out_message[label + "_bbox"] = np.asarray([top_left, lower_right]).astype(np.float32)

            # Dynamic specs for bounding boxes
            spec = HolovizOp.InputSpec(label + "_bbox", "rectangles")
            spec.color = self.get_unique_color(color_identifier[label])
            spec.line_width = 2
            specs.append(spec)

        # Handle captioning labels
        if caption != "":
            out_message["florence2_caption"] = np.asarray(
                [
                    (0.0, 0.0),
                ],
            ).astype(np.float32)

            # Dynamic specs for captions
            spec = HolovizOp.InputSpec("florence2_caption", "text")
            wrapped_caption = self.create_multiline_text(caption)
            spec.text = [wrapped_caption]
            spec.color = [1.0, 1.0, 1.0, 1.0]
            specs.append(spec)

        # Handle segmentation labels by drawing the segmentation with triangles
        for polygon_array in polygons:
            for i, polygon in enumerate(polygon_array):
                _polygon = np.array(polygon).reshape(-1, 2)
                if len(_polygon) < 3:
                    print("Invalid polygon:", _polygon)
                    continue
                root_x = _polygon[0][0] / width
                root_y = _polygon[0][1] / height
                circular_polygon = np.append(_polygon, [_polygon[0]], axis=0)
                triangle_coords = []
                for i in range(len(circular_polygon[:-1])):
                    x1 = circular_polygon[i][0] / width
                    y1 = circular_polygon[i][1] / height
                    x2 = circular_polygon[i + 1][0] / width
                    y2 = circular_polygon[i + 1][1] / height
                    triangle_coords.append((root_x, root_y))
                    triangle_coords.append((x1, y1))
                    triangle_coords.append((x2, y2))

                out_message[f"seg_triangle{i}"] = np.asarray(triangle_coords).astype(np.float32)
                spec = HolovizOp.InputSpec(f"seg_triangle{i}", "triangles")
                spec.color = self.get_unique_color("florence_segmentation")
                spec.line_width = 1
                specs.append(spec)

        # Emit the output message and specifications
        op_output.emit(out_message, "outputs")
        op_output.emit(specs, "output_specs")
