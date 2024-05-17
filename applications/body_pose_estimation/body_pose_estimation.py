# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
from argparse import ArgumentParser

import cupy as cp
import holoscan as hs
import numpy as np
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.operators import FormatConverterOp, HolovizOp, InferenceOp, V4L2VideoCaptureOp
from holoscan.resources import UnboundedAllocator


class FormatInferenceInputOp(Operator):
    """Operator to format input image for inference"""

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # Transpose
        tensor = cp.asarray(in_message.get("preprocessed"))
        tensor = cp.moveaxis(tensor, 2, 0)[cp.newaxis]
        # Copy as a contiguous array to avoid issue with strides
        tensor = cp.ascontiguousarray(tensor)

        # Create output message
        op_output.emit(dict(preprocessed=tensor), "out")


class PostprocessorOp(Operator):
    """Operator to post-process inference output:
    * Non-max suppression
    * Make boxes compatible with Holoviz

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Output tensor names
        self.outputs = [
            "boxes",
            "noses",
            "left_eyes",
            "right_eyes",
            "left_ears",
            "right_ears",
            "left_shoulders",
            "right_shoulders",
            "left_elbows",
            "right_elbows",
            "left_wrists",
            "right_wrists",
            "left_hips",
            "right_hips",
            "left_knees",
            "right_knees",
            "left_ankles",
            "right_ankles",
            "segments",
        ]

        # Indices for each keypoint as defined by YOLOv8 pose model
        self.NOSE = slice(5, 7)
        self.LEFT_EYE = slice(8, 10)
        self.RIGHT_EYE = slice(11, 13)
        self.LEFT_EAR = slice(14, 16)
        self.RIGHT_EAR = slice(17, 19)
        self.LEFT_SHOULDER = slice(20, 22)
        self.RIGHT_SHOULDER = slice(23, 25)
        self.LEFT_ELBOW = slice(26, 28)
        self.RIGHT_ELBOW = slice(29, 31)
        self.LEFT_WRIST = slice(32, 34)
        self.RIGHT_WRIST = slice(35, 37)
        self.LEFT_HIP = slice(38, 40)
        self.RIGHT_HIP = slice(41, 43)
        self.LEFT_KNEE = slice(44, 46)
        self.RIGHT_KNEE = slice(47, 49)
        self.LEFT_ANKLE = slice(50, 52)
        self.RIGHT_ANKLE = slice(53, 55)

    def setup(self, spec: OperatorSpec):
        """
        input: "in"    - Input tensors coming from output of inference model
        output: "out"  - The post-processed output after applying thresholding and non-max suppression.
                         Outputs are the boxes, keypoints, and segments.  See self.outputs for the list of outputs.
        params:
            iou_threshold:    Intersection over Union (IoU) threshold for non-max suppression (default: 0.5)
            score_threshold:  Score threshold for filtering out low scores (default: 0.5)
            image_dim:        Image dimensions for normalizing the boxes (default: None)

        Returns:
            None
        """
        spec.input("in")
        spec.output("out")
        spec.param("iou_threshold", 0.5)
        spec.param("score_threshold", 0.5)
        spec.param("image_dim", None)

    def get_keypoints(self, detection):
        # Keypoints to be returned including our own "neck" keypoint
        keypoints = {
            "nose": detection[self.NOSE],
            "left_eye": detection[self.LEFT_EYE],
            "right_eye": detection[self.RIGHT_EYE],
            "left_ear": detection[self.LEFT_EAR],
            "right_ear": detection[self.RIGHT_EAR],
            "neck": (detection[self.LEFT_SHOULDER] + detection[self.RIGHT_SHOULDER]) / 2,
            "left_shoulder": detection[self.LEFT_SHOULDER],
            "right_shoulder": detection[self.RIGHT_SHOULDER],
            "left_elbow": detection[self.LEFT_ELBOW],
            "right_elbow": detection[self.RIGHT_ELBOW],
            "left_wrist": detection[self.LEFT_WRIST],
            "right_wrist": detection[self.RIGHT_WRIST],
            "left_hip": detection[self.LEFT_HIP],
            "right_hip": detection[self.RIGHT_HIP],
            "left_knee": detection[self.LEFT_KNEE],
            "right_knee": detection[self.RIGHT_KNEE],
            "left_ankle": detection[self.LEFT_ANKLE],
            "right_ankle": detection[self.RIGHT_ANKLE],
        }

        return keypoints

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # Convert input to cupy array
        results = cp.asarray(in_message.get("inference_output"))[0]

        # Filter out low scores
        results = results[:, results[4, :] > self.score_threshold]
        scores = results[4, :]

        # If no detections, return zeros for all outputs
        if results.shape[1] == 0:
            out_message = Entity(context)
            zeros = hs.as_tensor(np.zeros([1, 2, 2]).astype(np.float32))

            for output in self.outputs:
                out_message.add(zeros, output)
            op_output.emit(out_message, "out")
            return

        results = results.transpose([1, 0])

        segments = []
        for i, detection in enumerate(results):
            # fmt: off
            kp = self.get_keypoints(detection)
            # Every two points defines a segment
            segments.append([kp["nose"], kp["left_eye"],      # nose <-> left eye
                             kp["nose"], kp["right_eye"],     # nose <-> right eye
                             kp["left_eye"], kp["left_ear"],  # ...
                             kp["right_eye"], kp["right_ear"],
                             kp["left_shoulder"], kp["right_shoulder"],
                             kp["left_shoulder"], kp["left_elbow"],
                             kp["left_elbow"], kp["left_wrist"],
                             kp["right_shoulder"], kp["right_elbow"],
                             kp["right_elbow"], kp["right_wrist"],
                             kp["left_shoulder"], kp["left_hip"],
                             kp["left_hip"], kp["left_knee"],
                             kp["left_knee"], kp["left_ankle"],
                             kp["right_shoulder"], kp["right_hip"],
                             kp["right_hip"], kp["right_knee"],
                             kp["right_knee"], kp["right_ankle"],
                             kp["left_hip"], kp["right_hip"],
                             kp["left_ear"], kp["neck"],
                             kp["right_ear"], kp["neck"],
                             ])
            # fmt: on

        cx, cy, w, h = results[:, 0], results[:, 1], results[:, 2], results[:, 3]
        x1, x2 = cx - w / 2, cx + w / 2
        y1, y2 = cy - h / 2, cy + h / 2

        data = {
            "boxes": cp.asarray(np.stack([x1, y1, x2, y2], axis=-1)).transpose([1, 0]),
            "noses": results[:, self.NOSE],
            "left_eyes": results[:, self.LEFT_EYE],
            "right_eyes": results[:, self.RIGHT_EYE],
            "left_ears": results[:, self.LEFT_EAR],
            "right_ears": results[:, self.RIGHT_EAR],
            "left_shoulders": results[:, self.LEFT_SHOULDER],
            "right_shoulders": results[:, self.RIGHT_SHOULDER],
            "left_elbows": results[:, self.LEFT_ELBOW],
            "right_elbows": results[:, self.RIGHT_ELBOW],
            "left_wrists": results[:, self.LEFT_WRIST],
            "right_wrists": results[:, self.RIGHT_WRIST],
            "left_hips": results[:, self.LEFT_HIP],
            "right_hips": results[:, self.RIGHT_HIP],
            "left_knees": results[:, self.LEFT_KNEE],
            "right_knees": results[:, self.RIGHT_KNEE],
            "left_ankles": results[:, self.LEFT_ANKLE],
            "right_ankles": results[:, self.RIGHT_ANKLE],
            "segments": cp.asarray(segments),
        }
        scores = cp.asarray(scores)

        out = self.nms(data, scores)

        # Rearrange boxes to be compatible with Holoviz
        out["boxes"] = cp.reshape(out["boxes"][None], (1, -1, 2))

        # Create output message
        out_message = Entity(context)
        for output in self.outputs:
            out_message.add(hs.as_tensor(out[output] / self.image_dim), output)
        op_output.emit(out_message, "out")

    def nms(self, inputs, scores):
        """Non-max suppression (NMS)
        Performs non-maximum suppression on input boxes according to their intersection-over-union (IoU).
        Filter out detections where the IoU is >= self.iou_threshold.

        See https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/ for an introduction to non-max suppression.

        Parameters
        ----------
        inputs : dictionary containing boxes, keypoints, and segments
        scores : array (n,)

        Returns
        ----------
        outputs : dictionary containing remaining boxes, keypoints, and segments after non-max supprerssion

        """

        boxes = inputs["boxes"]
        segments = inputs["segments"]

        if len(boxes) == 0:
            return cp.asarray([]), cp.asarray([])

        # Get coordinates
        x0, y0, x1, y1 = boxes[0, :], boxes[1, :], boxes[2, :], boxes[3, :]

        # Area of bounding boxes
        area = (x1 - x0 + 1) * (y1 - y0 + 1)

        # Get indices of sorted scores
        indices = cp.argsort(scores)

        # Output boxes and scores
        boxes_out, segments_out, scores_out = [], [], []

        selected_indices = []

        # Iterate over bounding boxes
        while len(indices) > 0:
            # Get index with highest score from remaining indices
            index = indices[-1]
            selected_indices.append(index)
            # Pick bounding box with highest score
            boxes_out.append(boxes[:, index])
            segments_out.extend(segments[index])
            scores_out.append(scores[index])

            # Get coordinates
            x00 = cp.maximum(x0[index], x0[indices[:-1]])
            x11 = cp.minimum(x1[index], x1[indices[:-1]])
            y00 = cp.maximum(y0[index], y0[indices[:-1]])
            y11 = cp.minimum(y1[index], y1[indices[:-1]])

            # Compute IOU
            width = cp.maximum(0, x11 - x00 + 1)
            height = cp.maximum(0, y11 - y00 + 1)
            overlap = width * height
            union = area[index] + area[indices[:-1]] - overlap
            iou = overlap / union

            # Threshold and prune
            left = cp.where(iou < self.iou_threshold)
            indices = indices[left]

        selected_indices = cp.asarray(selected_indices)

        outputs = {
            "boxes": cp.asarray(boxes_out),
            "segments": cp.asarray(segments_out),
            "noses": inputs["noses"][selected_indices],
            "left_eyes": inputs["left_eyes"][selected_indices],
            "right_eyes": inputs["right_eyes"][selected_indices],
            "left_ears": inputs["left_ears"][selected_indices],
            "right_ears": inputs["right_ears"][selected_indices],
            "left_shoulders": inputs["left_shoulders"][selected_indices],
            "right_shoulders": inputs["right_shoulders"][selected_indices],
            "left_elbows": inputs["left_elbows"][selected_indices],
            "right_elbows": inputs["right_elbows"][selected_indices],
            "left_wrists": inputs["left_wrists"][selected_indices],
            "right_wrists": inputs["right_wrists"][selected_indices],
            "left_hips": inputs["left_hips"][selected_indices],
            "right_hips": inputs["right_hips"][selected_indices],
            "left_knees": inputs["left_knees"][selected_indices],
            "right_knees": inputs["right_knees"][selected_indices],
            "left_ankles": inputs["left_ankles"][selected_indices],
            "right_ankles": inputs["right_ankles"][selected_indices],
        }

        return outputs


class BodyPoseEstimationApp(Application):
    def __init__(self, data, source="v4l2"):
        """Initialize the body pose estimation application"""

        super().__init__()

        # set name
        self.name = "Body Pose Estimation App"
        self.source = source

        if data == "none":
            data = os.path.join(
                os.environ.get("HOLOHUB_DATA_PATH", "../data"), "body_pose_estimation"
            )

        self.sample_data_path = data

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        # Determine if the DDS Publisher is enabled.
        dds_common_args = self.kwargs("dds_common")
        dds_publisher_args = dds_common_args | self.kwargs("dds_publisher")
        enable_dds_publisher = dds_publisher_args["enable"]
        del dds_publisher_args["enable"]

        if self.source == "v4l2":
            source = V4L2VideoCaptureOp(
                self,
                name="v4l2_source",
                allocator=pool,
                **self.kwargs("v4l2_source"),
            )
            source_output = "signal"
        elif self.source == "dds":
            try:
                from holohub.dds_video_subscriber import DDSVideoSubscriberOp
            except ImportError:
                print(
                    "ERROR: Can not import DDSVideoSubscriper module. Please make sure to "
                    "build this application using the '--with dds_video_subscriber' option."
                )
                sys.exit(1)
            dds_source_args = dds_common_args | self.kwargs("dds_source")
            source = DDSVideoSubscriberOp(
                self,
                name="dds_source",
                allocator=pool,
                **dds_source_args,
            )
            source_output = "output"

        format_input = FormatInferenceInputOp(
            self,
            name="transpose",
            pool=pool,
        )

        preprocessor_args = self.kwargs("preprocessor")
        preprocessor = FormatConverterOp(
            self,
            name="preprocessor",
            pool=pool,
            **preprocessor_args,
        )

        inference_args = self.kwargs("inference")
        inference_args["model_path_map"] = {
            "yolo_pose": os.path.join(self.sample_data_path, "yolov8l-pose.onnx")
        }

        inference = InferenceOp(
            self,
            name="inference",
            allocator=pool,
            **inference_args,
        )

        postprocessor_args = self.kwargs("postprocessor")
        postprocessor_args["image_width"] = preprocessor_args["resize_width"]
        postprocessor_args["image_height"] = preprocessor_args["resize_height"]
        postprocessor = PostprocessorOp(
            self,
            name="postprocessor",
            allocator=pool,
            **postprocessor_args,
        )

        holoviz_args = self.kwargs("holoviz")
        if enable_dds_publisher:
            holoviz_args["headless"] = True
            holoviz_args["enable_render_buffer_output"] = True
        holoviz = HolovizOp(self, allocator=pool, name="holoviz", **holoviz_args)

        if enable_dds_publisher:
            try:
                from holohub.dds_video_publisher import DDSVideoPublisherOp
            except ImportError:
                print(
                    "ERROR: Can not import DDSVideoPublisher module. Please make sure to "
                    "build this application using the '--with dds_video_publisher' option."
                )
                sys.exit(1)
            dds_publisher = DDSVideoPublisherOp(
                self,
                name="dds_publisher",
                **dds_publisher_args,
            )

        self.add_flow(source, holoviz, {(source_output, "receivers")})
        self.add_flow(source, preprocessor)
        self.add_flow(preprocessor, format_input)
        self.add_flow(format_input, inference, {("", "receivers")})
        self.add_flow(inference, postprocessor, {("transmitter", "in")})
        self.add_flow(postprocessor, holoviz, {("out", "receivers")})
        if enable_dds_publisher:
            self.add_flow(holoviz, dds_publisher, {("render_buffer_output", "input")})


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Body Pose Estimation Application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["v4l2", "dds"],
        default="v4l2",
        help=(
            "If 'v4l2', uses the v4l2 device specified in the yaml file."
            "If 'dds', uses the DDS video stream configured in the yaml file."
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
        config_file = os.path.join(os.path.dirname(__file__), "body_pose_estimation.yaml")
    else:
        config_file = args.config

    app = BodyPoseEstimationApp(args.data, args.source)
    app.config(config_file)
    app.run()
