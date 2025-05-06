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

import ctypes
import logging
import os
from argparse import ArgumentParser

import cupy as cp
from holoscan.conditions import BooleanCondition, CountCondition
from holoscan.core import Application, IOSpec, Operator, OperatorSpec
from holoscan.operators import (
    BayerDemosaicOp,
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    SegmentationPostprocessorOp,
    VideoStreamRecorderOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import BlockMemoryPool, UnboundedAllocator
from import_utils import lazy_import

from holohub.aja_source import AJASourceOp
from holohub.orsi_format_converter import OrsiFormatConverterOp
from holohub.orsi_segmentation_preprocessor import OrsiSegmentationPreprocessorOp
from operators.deidentification.pixelator import PixelatorOp

cuda = lazy_import("cuda.cuda")
hololink_module = lazy_import("hololink")


class AggregatorOp(Operator):
    """
    This operator is used to aggregate the messages from multiple operators into a single tensor map.
    """

    def setup(self, spec):
        spec.input("in", size=IOSpec.ANY_SIZE)
        spec.output("out")

    def compute(self, op_input, op_output, context):
        in_messages = op_input.receive("in")
        # Combine all tensors into a single dictionary
        out_message = {k: v for message in in_messages for k, v in message.items()}
        op_output.emit(out_message, "out")


class ForwardOp(Operator):
    """
    This operator is used to forward the input to the following operator(s).
    This is specially useful to abstract different possible conditional flows when one flow is broadcasted to multiple operators.
    """

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        op_output.emit(in_message, "out")


class HolovizDelegateOp(Operator):
    """
    This operator receives the input tensors and forwards them to the Holoviz operator.
    It also ensures that all required tensors are present in the input message.
    If any tensor is missing, it is initialized with zeros.
    """

    def __init__(
        self,
        *args,
        holoviz_tensor_names: list[str] | None = None,
        segmentation_shape: tuple[int, int, int] = (1, 1, 1),
        **kwargs,
    ):
        self.holoviz_tensor_names = holoviz_tensor_names or []
        self.segmentation_zeros = cp.zeros(segmentation_shape, dtype=cp.uint8)
        self.detection_zeros = cp.zeros([1, 2, 2], dtype=cp.float32)
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        out_message = op_input.receive("in")
        missing_tensors = set(self.holoviz_tensor_names) - set(out_message.keys())
        if missing_tensors:
            for tensor in missing_tensors:
                if tensor == "out_tensor":
                    out_message[tensor] = self.segmentation_zeros
                else:
                    out_message[tensor] = self.detection_zeros
        op_output.emit(out_message, "out")


class FrameSamplerOp(Operator):
    """
    This operator decimates the input video stream by sampling frames at a specified interval.
    It allows for reducing the frame rate by only forwarding every Nth frame, where N is the interval.
    """

    def __init__(self, *args, interval: int = 1, **kwargs):
        if interval <= 0:
            raise ValueError("The 'interval' parameter must be greater than 0.")
        self.interval = interval
        self.count = 0
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        self.count += 1
        if self.count % self.interval == 0:
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
        # Receive the decision and update the decision attribute
        decision_message = op_input.receive("decision")
        self.decision = decision_message["decision"]
        # Forward the input to the output
        op_output.emit(in_message, "out")


class OutOfBodyPostprocessorOp(Operator):
    """
    This operator is used to postprocess the out of body inference output.
    """

    def __init__(
        self,
        *args,
        in_tensor_name: str = "out_of_body_inferred",
        out_tensor_name: str = "decision",
        **kwargs,
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
        is_out_of_body = out_of_body_inferred.item() > 1
        out_message = {self.out_tensor_name: is_out_of_body}
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
        label_dict: dict | None = None,
        label_text_size: float = 0.05,
        scores_threshold: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.label_text_size = label_text_size
        self.scores_threshold = scores_threshold
        self.label_dict = label_dict or {}

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    @staticmethod
    def append_size_to_text_coord(text_coord: cp.ndarray, size: float) -> cp.ndarray:
        """Appends size information to text coordinates.

        Args:
            text_coord: Array of shape [1, N, 2] containing x,y coordinates
            size: Text size value to append

        Returns:
            Array of shape [1, N, 3] with size appended to each coordinate
        """
        text_size = cp.ones((1, text_coord.shape[1], 1)) * size
        return cp.append(text_coord, text_size, 2).astype(cp.float32)

    def _process_detections(
        self, inferred_bboxes: cp.ndarray, inferred_scores: cp.ndarray, inferred_labels: cp.ndarray
    ):
        """Process detection outputs and filter by confidence threshold.

        Args:
            inferred_bboxes: Bounding box coordinates
            inferred_scores: Detection confidence scores
            inferred_labels: Class labels

        Returns:
            Tuple of filtered bboxes, labels and whether detections exist
        """
        ix = inferred_scores.flatten() >= self.scores_threshold
        has_rect = ix.any()

        filtered_bboxes = inferred_bboxes[:, ix, :]  # shape [1, num_bbox, 4]
        filtered_labels = inferred_labels[:, ix].flatten()  # shape [num_bbox]

        return filtered_bboxes, filtered_labels, has_rect

    def _process_with_labels(
        self, inferred_bboxes: cp.ndarray, inferred_labels: cp.ndarray, has_rect: bool
    ):
        """Process detections when label dictionary is provided.

        Args:
            inferred_bboxes: Filtered bounding boxes
            inferred_labels: Filtered class labels
            has_rect: Whether any detections exist

        Returns:
            Dict of bbox coordinates and text coordinates per label
        """
        bbox_coords = {label: cp.zeros([1, 2, 2], dtype=cp.float32) for label in self.label_dict}
        text_coords = {
            label: cp.zeros([1, 1, 2], dtype=cp.float32) - 1.0 for label in self.label_dict
        }

        if has_rect:
            for label in self.label_dict:
                curr_l_ix = inferred_labels == int(label)
                if curr_l_ix.any():
                    bbox_coords[label] = cp.reshape(inferred_bboxes[0, curr_l_ix, :], (1, -1, 2))
                    text_coords[label] = self.append_size_to_text_coord(
                        cp.reshape(inferred_bboxes[0, curr_l_ix, :2], (1, -1, 2)),
                        self.label_text_size,
                    )

        return bbox_coords, text_coords

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")

        # Get detection outputs
        inferred_bboxes = cp.asarray(in_message["inference_output_detection_boxes"]).get()
        inferred_scores = cp.asarray(in_message["inference_output_detection_scores"]).get()
        inferred_labels = cp.asarray(in_message["inference_output_detection_classes"]).get()

        # Process and filter detections
        bboxes, labels, has_rect = self._process_detections(
            inferred_bboxes, inferred_scores, inferred_labels
        )

        # Prepare output message
        out_message = {}

        if self.label_dict:
            # Process with label categories
            bbox_coords, text_coords = self._process_with_labels(bboxes, labels, has_rect)
            for label in self.label_dict:
                out_message[f"rectangles{label}"] = bbox_coords[label]
                out_message[f"label{label}"] = text_coords[label]
        else:
            # Single category output
            bbox_coords = (
                cp.reshape(bboxes, (1, -1, 2))
                if has_rect
                else cp.zeros([1, 2, 2], dtype=cp.float32)
            )
            out_message["rectangles"] = bbox_coords

        op_output.emit(out_message, "out")


class AISurgicalVideoWorkflow(Application):
    """
    Real-Time End-to-End AI Surgical Video Workflow

    This class defines the workflow for processing surgical video streams with AI models.
    It supports three different sources:
        - Holoscan Sensor Bridge (HSB)
        - AJA Card
        - Video Replayer
    The workflow is composed of the following overall components:
        - Out-of-Body Detection
        - Conditional Flow Control
        - Multi-AI Surgical Instrument Detection and Segmentation
        - Deidentification
        - Visualization.
    """

    def __init__(
        self,
        source=None,
        data=None,
        headless=False,
        fullscreen=False,
        cuda_context=None,
        cuda_device_ordinal=None,
        hololink_channel=None,
        ibv_name=None,
        ibv_port=None,
        camera=None,
        camera_mode=None,
        frame_limit=None,
        recording_dir=None,
        recording_basename="ai_surgical_video_output",
        recording_frame_interval=1,
    ):
        super().__init__()
        # Set application name
        self.name = "Real-Time End-to-End AI Surgical Video Workflow"
        # Validate the path to the data directory
        self.data_dir = data if data is not None else os.environ.get("HOLOHUB_DATA_PATH", "../data")
        if not self.data_dir or not os.path.exists(self.data_dir):
            raise ValueError(f"Data path does not exist: {self.data_dir}")
        # Validate source
        self.source = source.lower() if source is not None else "replayer"
        self.supported_sources = ["replayer", "aja", "hsb"]

        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._ibv_name = ibv_name
        self._ibv_port = ibv_port
        self._camera = camera
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._recording_dir = recording_dir
        self._recording_basename = recording_basename
        self._enable_recording = self._recording_dir is not None
        self._recording_frame_interval = recording_frame_interval

    def compose(self):
        logging.info("Setup source and camera")
        # Memory allocator for some operators
        pool = UnboundedAllocator(self, name="pool")
        logging.info(f"{self.source=}")
        # ------------------------------------------------------------------------------------------
        # Configure AJA capture card
        # ------------------------------------------------------------------------------------------
        if self.source == "aja":
            in_dtype = "rgba8888"
            aja = AJASourceOp(self, name="aja_source", **self.kwargs("aja"))
        # ------------------------------------------------------------------------------------------
        # Setup video replayer
        # ------------------------------------------------------------------------------------------
        elif self.source.startswith("replayer"):
            replayer_kwargs = self.kwargs("replayer")
            in_dtype = "rgb888"
            # Prifix the video directory with the data directory and validate
            video_dir = os.path.join(self.data_dir, replayer_kwargs["directory"])
            if not os.path.exists(video_dir):
                raise ValueError(f"Video directory not found: {video_dir}")
            replayer_kwargs["directory"] = video_dir
            if self._frame_limit is not None:
                replayer_kwargs["count"] = self._frame_limit
            if self._enable_recording:
                replayer_kwargs["realtime"] = False
            replayer = VideoStreamReplayerOp(self, name="video_replayer", **replayer_kwargs)
        # ------------------------------------------------------------------------------------------
        # Setup Holoscan Sensor Bridge
        # ------------------------------------------------------------------------------------------
        elif self.source == "hsb":
            in_dtype = "rgb888"

            if self._frame_limit:
                self._count = CountCondition(self, name="count", count=self._frame_limit)
                condition = self._count
            else:
                self._ok = BooleanCondition(self, name="ok", enable_tick=True)
                condition = self._ok
            self._camera.set_mode(self._camera_mode)

            # Create the CSI to bayer converter.
            csi_to_bayer_pool = BlockMemoryPool(
                self,
                name="pool",
                # storage_type of 1 is device memory
                storage_type=1,
                block_size=self._camera._width
                * ctypes.sizeof(ctypes.c_uint16)
                * self._camera._height,
                num_blocks=2,
            )
            hsb_csi_to_bayer = hololink_module.operators.CsiToBayerOp(
                self,
                name="csi_to_bayer",
                allocator=csi_to_bayer_pool,
                cuda_device_ordinal=self._cuda_device_ordinal,
            )
            # The call to camera.configure(...) earlier set our image dimensions
            # and bytes per pixel. This call asks the camera to configure the
            # converter accordingly.
            self._camera.configure_converter(hsb_csi_to_bayer)

            # hsb_csi_to_bayer now knows the image dimensions and bytes per pixel,
            # and can compute the overall size of the received image data.
            frame_size = hsb_csi_to_bayer.get_csi_length()
            frame_context = self._cuda_context

            # Create a receiver object that fills out our frame buffer. The receiver
            # operator knows how to configure hololink_channel to send its data
            # to us and to provide an end-of-frame indication at the right time.
            hsb_receiver = hololink_module.operators.RoceReceiverOp(
                self,
                condition,
                name="receiver",
                frame_size=frame_size,
                frame_context=frame_context,
                ibv_name=self._ibv_name,
                ibv_port=self._ibv_port,
                hololink_channel=self._hololink_channel,
                device=self._camera,
            )
            bayer_format = self._camera.bayer_format()
            pixel_format = self._camera.pixel_format()
            hsb_image_processor = hololink_module.operators.ImageProcessorOp(
                self,
                name="image_processor",
                # Optical black value for imx274 is 50
                optical_black=50,
                bayer_format=bayer_format.value,
                pixel_format=pixel_format.value,
            )

            rgb_components_per_pixel = 3
            bayer_pool = BlockMemoryPool(
                self,
                name="pool",
                # storage_type of 1 is device memory
                storage_type=1,
                block_size=self._camera._width
                * rgb_components_per_pixel
                * ctypes.sizeof(ctypes.c_uint16)
                * self._camera._height,
                num_blocks=2,
            )
            hsb_demosaic = BayerDemosaicOp(
                self,
                name="demosaic",
                pool=bayer_pool,
                generate_alpha=False,
                bayer_grid_pos=bayer_format.value,
                interpolation_mode=0,
            )

            hsb_image_shift = hololink_module.operators.ImageShiftToUint8Operator(
                self, name="image_shift", shift=8
            )
        else:
            raise ValueError(
                f"Unsupported source: {self.source}. Please use {' or '.join(self.supported_sources)}."
            )

        # ------------------------------------------------------------------------------------------
        # Out of Body Detection
        # ------------------------------------------------------------------------------------------
        # Preprocessor: ensures correct format for inference
        # Orsi operators
        out_of_body_format_converter = OrsiFormatConverterOp(
            self,
            name="out_of_body_format_converter",
            allocator=pool,
            **self.kwargs("out_of_body_format_converter"),
        )
        out_of_body_normalizer = OrsiSegmentationPreprocessorOp(
            self,
            name="out_of_body_normalizer",
            allocator=pool,
            **self.kwargs("out_of_body_normalizer"),
        )
        out_of_body_inference = InferenceOp(
            self,
            name="out_of_body_inference",
            allocator=pool,
            model_path_map={
                "out_of_body": os.path.join(
                    self.data_dir, "orsi", "models", "anonymization_model.onnx"
                )
            },
            **self.kwargs("out_of_body_inference"),
        )
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
        holoviz_tensors = [
            dict(name="", type="color"),  # source image
            dict(name="out_tensor", type="color_lut"),  # segmentation output
        ]
        # Add label-specific tensors for each detection label
        label_dict = detection_postprocessor.label_dict
        rectangle_defaults = {"type": "rectangles", "opacity": 0.7, "line_width": 4}
        text_defaults = {"type": "text", "opacity": 0.7}
        if label_dict:
            for label, label_info in label_dict.items():
                # Prepare color with alpha
                color = label_info["color"] + [1.0]
                # Make text label a list
                text = [label_info["text"]]
                # Add rectangle tensor
                holoviz_tensors.append(
                    {"name": f"rectangles{label}", "color": color, **rectangle_defaults}
                )
                # Add text label tensor
                holoviz_tensors.append(
                    {"name": f"label{label}", "color": color, "text": text, **text_defaults}
                )
        else:
            # Add default red rectangle tensor if no labels
            holoviz_tensors.append(
                {**rectangle_defaults, "name": "rectangles", "color": [1.0, 0.0, 0.0, 1.0]}
            )
        # Holoviz operators for visualization
        segmentation_shape = (
            self.kwargs("segmentation_preprocessor")["resize_height"],
            self.kwargs("segmentation_preprocessor")["resize_width"],
            1,
        )
        holoviz_delegate = HolovizDelegateOp(
            self,
            name="holoviz_delegate_op",
            holoviz_tensor_names=[tensor["name"] for tensor in holoviz_tensors],
            segmentation_shape=segmentation_shape,
        )
        holoviz = HolovizOp(
            self,
            allocator=pool,
            name="holoviz",
            tensors=holoviz_tensors,
            fullscreen=self._fullscreen,
            headless=self._headless,
            enable_render_buffer_output=self._enable_recording,
            **self.kwargs("holoviz"),
        )
        # ------------------------------------------------------------------------------------------
        # Recording
        # ------------------------------------------------------------------------------------------
        if self._enable_recording:
            # Convert the Holoviz output frames
            recorder_format_converter = FormatConverterOp(
                self,
                name="recorder_format_converter_op",
                in_dtype="rgba8888",
                out_dtype="rgb888",
                pool=UnboundedAllocator(self, name="recorder_pool"),
            )
            # Decimate the input frames
            frame_sampler = FrameSamplerOp(
                self, name="frame_sampler_op", interval=self._recording_frame_interval
            )
            # Record frames to PNG files
            recorder = VideoStreamRecorderOp(
                self,
                name="recorder_op",
                directory=self._recording_dir,
                basename=self._recording_basename,
            )

        # ------------------------------------------------------------------------------------------
        # Auxiliary operators
        # ------------------------------------------------------------------------------------------
        # Source operator
        source = ForwardOp(self, name="source_op")
        # Conditional operator
        condition = ConditionOp(self, name="condition_op")
        # Broadcaster operator
        broadcaster = ForwardOp(self, name="broadcaster_op")
        # Postprocessor aggregator operator
        postprocessor_aggregator = AggregatorOp(self, name="postprocessor_aggregator_op")

        # Dynamic flow callback
        def dynamic_flow_callback(op):
            if op.decision:
                op.add_dynamic_flow(deidentification)
            else:
                op.add_dynamic_flow(broadcaster)

        # ------------------------------------------------------------------------------------------
        # Deidentification
        # ------------------------------------------------------------------------------------------
        deidentification = PixelatorOp(
            self,
            name="deidentification_op",
            **self.kwargs("deidentification"),
        )
        # ------------------------------------------------------------------------------------------
        # Create the pipeline
        # -----------------------------------------------------------------------------------------
        # Set the source
        if self.source == "hsb":
            self.add_flow(hsb_receiver, hsb_csi_to_bayer, {("output", "input")})
            self.add_flow(hsb_csi_to_bayer, hsb_image_processor, {("output", "input")})
            self.add_flow(hsb_image_processor, hsb_demosaic, {("output", "receiver")})
            self.add_flow(hsb_demosaic, hsb_image_shift, {("transmitter", "input")})
            self.add_flow(hsb_image_shift, source, {("output", "in")})
        elif self.source == "aja":
            self.add_flow(aja, source, {("video_buffer_output", "in")})
        else:
            self.add_flow(replayer, source)
        # __________________________________________________________________
        # Main Branch
        # Out of body detection application
        self.add_flow(source, out_of_body_format_converter)
        self.add_flow(out_of_body_format_converter, out_of_body_normalizer)
        self.add_flow(out_of_body_normalizer, out_of_body_inference, {("", "receivers")})
        self.add_flow(out_of_body_inference, out_of_body_postprocessor, {("transmitter", "in")})
        # Feed the source and out of body detection decision to the conditional operator
        self.add_flow(out_of_body_postprocessor, condition, {("out", "decision")})
        self.add_flow(source, condition, {("out", "in")})
        # __________________________________________________________________
        # Dynamic flow condition based on conditional operator
        self.add_flow(condition, deidentification, {("out", "in")})
        self.add_flow(condition, broadcaster, {("out", "in")})
        self.set_dynamic_flows(condition, dynamic_flow_callback)
        # __________________________________________________________________
        # Branch 1: rest of deidentification application
        # connect the deidentification output to the holoviz delegate
        self.add_flow(deidentification, holoviz_delegate)
        # __________________________________________________________________
        # Branch 2: rest of multi-ai detection and segmentation application
        self.add_flow(broadcaster, detection_preprocessor, {("out", "")})
        self.add_flow(broadcaster, segmentation_preprocessor, {("out", "")})
        self.add_flow(broadcaster, postprocessor_aggregator, {("out", "in")})
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
        # ------------------------------------------------------------------------------------------
        # Recording
        # ------------------------------------------------------------------------------------------
        if self._enable_recording:
            self.add_flow(
                holoviz, recorder_format_converter, {("render_buffer_output", "source_video")}
            )
            self.add_flow(recorder_format_converter, frame_sampler)
            self.add_flow(frame_sampler, recorder)
        # ------------------------------------------------------------------------------------------


def main(args):
    # __________________________________________________________________
    # Set up the sensor bridge device
    if args.source == "hsb":
        # Get handles to GPU
        cuda.cuInit(0)
        cu_device_ordinal = 0
        _, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
        _, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)

        # Get a handle to the data source
        channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
        logging.info(f"{channel_metadata=}")
        hololink_channel = hololink_module.DataChannel(channel_metadata)

        # Now that we can communicate, create the camera controller
        camera = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
            hololink_channel, expander_configuration=args.expander_configuration
        )
        camera_mode = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode(args.camera_mode)

        # If the InfiniBand device is not specified, use the first available InfiniBand device
        if args.ibv_name is None:
            args.ibv_name = hololink_module.infiniband_devices()[0]

        # __________________________________________________________________
        # Set up our Holoscan pipeline
        application = AISurgicalVideoWorkflow(
            source=args.source,
            data=args.data,
            headless=args.headless,
            fullscreen=args.fullscreen,
            cuda_context=cu_context,
            cuda_device_ordinal=cu_device_ordinal,
            hololink_channel=hololink_channel,
            ibv_name=args.ibv_name,
            ibv_port=args.ibv_port,
            camera=camera,
            camera_mode=camera_mode,
            frame_limit=args.frame_limit,
            recording_dir=args.recording_dir,
            recording_frame_interval=args.recording_frame_interval,
        )
        application.config(args.config)

        # __________________________________________________________________
        # Connect and initialize the sensor bridge device
        hololink = hololink_channel.hololink()
        hololink.start()  # Establish a connection to the sensor bridge device
        if not args.skip_reset:
            hololink.reset()  # Drive the sensor bridge to a known state
        if args.ptp_sync:
            ptp_sync_timeout_s = 10
            ptp_sync_timeout = hololink_module.Timeout(ptp_sync_timeout_s)
            logging.debug("Waiting for PTP sync.")
            if not hololink.ptp_synchronize(ptp_sync_timeout):
                logging.error(
                    f"Failed to synchronize PTP after {ptp_sync_timeout_s} seconds; ignoring."
                )
            else:
                logging.debug("PTP synchronized.")

        # __________________________________________________________________
        # Configure the camera
        if not args.skip_reset:
            camera.setup_clock()
        camera.configure(camera_mode)
        camera.set_digital_gain_reg(0xF)
        if args.pattern is not None:
            camera.test_pattern(args.pattern)

        # __________________________________________________________________
        # Run our Holoscan pipeline
        logging.info("Calling run")
        application.is_metadata_enabled = False  # disable metadata
        application.run()  # we don't usually return from this call.

        # __________________________________________________________________
        # Clean up the sensor bridge device
        hololink.stop()
        cuda.cuDevicePrimaryCtxRelease(cu_device)

    else:
        application = AISurgicalVideoWorkflow(
            source=args.source,
            data=args.data,
            headless=args.headless,
            fullscreen=args.fullscreen,
            frame_limit=args.frame_limit,
            recording_dir=args.recording_dir,
            recording_frame_interval=args.recording_frame_interval,
        )
        application.config(args.config)
        application.run()


if __name__ == "__main__":
    parser = ArgumentParser(description="Real-Time End-to-End AI Surgical Video Workflow")
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer", "aja", "hsb"],
        default="replayer",
        help="""
        If 'replayer', replay a prerecorded video. 
        If 'aja' use an AJA capture card as the source. 
        If 'hsb' use the Holoscan Sensor Bridge as the source.""",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Set config path to override the default config file location",
    )
    parser.add_argument(
        "-d",
        "--data",
        help="Set the path the data directory. If not provided, use the HOLOHUB_DATA_PATH environment variable.",
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--fullscreen", action="store_true", help="Run in fullscreen mode")
    parser.add_argument(
        "--camera-mode",
        type=int,
        default=0,
        help="Camera mode to use [0,1,2,3]",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "--ibv-name",
        help="IBV device to use",
    )
    parser.add_argument(
        "--ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device",
    )
    parser.add_argument(
        "--expander-configuration",
        type=int,
        default=0,
        choices=(0, 1),
        help="I2C Expander configuration",
    )
    parser.add_argument(
        "--pattern",
        type=int,
        choices=range(12),
        help="Configure to display a test pattern.",
    )
    parser.add_argument(
        "--ptp-sync",
        action="store_true",
        help="After reset, wait for PTP time to synchronize.",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Don't call reset on the hololink device",
    )
    parser.add_argument(
        "--recording-dir",
        type=str,
        default=None,
        help="Directory to save the recording",
    )
    parser.add_argument(
        "--recording-basename",
        type=str,
        default="ai_surgical_video_output",
        help="Basename of the recording",
    )
    parser.add_argument(
        "--recording-frame-interval",
        type=int,
        default=1,
        help="Recording frames interval",
    )
    args = parser.parse_args()

    main(args)
