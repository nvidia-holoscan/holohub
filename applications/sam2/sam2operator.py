# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import datetime

import cupy as cp
import cupyx.scipy.ndimage
import holoscan as hs
import numpy as np
import torch
from holoscan.core import Operator, OperatorSpec
from holoscan.gxf import Entity
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils import CupyArrayPainter, DecoderInputData, PointMover, save_cupy_tensor


class SAM2Operator(Operator):
    """Operator to perform inference using the SAM2 SAM2ImagePredictor model"""

    def __init__(self, *args, checkpoint_path, model_cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = ImagePredictorProcessor(checkpoint_path, model_cfg)

    def setup(self, spec: OperatorSpec):
        # input port for the image tensor
        spec.input("image")
        spec.input("point_coords")
        # output port for the masks, scores, and logits
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # get the image tensor from the input port
        image_message = op_input.receive("image")
        # convert to a cupy array
        image = cp.asarray(image_message.get("encoder_tensor"), order="C").get()[0]
        # cast to uint8 again
        # image = image.astype(cp.uint8)
        # transpose the image tensor from (C, H, W) to (H, W, C)
        image = image.transpose(1, 2, 0)

        # Set input point and label
        input_point = cp.asarray(op_input.receive("point_coords").get("point_coords"))
        # input_point = np.array([[500, 375]])
        input_label = np.array([1])

        # Compute masks, scores, and logits
        masks, scores, logits = self.processor.compute(image, input_point, input_label)
        # dimension of tensors must be at least 2 for each tensor, so add a new axis to scores
        scores = scores[:, np.newaxis]
        # Add a batch dimension to masks in the first axis
        masks = masks[np.newaxis]
        logits = logits[np.newaxis]

        # publish the masks, scores, and logits to the output port
        data = {
            "masks": cp.ascontiguousarray(cp.asarray(masks)),
            "scores": cp.ascontiguousarray(cp.asarray(scores)),
            "logits": cp.ascontiguousarray(cp.asarray(logits)),
        }
        out_message = Entity(context)
        for key, value in data.items():
            out_message.add(hs.as_tensor(value), key)
        op_output.emit(out_message, "out")


class ImagePredictorProcessor:
    """Wrapper around the SAM2ImagePredictor class"""

    def __init__(self, checkpoint_path, model_cfg, device="cuda"):
        self.model = build_sam2(
            model_cfg, checkpoint_path, device=device, apply_postprocessing=False
        )
        self.predictor = SAM2ImagePredictor(self.model)

        # use bfloat16 for the entire notebook
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def compute(self, image, point_coords, point_labels):
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        return masks, scores, logits


class SamPostprocessorOp(Operator):
    """Operator to post-process inference output:"""

    def __init__(
        self,
        *args,
        out_tensor,
        save_intermediate=False,
        verbose=False,
        slice_dim: int = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Output tensor names
        self.outputs = out_tensor
        self.slice_dim = slice_dim
        self.transpose_tuple = None
        self.threshold = None
        self.cast_to_uint8 = False
        self.counter = 0
        self.painter = CupyArrayPainter()
        self.save_intermediate = save_intermediate
        self.verbose = verbose

    def setup(self, spec: OperatorSpec):
        """
        input: "in"    - Input tensors coming from output of inference model
        output: "out"  -

        Returns:
            None
        """
        spec.input("in")
        spec.output("out")

    def mask_to_rgba(self, tensor, channel_dim=-1, color=None):
        """convert a tensor of shape (1, 1, 1024, 1024) to a tensor of shape (1, 3, 1024, 1024) by repeating the tensor along the channel dimension
        assuming the input tensor is a mask tensor, containing 0s and 1s.
        set a color for the mask, yellow by default.
        if color has length 3, it will be converted to a 4 channel tensor by adding 255 as the last channel
        the last number in the color tuple is the alpha channel

        Args:
            tensor (_type_): tensor with mask
            channel_dim (int, optional): dimension of the channels. Defaults to -1.
            color (tuple, optional): color for display. Defaults to (255, 255, 0).

        Returns:
            _type_: _description_
        """
        # check that the length of the color is 4
        if color is None:
            color = cp.array([255, 255, 0, 128], dtype=cp.uint8)
        assert len(color) == 4, "Color should be a tuple of length 4"
        tensor = cp.concatenate([tensor] * 4, axis=channel_dim)
        tensor[tensor == 1] = color
        return tensor

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")
        # Convert input to cupy array
        # in SAM2 the masks are binary.
        results = cp.asarray(in_message.get("masks"))
        logits = cp.asarray(in_message.get("logits"))
        scores = cp.asarray(in_message.get("scores"))
        max_score_index = cp.argmax(scores)
        if self.save_intermediate:
            save_cupy_tensor(
                folder_path="applications/segment_everything/downloads/numpy",
                tensor=results,
                counter=self.counter,
                word="low_res_masks",
                verbose=self.verbose,
            )
        if self.verbose:
            print(results.flags)
            print("-------------------postprocessing")
            print(type(results))
            print(results.shape)

        # scale the tensor
        scaled_tensor = self.scale_tensor_with_aspect_ratio(results, 1024)
        scaled_logits = self.scale_tensor_with_aspect_ratio(logits, 1024)
        if self.verbose:
            print(f"Scaled tensor {scaled_tensor.shape}\n")
            print(scaled_tensor.flags)
        if self.save_intermediate:
            save_cupy_tensor(
                folder_path="applications/segment_everything/downloads/numpy",
                tensor=scaled_tensor,
                counter=self.counter,
                word="scaled",
                verbose=self.verbose,
            )

        # undo padding
        unpadded_tensor = self.undo_pad_on_tensor(scaled_tensor, (1024, 1024)).astype(cp.float32)
        unpadded_logits = self.undo_pad_on_tensor(scaled_logits, (1024, 1024)).astype(cp.float32)
        if self.verbose:
            print(f"unpadded tensor {unpadded_tensor.shape}\n")
            print(unpadded_tensor.flags)

        self.slice_dim = max_score_index
        unpadded_tensor = unpadded_tensor[:, self.slice_dim, :, :]
        unpadded_tensor = cp.expand_dims(unpadded_tensor, 1).astype(cp.float32)
        unpadded_logits = unpadded_logits[:, self.slice_dim, :, :]
        unpadded_logits = cp.expand_dims(unpadded_logits, 1).astype(cp.float32)

        if self.save_intermediate:
            save_cupy_tensor(
                folder_path="applications/segment_everything/downloads/numpy",
                tensor=unpadded_tensor,
                counter=self.counter,
                word="sliced",
                verbose=self.verbose,
            )

        if self.transpose_tuple is not None:
            unpadded_tensor = cp.transpose(unpadded_tensor, self.transpose_tuple).astype(cp.float32)
            unpadded_logits = cp.transpose(unpadded_logits, self.transpose_tuple).astype(cp.float32)
            if self.save_intermediate:
                save_cupy_tensor(
                    folder_path="applications/segment_everything/downloads/numpy",
                    tensor=unpadded_tensor,
                    counter=self.counter,
                    word="transposed",
                    verbose=self.verbose,
                )

        # threshold the tensor
        if self.threshold is not None:
            unpadded_tensor = cp.where(unpadded_tensor > self.threshold, 1, 0).astype(cp.float32)
            unpadded_logits = cp.where(unpadded_logits > self.threshold, 1, 0).astype(cp.float32)
            if self.save_intermediate:
                save_cupy_tensor(
                    folder_path="applications/segment_everything/downloads/numpy",
                    tensor=unpadded_tensor,
                    counter=self.counter,
                    word="thresholded",
                    verbose=self.verbose,
                )

        # cast to uint8 datatype
        if self.cast_to_uint8:
            print(unpadded_tensor.flags)
            unpadded_tensor = cp.asarray(unpadded_tensor, dtype=cp.uint8)
            unpadded_logits = cp.asarray(unpadded_logits, dtype=cp.uint8)
            if self.save_intermediate:
                save_cupy_tensor(
                    folder_path="applications/segment_everything/downloads/numpy",
                    tensor=unpadded_tensor,
                    counter=self.counter,
                    word="casted",
                    verbose=self.verbose,
                )

        if self.verbose:
            print(
                f"unpadded_tensor tensor, casted to {unpadded_tensor.dtype} and shape {unpadded_tensor.shape}\n"
            )

        # save the cupy tensor
        if self.save_intermediate:
            save_cupy_tensor(
                folder_path="applications/segment_everything/downloads/numpy",
                tensor=unpadded_tensor,
                counter=self.counter,
                word="unpadded",
                verbose=self.verbose,
            )

        # Create output message
        # create tensor with 3 dims for vis, by squeezing the tensor in the batch dimension
        unpadded_tensor = cp.squeeze(unpadded_tensor, axis=(0, 1))
        unpadded_logits = cp.squeeze(unpadded_logits, axis=(0, 1))
        unpadded_tensor = self.painter.to_rgba(unpadded_tensor)
        unpadded_logits = self.painter.to_rgba(unpadded_logits)
        # make array ccontiguous
        unpadded_tensor = cp.ascontiguousarray(unpadded_tensor)
        unpadded_logits = cp.ascontiguousarray(unpadded_logits)
        output_dict = {"masks": unpadded_tensor, "logits": unpadded_logits}

        out_message = Entity(context)
        for output in self.outputs:
            out_message.add(hs.as_tensor(output_dict[output]), output)
        op_output.emit(out_message, "out")

    def scale_tensor_with_aspect_ratio(self, tensor, max_size, order=1):
        # assumes tensor dimension (batch, height, width)
        height, width = tensor.shape[-2:]
        aspect_ratio = width / height
        if width > height:
            new_width = max_size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(new_height * aspect_ratio)

        scale_factors = (new_height / height, new_width / width)
        # match the rank of the scale_factors to the tensor rank
        scale_factors = (1,) * (tensor.ndim - 2) + scale_factors
        # resize the tensor to the new shape using cupy
        scaled_tensor = cupyx.scipy.ndimage.zoom(tensor, scale_factors, order=order)

        return scaled_tensor

    def undo_pad_on_tensor(self, tensor, original_shape):
        if isinstance(tensor, cp.ndarray):
            # get number of dimensions
            n_dims = tensor.ndim
        else:
            n_dims = tensor.dim()
        width, height = original_shape[:2]
        # unpad the tensor
        if n_dims == 4:
            unpadded_tensor = tensor[:, :, :height, :width]
        elif n_dims == 3:
            unpadded_tensor = tensor[:, :height, :width]
        else:
            raise ValueError("Invalid tensor dimension")
        return unpadded_tensor


class FormatInferenceInputOp(Operator):
    """Operator to format input image for inference"""

    def __init__(
        self, *args, mean=None, std=None, save_intermediate=False, verbose=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.mean = mean
        self.std = std
        if self.mean is None:
            self.mean = cp.array([123.675, 116.28, 103.53])
        if self.std is None:
            self.std = cp.array([58.395, 57.12, 57.375])
        self.save_intermediate = save_intermediate

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")
        if self.verbose:
            print("----------------------------------Inference Input")
            print(in_message)
            print(in_message.get("preprocessed"))

        # Transpose
        tensor = cp.asarray(in_message.get("preprocessed"))
        # Normalize
        # tensor = self.normalize_image(tensor)
        # convert to numpy array
        tensor = tensor.get()

        # to RGB
        tensor = Image.fromarray(tensor)
        tensor = tensor.convert("RGB")
        tensor = np.array(tensor)
        # The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
        #   with pixel values in [0, 255].
        # reshape
        tensor = np.moveaxis(tensor, 2, 0)[np.newaxis]
        tensor = cp.asarray(tensor, order="C", dtype=cp.uint8)
        tensor = cp.ascontiguousarray(tensor)

        # saving input
        if self.save_intermediate:
            save_cupy_tensor(
                folder_path="applications/segment_everything/downloads/numpy",
                tensor=tensor,
                word="input",
                verbose=self.verbose,
            )
        if self.verbose:
            print(f"---------------------------reformatted tensor shape: {tensor.shape}")

        # Create output message
        op_output.emit(dict(encoder_tensor=tensor), "out")

    def normalize_image(self, image):
        image = (image - self.mean) / self.std
        return image


class PointPublisher(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = datetime.datetime.now()
        point_mover_kwargs = kwargs["point_mover"]
        self.point_mover = PointMover(**point_mover_kwargs[0])

    def setup(self, spec: OperatorSpec):
        spec.output("out")
        spec.output("point_viz")

    def compute(self, op_input, op_output, context):
        # Get current time
        current_time = datetime.datetime.now()
        # Calculate time difference
        time_diff = current_time - self.start_time
        # as seconds and microseconds
        time_since_start = time_diff.seconds + time_diff.microseconds / 1e6
        # Get position of the point
        position = self.point_mover.get_position(time_since_start)
        # Create output message
        out_message = Entity(context)
        out_message.add(hs.as_tensor(cp.array([position], dtype=cp.float32)), "point_coords")
        op_output.emit(out_message, "out")

        # Create output message for visualization
        # the point_coords are scaled to the range 0-1
        position_array = np.array([position])
        position = DecoderInputData.scale_coords(
            position_array,
            orig_height=1024,
            orig_width=1024,
            resized_height=1,
            resized_width=1,
            dtype=np.float32,
        )
        point_viz_message = Entity(context)
        point_viz_message.add(hs.as_tensor(position), "point_coords")
        op_output.emit(point_viz_message, "point_viz")
