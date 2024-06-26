import datetime

import cupy as cp
import cupyx.scipy.ndimage
import holoscan as hs
import numpy as np
from holoscan.core import Operator, OperatorSpec
from holoscan.gxf import Entity
from utils import CupyArrayPainter, DecoderInputData, PointMover, save_cupy_tensor


class DecoderConfigurator(Operator):
    def __init__(self, *args, save_intermediate=False, verbose=False, **kwargs):
        super().__init__(*args, **kwargs)
        if "input_point" in kwargs:
            input_point = kwargs["input_point"]
        else:
            input_point = None
        self.verbose = verbose
        self.save_intermediate = save_intermediate
        # Output tensor names
        self.outputs = [
            "image_embeddings",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
        ]
        self.viz_outputs = ["point_coords"]
        self.decoder_input = DecoderInputData.create_decoder_inputs_from(
            dtype=np.float32, input_point=input_point
        )
        print(f"created inputs {self.decoder_input}")
        print(self.decoder_input)
        self.decoder_input.point_coords = DecoderInputData.scale_coords(
            self.decoder_input.point_coords,
            orig_height=1024,
            orig_width=1024,
            resized_height=1024,
            resized_width=1024,
            dtype=np.float32,
        )
        print(f"after scaling {self.decoder_input}")
        self.decoder_input.orig_im_size = np.array([1024, 1024], dtype=np.float32)
        print("---------------------init Decoder Config complete")

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.input("point_in")
        spec.output("out")
        spec.output("point")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        in_point = op_input.receive("point_in")
        in_point = cp.asarray(in_point.get("point_coords"), order="C").get()[0]
        # update the point in the decoder input
        self.decoder_input.point_coords, self.decoder_input.point_labels = (
            DecoderInputData.point_coords(point=in_point)
        )

        image_tensor = cp.asarray(in_message.get("image_embeddings"), order="C")
        if self.save_intermediate:
            # save the image embeddings
            save_cupy_tensor(
                folder_path="applications/segment_everything/downloads/numpy",
                tensor=image_tensor,
                word="image_embeddings",
                verbose=self.verbose,
            )

        if self.verbose:
            print(image_tensor.shape, image_tensor.dtype)
            print(self.decoder_input)
            # Get input message
            print(in_message)
            print(in_message.get("image_embeddings"))
        data = {
            "image_embeddings": image_tensor,
            "point_coords": cp.asarray(self.decoder_input.point_coords, order="C"),
            "point_labels": cp.asarray(self.decoder_input.point_labels, order="C"),
            "mask_input": cp.asarray(self.decoder_input.mask_input, order="C"),
            "has_mask_input": cp.asarray(self.decoder_input.has_mask_input, order="C"),
        }
        # deep copy the point_coords
        copy_point_coords = cp.copy(data["point_coords"])
        # choose the first point
        copy_point_coords = copy_point_coords[0, 0, :]
        copy_point_coords = DecoderInputData.scale_coords(
            copy_point_coords,
            orig_height=1024,
            orig_width=1024,
            resized_height=1,
            resized_width=1,
            dtype=np.float32,
        )
        # Create output message
        out_message = Entity(context)
        for i, output in enumerate(self.outputs):
            out_message.add(hs.as_tensor(data[output]), output)
        op_output.emit(out_message, "out")

        # create output for the point_coords
        point_message = Entity(context)
        point_message.add(hs.as_tensor(copy_point_coords), "point_coords")
        op_output.emit(point_message, "point")


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
        tensor = self.normalize_image(tensor)
        # reshape
        tensor = np.moveaxis(tensor, 2, 0)[np.newaxis]
        tensor = cp.asarray(tensor, order="C", dtype=cp.float32)
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


class SamPostprocessorOp(Operator):
    """Operator to post-process inference output:"""

    def __init__(self, *args, save_intermediate=False, verbose=False, **kwargs):
        super().__init__(*args, **kwargs)
        # Output tensor names
        self.outputs = ["out_tensor"]
        self.slice_dim = 3
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
        results = cp.asarray(in_message.get("low_res_masks"))
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
        if self.verbose:
            print(f"unpadded tensor {unpadded_tensor.shape}\n")
            print(unpadded_tensor.flags)

        if self.slice_dim is not None:
            unpadded_tensor = unpadded_tensor[:, self.slice_dim, :, :]
            unpadded_tensor = cp.expand_dims(unpadded_tensor, 1).astype(cp.float32)

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
        unpadded_tensor = self.painter.to_rgba(unpadded_tensor)
        # make array ccontiguous
        unpadded_tensor = cp.ascontiguousarray(unpadded_tensor)

        out_message = Entity(context)
        for output in self.outputs:
            out_message.add(hs.as_tensor(unpadded_tensor), output)
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


class PointPublisher(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = datetime.datetime.now()
        point_mover_kwargs = kwargs["point_mover"]
        self.point_mover = PointMover(**point_mover_kwargs[0])

    def setup(self, spec: OperatorSpec):
        spec.output("out")

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
