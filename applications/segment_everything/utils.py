import math
import os
from copy import deepcopy

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np


def save_cupy_tensor(tensor, folder_path, counter=0, word="", verbose=False):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f"{word}_{counter}.npy")
    cp.save(file_path, tensor)
    if verbose:
        print(f"Saved tensor to {file_path} \n")
        print(f"tensor dtype is {tensor.dtype}")


class DecoderInputData:
    def __init__(
        self,
        image_embeddings=None,
        point_coords=None,
        point_labels=None,
        mask_input=None,
        has_mask_input=None,
        orig_im_size=None,
        dtype=np.float32,
    ):

        self.image_embeddings = image_embeddings
        self.point_coords = point_coords
        self.point_labels = point_labels
        self.mask_input = mask_input
        self.has_mask_input = has_mask_input
        self.orig_im_size = orig_im_size
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"DecoderInputData(image_embeddings={self.image_embeddings}, point_coords={self.point_coords}, point_labels={self.point_labels}, mask_input={self.mask_input}, has_mask_input={self.has_mask_input}, orig_im_size={self.orig_im_size}), dtype={self.dtype})"

    @staticmethod
    def point_coords(point=None, label=None):
        if point is None:
            point = (500, 500)
        if label is None:
            label = 1
        input_point = np.array([point], dtype=np.float32)
        input_label = np.array([label], dtype=np.float32)
        zero_point = np.zeros((1, 2), dtype=np.float32)
        # zero_point = input_point
        negative_label = np.array([-1], dtype=np.float32)
        coord = np.concatenate((input_point, zero_point), axis=0)[None, :, :]
        label = np.concatenate((input_label, negative_label), axis=0)[None, :]
        return coord, label

    @staticmethod
    def create_decoder_inputs_from(
        input_point=None, input_label=None, input_box=None, box_labels=None, dtype=np.float32
    ):

        onnx_coord, onnx_label = DecoderInputData.point_coords(input_point, input_label)
        if input_box is not None:
            input_box = input_box.reshape(2, 2)
            onnx_coord = np.concatenate([onnx_coord, input_box], axis=0)[None, :, :]
            onnx_label = np.concatenate([onnx_label, box_labels], axis=0)[None, :].astype(dtype)

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=dtype)
        onnx_has_mask_input = np.zeros((1, 1, 1, 1), dtype=dtype)

        print(onnx_has_mask_input.ndim)

        return DecoderInputData(
            point_coords=onnx_coord,
            point_labels=onnx_label,
            mask_input=onnx_mask_input,
            has_mask_input=onnx_has_mask_input,
            dtype=dtype,
        )

    @staticmethod
    def scale_coords(
        coords: np.ndarray,
        orig_height=1024,
        orig_width=1024,
        resized_height=1024,
        resized_width=1024,
        dtype=np.float32,
    ) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension
        """
        old_h, old_w = orig_height, orig_width
        new_h, new_w = resized_height, resized_width
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords.astype(dtype)


class CupyArrayPainter:
    def __init__(self, colormap: cp.ndarray = None):
        if colormap is None:
            colormap = plt.get_cmap("viridis")
            colormap = colormap(np.linspace(0, 1, 256))
            colormap = cp.asarray(colormap * 255, dtype=cp.uint8)
        self.colormap = colormap

    def normalize_data(self, data):
        min_val = data.min()
        max_val = data.max()
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data

    def apply_colormap(self, data):
        # Scale normalized_data to index into the colormap
        indices = (data * (self.colormap.shape[0] - 1)).astype(cp.int32)

        # Get the RGB values from the colormap
        rgba_image = self.colormap[indices]
        return rgba_image

    def to_rgba(self, data):
        normalized_data = self.normalize_data(data)
        rgba_image = self.apply_colormap(normalized_data)
        return rgba_image


class PointMover:
    def __init__(self, width, height, radius, center_x, center_y, frequency=1):
        self.width = width
        self.height = height
        self.radius = radius
        self.center_x = center_x
        self.center_y = center_y
        self.frequency = frequency

    def get_position(self, time):
        """
        Compute the position of the point at a given time.

        :param time: The time parameter, in seconds.
        :return: (x, y) tuple representing the coordinates on the 2D canvas.
        """
        # Calculate the angle based on time
        # the circle frequency can be adjusted, and is in Hz
        circular_frequency = self.frequency
        # the angle is computed based on the circular frequency and time
        angle = 2 * math.pi * circular_frequency * time

        # Calculate the x and y coordinates based on the angle
        x = self.center_x + self.radius * math.cos(angle)
        y = self.center_y + self.radius * math.sin(angle)

        # Ensure the point stays within the canvas boundaries
        x = min(max(x, 0), self.width)
        y = min(max(y, 0), self.height)

        return (x, y)
