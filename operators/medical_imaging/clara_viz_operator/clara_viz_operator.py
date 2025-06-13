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

import numpy as np
from holoscan.core import Fragment, Operator, OperatorSpec

from operators.medical_imaging.utils.importutil import optional_import

DataDefinition, _ = optional_import("clara.viz.core", name="DataDefinition")
Widget, _ = optional_import("clara.viz.widgets", name="Widget")
display, _ = optional_import("IPython.display", name="display")
interactive, _ = optional_import("ipywidgets", name="interactive")
Dropdown, _ = optional_import("ipywidgets", name="Dropdown")
Box, _ = optional_import("ipywidgets", name="Box")
VBox, _ = optional_import("ipywidgets", name="VBox")


class ClaraVizOperator(Operator):
    """
    This operator uses Clara Viz to provide interactive view of a 3D volume including segmentation mask.

    Named input(s):
        image: Image object of the input image, including key metadata, e.g. pixel spacings and orientations.
        seg_image: Image object of the segmentation image derived from the input image.
    """

    def __init__(self, fragment: Fragment, *args, **kwargs):
        """Constructor of the operator.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
        """

        self.input_name_image = "image"
        self.input_name_seg_image = "seg_image"

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_image)
        spec.input(self.input_name_seg_image)
        # There is no output for downstream receiver(s), but interactive UI.

    @staticmethod
    def _build_array(image, order):
        numpy_array = image.asnumpy()

        array = DataDefinition.Array(array=numpy_array, order=order)
        array.element_size = [1.0]
        array.element_size.append(image.metadata().get("col_pixel_spacing", 1.0))
        array.element_size.append(image.metadata().get("row_pixel_spacing", 1.0))
        array.element_size.append(image.metadata().get("depth_pixel_spacing", 1.0))

        # the renderer is expecting data in RIP order (Right Inferior Posterior) which results in
        # this matrix
        target_affine_transform = [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        dicom_affine_transform = image.metadata().get("dicom_affine_transform", np.identity(4))

        affine_transform = np.matmul(target_affine_transform, dicom_affine_transform)

        array.permute_axes = [
            0,
            max(range(3), key=lambda k: abs(affine_transform[0][k])) + 1,
            max(range(3), key=lambda k: abs(affine_transform[1][k])) + 1,
            max(range(3), key=lambda k: abs(affine_transform[2][k])) + 1,
        ]

        array.flip_axes = [
            False,
            affine_transform[0][array.permute_axes[1] - 1] < 0.0,
            affine_transform[1][array.permute_axes[2] - 1] < 0.0,
            affine_transform[2][array.permute_axes[3] - 1] < 0.0,
        ]

        return array

    def compute(self, op_input, op_output, context):
        """Displays the input image and segmentation mask

        Args:
            op_input (InputContext): An input context for the operator.
            op_output (OutputContext): An output context for the operator.
            context (ExecutionContext): An execution context for the operator.
        """
        input_image = op_input.receive(self.input_name_image)
        if not input_image:
            raise ValueError("Original density image not received in the input.")
        input_seg_image = op_input.receive(self.input_name_seg_image)
        if not input_seg_image:
            raise ValueError("Segmentation image not received in the input.")

        # build the data definition
        data_definition = DataDefinition()

        data_definition.arrays.append(self._build_array(input_image, "DXYZ"))

        data_definition.arrays.append(self._build_array(input_seg_image, "MXYZ"))

        widget = Widget()
        widget.select_data_definition(data_definition)
        # default view mode is 'CINEMATIC' switch to 'SLICE_SEGMENTATION' since we have no transfer functions defined
        widget.settings["Views"][0]["mode"] = "SLICE_SEGMENTATION"
        widget.settings["Views"][0]["cameraName"] = "Top"
        widget.set_settings()

        # add controls
        def set_view_mode(view_mode):
            widget.settings["Views"][0]["mode"] = view_mode
            if view_mode == "CINEMATIC":
                widget.settings["Views"][0]["cameraName"] = "Perspective"
            elif widget.settings["Views"][0]["cameraName"] == "Perspective":
                widget.settings["Views"][0]["cameraName"] = "Top"
            widget.set_settings()

        widget_view_mode = interactive(
            set_view_mode,
            view_mode=Dropdown(
                options=[
                    ("Cinematic", "CINEMATIC"),
                    ("Slice", "SLICE"),
                    ("Slice Segmentation", "SLICE_SEGMENTATION"),
                ],
                value="SLICE_SEGMENTATION",
                description="View mode",
            ),
        )

        def set_camera(camera):
            if widget.settings["Views"][0]["mode"] != "CINEMATIC":
                widget.settings["Views"][0]["cameraName"] = camera
                widget.set_settings()

        widget_camera = interactive(
            set_camera,
            camera=Dropdown(options=["Top", "Right", "Front"], value="Top", description="Camera"),
        )

        display(Box([widget, VBox([widget_view_mode, widget_camera])]))
