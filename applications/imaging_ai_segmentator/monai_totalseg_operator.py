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

import logging
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Dict, Sequence, Union

import monai
import torch
from holoscan.core import ConditionType, Fragment, Operator, OperatorSpec
from monai.config import KeysCollection, NdarrayTensor
from monai.networks.layers import GaussianFilter
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    SaveImaged,
    ScaleIntensityd,
    Spacingd,
)
from numpy import uint8

from operators.medical_imaging.core import AppContext, Model
from operators.medical_imaging.monai_seg_inference_operator import (
    InfererType,
    InMemImageReader,
    MonaiSegInferenceOperator,
)


# from https://github.com/Project-MONAI/MONAI/issues/3178
class Antialiasingd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        sigma: Union[Sequence[float], float] = 1.0,
        approx: str = "erf",
        threshold: float = 0.5,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.sigma = sigma
        self.approx = approx
        self.threshold = threshold

    def __call__(self, data: Mapping[Hashable, NdarrayTensor]) -> Dict[Hashable, NdarrayTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]

            gaussian_filter = GaussianFilter(img.ndim - 1, self.sigma, approx=self.approx)

            labels = torch.unique(img)[1:]
            new_img = torch.zeros_like(img)
            for label in labels:
                label_mask = (img == label).to(torch.float)
                blurred = gaussian_filter(label_mask.unsqueeze(0)).squeeze(0)
                new_img[blurred > self.threshold] = label
            d[key] = new_img
        return d


class MonaiTotalSegOperator(Operator):
    """Performs liver and tumor segmentation using a DL model with an image converted from a DICOM CT series.

    The model used in this application is from NVIDIA, and includes configurations for both the pre and post
    transforms as well as inferer. The MONAI Core transforms are used, as such, these transforms are
    simply ported to this operator.

    This operator makes use of the App SDK MonaiSegInferenceOperator in a composition approach.
    It creates the pre-transforms as well as post-transforms with MONAI dictionary based transforms.
    Note that the App SDK InMemImageReader, derived from MONAI ImageReader, is passed to LoadImaged.
    This derived reader is needed to parse the in memory image object, and return the expected data structure.
    Loading of the model, and predicting using the in-proc PyTorch inference is done by MonaiSegInferenceOperator.

    Named Input:
        image: Image object.

    Named Outputs:
        seg_image: Image object of the segmentation object.
        saved_images_folder: Path to the folder with intermediate image output, not requiring a downstream receiver.
    """

    DEFAULT_OUTPUT_FOLDER = Path.cwd() / "output/saved_images_folder"

    def __init__(
        self,
        fragment: Fragment,
        *args,
        app_context: AppContext,
        model_path: Path,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
        **kwargs,
    ):
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"

        self.model_path = self._find_model_file_path(model_path)
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.app_context = app_context
        self.input_name_image = "image"
        self.output_name_seg = "seg_image"
        self.output_name_saved_images_folder = "saved_images_folder"

        # Call the base class __init__() last.
        # Also, the base class has an attribute called fragment for storing the fragment object
        super().__init__(fragment, *args, **kwargs)

    def _find_model_file_path(self, model_path: Path):
        # We'd just have to find the first file and return it for now,
        # because it is known that in Holoscan App Package, the path is the root models folder
        if model_path:
            if model_path.is_file():
                return model_path
            elif model_path.is_dir():
                for file in model_path.rglob("*"):
                    if file.is_file():
                        return file

        raise ValueError(f"Model file not found in the provided path: {model_path}")

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_image)
        spec.output(self.output_name_seg)
        spec.output(self.output_name_saved_images_folder).condition(
            ConditionType.NONE
        )  # Output not requiring a receiver

    def compute(self, op_input, op_output, context):
        input_image = op_input.receive(self.input_name_image)
        if not input_image:
            raise ValueError("Input image is not found.")

        # This operator gets an in-memory Image object, so a specialized ImageReader is needed.
        _reader = InMemImageReader(input_image)

        # In this example, the input image, once loaded at the beginning of the pre-transforms, can
        # be saved on disk, so can the segmentation prediction image at the end of the post-transform.
        # They are both saved in the same subfolder of the application output folder, with names
        # distinguished by the postfix. They can also be saved in different subfolder if need be.
        # These images files can then be packaged for rendering.
        # In the code below, saving of the image files are disabled to save 10 seconds if nii, and 20 if nii.gz
        pre_transforms = self.pre_process(_reader, str(self.output_folder))
        post_transforms = self.post_process(pre_transforms, str(self.output_folder))

        # Load the PyTorch model directly as the model factory limits the support to TorchScript as of now

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _blocks_down: tuple = (1, 2, 2, 4)
        _blocks_up: tuple = (1, 1, 1)

        # Create the model and push to device
        model = monai.networks.nets.segresnet.SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=105,
            init_filters=32,
            blocks_down=_blocks_down,
            blocks_up=_blocks_up,
            dropout_prob=0.2,
        ).to(_device)

        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        loaded_model = Model(self.model_path, name="monai_total_seg")
        loaded_model.predictor = model

        self.app_context.models = loaded_model
        # Delegates inference and saving output to the built-in operator.
        infer_operator = MonaiSegInferenceOperator(
            self.fragment,
            roi_size=(
                96,
                96,
                96,
            ),
            pre_transforms=pre_transforms,
            post_transforms=post_transforms,
            overlap=0.25,
            app_context=self.app_context,
            model_name="",
            inferer=InfererType.SLIDING_WINDOW,
            sw_batch_size=1,
            model_path=self.model_path,
            name="monai_seg_inference_op",
        )

        # Setting the keys used in the dictionary based transforms
        infer_operator.input_dataset_key = self._input_dataset_key
        infer_operator.pred_dataset_key = self._pred_dataset_key

        # Now emit data to the output ports of this operator
        op_output.emit(infer_operator.compute_impl(input_image, context), self.output_name_seg)
        self._logger.debug(
            f"Setting {self.output_name_saved_images_folder} with {self.output_folder}"
        )
        op_output.emit(self.output_folder, self.output_name_saved_images_folder)

    def pre_process(self, img_reader, out_dir: str = "./input_images") -> Compose:
        """Composes transforms for preprocessing input before predicting on a model."""

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        my_key = self._input_dataset_key
        return Compose(
            [
                LoadImaged(keys=my_key, reader=img_reader),
                EnsureTyped(keys=my_key),
                EnsureChannelFirstd(keys=my_key),
                Orientationd(keys=my_key, axcodes="RAS"),
                # The SaveImaged transform can be commented out to save 5 seconds.
                # Uncompress NIfTI file, nii, is used favoring speed over size, but can be changed to nii.gz
                SaveImaged(
                    keys=my_key,
                    output_dir=out_dir,
                    output_postfix="",
                    resample=False,
                    output_ext=".nii",
                ),
                Spacingd(keys=my_key, pixdim=(1.5, 1.5, 1.5), mode=("bilinear")),
                NormalizeIntensityd(keys=my_key, nonzero=True),
                ScaleIntensityd(keys=my_key, minv=-1.0, maxv=1.0),
            ]
        )

    def post_process(
        self, pre_transforms: Compose, out_dir: str = "./prediction_output"
    ) -> Compose:
        """Composes transforms for postprocessing the prediction results."""

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        pred_key = self._pred_dataset_key
        return Compose(
            [
                Activationsd(keys=pred_key, softmax=True),
                AsDiscreted(keys=pred_key, argmax=True),
                Invertd(
                    keys=pred_key,
                    transform=pre_transforms,
                    orig_keys=self._input_dataset_key,
                    nearest_interp=True,
                    to_tensor=True,
                ),
                # Smoothen segmentation volume
                Antialiasingd(
                    keys=pred_key,
                ),
                # The SaveImaged transform can be commented out to save 5 seconds.
                # Uncompress NIfTI file, nii, is used favoring speed over size, but can be changed to nii.gz
                SaveImaged(
                    keys=pred_key,
                    output_dir=out_dir,
                    output_postfix="seg",
                    output_dtype=uint8,
                    resample=False,
                    output_ext=".nii",
                ),
            ]
        )
