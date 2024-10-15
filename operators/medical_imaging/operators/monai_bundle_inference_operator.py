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

import json
import logging
import os
import pickle
import tempfile
import time
import glob
import zipfile
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from holoscan.core import Fragment, OperatorSpec

from operators.medical_imaging.core import AppContext, Image, IOType
from operators.medical_imaging.utils.importutil import optional_import

from .inference_operator import InferenceOperator

MONAI_UTILS = "monai.utils"
nibabel, _ = optional_import("nibabel", "3.2.1")
torch, _ = optional_import("torch", "1.10.2")

NdarrayOrTensor, _ = optional_import("monai.config", name="NdarrayOrTensor")
MetaTensor, _ = optional_import("monai.data.meta_tensor", name="MetaTensor")
PostFix, _ = optional_import(
    "monai.utils.enums", name="PostFix"
)  # For the default meta_key_postfix
ensure_tuple, _ = optional_import(MONAI_UTILS, name="ensure_tuple")
convert_to_dst_type, _ = optional_import(MONAI_UTILS, name="convert_to_dst_type")
MetaKeys, _ = optional_import(MONAI_UTILS, name="MetaKeys")
SpaceKeys, _ = optional_import(MONAI_UTILS, name="SpaceKeys")
create_workflow, _ = optional_import("monai.bundle", name="create_workflow")


__all__ = ["MonaiBundleInferenceOperator", "IOMapping", "BundleConfigNames"]


DISALLOW_LOAD_SAVE = ["LoadImage", "SaveImage"]
DISALLOW_SAVE = ["SaveImage"]


def filter_compose(compose, disallowed_prefixes):
    """
    Removes transforms from the given Compose object whose names begin with `disallowed_prefixes`.
    """
    filtered = []
    for t in compose.transforms:
        tname = type(t).__name__
        if not any(dis in tname for dis in disallowed_prefixes):
            filtered.append(t)

    compose.transforms = tuple(filtered)
    return compose


def is_map_compose(compose):
    """
    Returns True if the given Compose object uses MapTransform instances.
    """
    return isinstance(first(compose.transforms), MapTransform)


class IOMapping:
    """This object holds an I/O definition for an operator."""

    def __init__(
        self,
        label: str,
        data_type: Type,
        storage_type: IOType,
    ):
        """Creates an object holding an operator I/O definitions.

        Limitations apply with the combination of data_type and storage_type, which will
        be validated at runtime.

        Args:
            label (str): Label for the operator input or output.
            data_type (Type): Datatype of the I/O data content.
            storage_type (IOType): The storage type expected, i.e. IN_MEMORY or DISK.
        """
        self.label: str = label
        self.data_type: Type = data_type
        self.storage_type: IOType = storage_type


class MONAIBundleWorkflowType:
    train = "train"
    infer = "inference"



class MonaiBundleInferenceOperator(InferenceOperator):
    """This inference operator automates the inference operation for a given MONAI Bundle.

    This inference operator configures itself based on the bundle workflow setup. When using bundle workflow, only inputs, outputs,
    and accordingly map need to be set for operator running. Its compute method is meant to be general purpose to most any bundle
    such that it will handle any streaming input and output specified in the bundle, using the bundle workflow.
    A number of methods are provided which define parts of functionality relating to this behavior, users may wish
    to overwrite these to change behavior is needed for specific bundles.

    This operator is expected to be linked with both source and destination operators, e.g. receiving an numpy array object from
    an image lodaer operator, and passing a segmentation mask to an image saver operator.
    In this cases, the I/O storage type can only be `IN_MEMORY` due to the restrictions imposed by the application executor.

    For the time being, the input and output to this operator are limited to in_memory object.
    """

    known_io_data_types = {
        "image": Image,  # Image object
        "series": np.ndarray,
        "tuples": np.ndarray,
        "probabilities": Dict[str, Any],  # dictionary containing probabilities and predicted labels
    }

    kw_preprocessed_inputs = "preprocessed_inputs"

    def __init__(
        self,
        fragment: Fragment,
        *args,
        app_context: AppContext,
        input_mapping: List[IOMapping],
        output_mapping: List[IOMapping],
        bundle_path: Union[Path, str],
        **kwargs,
    ):
        """Create an instance of this class, associated with an Application/Fragment.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            app_context (AppContext): Object holding the I/O and model paths, and potentially loaded models.
            input_mapping (List[IOMapping]): Define the inputs' name, type, and storage type.
            output_mapping (List[IOMapping]): Defines the outputs' name, type, and storage type.
            bundle_path (Union[Path, str]): Known path to the bundle file.
        """

        self._executing = False
        self._lock = Lock()

        self._bundle_path = bundle_path
        self._workflow = self._create_bundle_workflow()
        if not self._workflow:
            raise AttributeError(f"Cannot create MONAIBundleInferenceOperator from path {self._bundle_path}")

        self._device = self._workflow.device
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping

        self._fragment = fragment  # In case it is needed.
        self.app_context = app_context

        super().__init__(fragment, *args, **kwargs)

    def _create_bundle_workflow(self):
        """
        Create the MONAI bundle workflow to perform inference.
        The workflow object can be created through two ways:
            1. Through a MONAI bundle config file
            2. Through a python MONAI bundle
        The second one is a placeholder for the pure PythonWorkflow which does not
        contain any config files.
        """
        config_file = self._get_inference_config()
        # Create bundle workflow through config file
        if os.path.exists(config_file):
            workflow = create_workflow(workflow_type=MONAIBundleWorkflowType.infer, config_file=config_file, bundle_root=self.bundle_path)
        else:
            workflow = create_workflow(workflow_type=MONAIBundleWorkflowType.infer, workflow_name="PythonWorkflow")
        return workflow

    def _get_inference_config(self):
        """Get the inference config file."""
        inference_config_list = glob.glob(os.path.join(self.bundle_path, "configs", "inference.*"))
        return inference_config_list[0] if inference_config_list else None

    @property
    def bundle_path(self) -> Union[Path, None]:
        """The path of the MONAI Bundle model."""
        return self._bundle_path

    @bundle_path.setter
    def bundle_path(self, bundle_path: Union[str, Path]):
        if not bundle_path or not Path(bundle_path).expanduser().is_file():
            raise ValueError(f"Value, {bundle_path}, is not a valid file path.")
        self._bundle_path = Path(bundle_path).expanduser().resolve()


    def _get_io_data_type(self, conf):
        """
        Gets the input/output type of the given input or output metadata dictionary. The known Python types for input
        or output types are given in the dictionary `BundleOperator.known_io_data_types` which relate type names to
        the actual type. if `conf["type"]` is an actual object that's not a string then this is assumed to be the
        type specifier and is returned. The fallback type is `bytes` which indicates the value is a pickled object.

        Args:
            conf: configuration dictionary for an input or output from the "network_data_format" metadata section

        Returns:
            A concrete type associated with this input/output type, this can be Image or np.ndarray or a Python type
        """

        # The Bundle's network_data_format for inputs and outputs does not indicate the storage type, i.e. IN_MEMORY
        # or DISK, for the input(s) and output(s) of the operators. Configuration is required, though limited to
        # IN_MEMORY for now.
        # Certain association and transform are also required. The App SDK IN_MEMORY I/O can hold
        # Any type, so if the type match and content format matches, data can simply be used as is, however, with
        # the Type being Image, the object needs to be converted before being used as the expected "image" type.
        ctype = conf["type"]
        if ctype in self.known_io_data_types:  # known type name from the specification
            return self.known_io_data_types[ctype]
        elif isinstance(ctype, type):  # type object
            return ctype
        else:  # don't know, something that hasn't been figured out
            logging.warn(
                f"I/O data type, {ctype}, is not a known/supported type. Return as Type object."
            )
            return object

    def _add_inputs(self, input_mapping: List[IOMapping]):
        """Adds operator inputs as specified."""

        [self.add_input(v.label, v.data_type, v.storage_type) for v in input_mapping]

    def _add_outputs(self, output_mapping: List[IOMapping]):
        """Adds operator outputs as specified."""

        [self.add_output(v.label, v.data_type, v.storage_type) for v in output_mapping]

    def setup(self, spec: OperatorSpec):
        [spec.input(v.label) for v in self._input_mapping]
        for v in self._output_mapping:
            if (
                v.storage_type == IOType.IN_MEMORY
            ):  # As of now the output port type can only be in_memory object.
                spec.output(v.label)

    def compute(self, op_input, op_output, context):
        """Infers with the input(s) and saves the prediction result(s) to output

        Args:
            op_input (InputContext): An input context for the operator.
            op_output (OutputContext): An output context for the operator.
            context (ExecutionContext): An execution context for the operator.
        """

        first_input_name = list(self._inputs.keys())[0]

        inputs: Any = {}  # Use type Any to quiet MyPy type checking complaints.
        for name in self._inputs.keys():
            # Input MetaTensor creation is based on the same logic in monai LoadImage
            # value: NdarrayOrTensor  # MyPy complaints
            value, meta_data = self._receive_input(name, op_input, context)
            value = convert_to_dst_type(value, dst=value)[0]
            if not isinstance(meta_data, dict):
                raise ValueError("`meta_data` must be a dict.")
            value = MetaTensor.ensure_torch_and_prune_meta(value, meta_data)
            inputs[name] = value
            # Named metadata dict not needed any more, as it is in the MetaTensor

        self._workflow.dataflow.clear()
        self._workflow.dataflow.update(inputs)
        start = time.time()
        self._workflow.run()
        logging.debug(f"Bundle inference elapsed time (seconds): {time.time() - start}")
        first_input_v = inputs[first_input_name]  # keep a copy of value for later use

        for name in self._outputs.keys():
            # Note that the input metadata needs to be passed.
            # Please see the comments in the called function for the reasons.
            self._send_output(self._workflow.dataflow[name], name, first_input_v.meta, op_output, context)

    def _receive_input(self, name: str, op_input, context):
        """Extracts the input value for the given input name."""

        # The op_input can have the storage type of IN_MEMORY with the data type being Image or others,
        # as well as the other type of DISK with data type being DataPath.
        # The problem is, the op_input object does not have an attribute for the storage type, which
        # needs to be inferred from data type, with DataPath meaning DISK storage type. The file
        # content type may be interpreted from the bundle's network input type, but it is indirect
        # as the op_input is the input for processing transforms, not necessarily directly for the network.
        in_conf = self._inputs[name]
        itype = self._get_io_data_type(in_conf)
        value = op_input.receive(name)

        metadata = None
        if isinstance(value, Path):
            if not value.exists():
                raise ValueError(f"Input path, {value}, does not exist.")

            file_path = value / name
            # The named input can only be a folder as of now, but just in case things change.
            if value.is_file():
                file_path = value
            elif not file_path.exists() and value.is_dir():
                # Expect one and only one file exists for use.
                files = [f for f in value.glob("*") if f.is_file()]
                if len(files) != 1:
                    raise ValueError(f"Input path, {value}, should have one and only one file.")

                file_path = files[0]

            # Only Python pickle file and or numpy file are supported as of now.
            with open(file_path, "rb") as f:
                if itype == np.ndarray:
                    value = np.load(file_path, allow_pickle=True)
                else:
                    value = pickle.load(f)

        # Once extracted, the input data may be further processed depending on its actual type.
        if isinstance(value, Image):
            # Need to get the image ndarray as well as metadata
            value, metadata = self._convert_from_image(value)
            logging.debug(f"Shape of the converted input image: {value.shape}")
            logging.debug(f"Metadata of the converted input image: {metadata}")
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value).to(self._device)

        # else value is some other object from memory

        return value, metadata

    def _send_output(self, value: Any, name: str, metadata: Dict, op_output, context):
        """Send the given output value to the output context."""

        logging.debug(f"Setting output {name}")

        out_conf = self._outputs[name]
        otype = self._get_io_data_type(out_conf)

        if otype == Image:
            # The value must be torch.tensor or ndarray. Note also that by convention the image/tensor
            # out of the MONAI post processing is [CWHD] with dim for batch already squeezed out.
            # Prediction image, e.g. segmentation image, needs to have its dimensions
            # rearranged to fit the conventions used by Image class, i.e. [DHW], without channel dim.
            # Also, based on known use cases, e.g. prediction being seg image and the destination
            # operators expect the data type to be unit8, conversion needs to be done as well.
            # Metadata, such as pixel spacing and orientation, also needs to be set in the Image object,
            # which is why metadata is expected to be passed in.
            # TODO: Revisit when multi-channel images are supported.

            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            elif not isinstance(value, np.ndarray):
                raise TypeError("arg 1 must be of type torch.Tensor or ndarray.")

            logging.debug(f"Output {name} numpy image shape: {value.shape}")
            result: Any = Image(
                np.swapaxes(np.squeeze(value, 0), 0, 2).astype(np.uint8), metadata=metadata
            )
            logging.debug(f"Converted Image shape: {result.asnumpy().shape}")
        elif otype == np.ndarray:
            result = np.asarray(value)
        elif out_conf["type"] == "probabilities":
            _, value_class = value.max(dim=0)
            prediction = [out_conf["channel_def"][str(int(v))] for v in value.flatten()]

            result = {"result": prediction, "probabilities": value.cpu().numpy()}
        elif isinstance(value, torch.Tensor):
            result = value.cpu().numpy()

        # The operator output currently has many limitation depending on if the operator is
        # a leaf node or not. The get method throws for non-leaf node, irrespective of storage type,
        # and for leaf node if the storage type is IN_MEMORY.
        try:
            op_output_config = op_output.get(name)
            if isinstance(op_output_config, Path):
                output_file = op_output_config / name
                output_file.parent.mkdir(exist_ok=True)
                # Save pickle file
                with open(output_file, "wb") as wf:
                    pickle.dump(result, wf)

                # Cannot (re)set/modify the op_output path to the actual file like below
                # op_output.set(str(output_file), name)
            else:
                op_output.emit(result, name)
        except Exception:
            # The following throws if the output storage type is DISK, but The OutputContext
            # currently does not expose the storage type. Try and let it throw if need be.
            op_output.emit(result, name)

    def _convert_from_image(self, img: Image) -> Tuple[np.ndarray, Dict]:
        """Converts the Image object to the expected numpy array with metadata dictionary.

        Args:
            img: A SDK Image object.
        """

        # The Image class provides a numpy array and a metadata dict without a defined set of keys.
        # In most scenarios, if not all, DICOM series is converted to Image by the
        # DICOMSeriesToVolumeOperator, but the generated metadata lacks the specifics keys expected
        # by the MONAI transforms. So there is need to convert the Image object.
        # Also, there is not a defined key to express the source or producer of an Image object, so,
        # one has to inspect certain keys, based on known conversion, to infer the producer.
        # An issues already exists for the improvement of the Image class.

        img_meta_dict: Dict = img.metadata()

        if (
            not img_meta_dict
            or ("spacing" in img_meta_dict and "original_affine" in img_meta_dict)
            or "row_pixel_spacing" not in img_meta_dict
        ):
            return img.asnumpy(), img_meta_dict
        else:
            return self._convert_from_image_dicom_source(img)

    def _convert_from_image_dicom_source(self, img: Image) -> Tuple[np.ndarray, Dict]:
        """Converts the Image object to the expected numpy array with metadata dictionary.

        Args:
            img: A SDK Image object converted from DICOM instances.
        """

        img_meta_dict: Dict = img.metadata()
        meta_dict = deepcopy(img_meta_dict)

        # The MONAI ImageReader, e.g. the ITKReader, arranges the image spatial dims in WHD,
        # so the "spacing" needs to be expressed in such an order too, as expected by the transforms.
        meta_dict["spacing"] = np.asarray(
            [
                img_meta_dict["row_pixel_spacing"],
                img_meta_dict["col_pixel_spacing"],
                img_meta_dict["depth_pixel_spacing"],
            ]
        )
        # Use defines MetaKeys directly
        meta_dict[MetaKeys.ORIGINAL_AFFINE] = np.asarray(
            img_meta_dict.get("nifti_affine_transform", None)
        )
        meta_dict[MetaKeys.AFFINE] = meta_dict[MetaKeys.ORIGINAL_AFFINE].copy()
        meta_dict[MetaKeys.SPACE] = SpaceKeys.LPS  # not using SpaceKeys.RAS or affine_lps_to_ras

        # Similarly the Image ndarray has dim order DHW, to be rearranged to WHD.
        # TODO: Need to revisit this once multi-channel image is supported and the Image class itself
        #       is enhanced to provide attributes or functions for channel and dim order details.
        converted_image = np.swapaxes(img.asnumpy(), 0, 2)

        # The spatial shape is then that of the converted image, in WHD
        meta_dict[MetaKeys.SPATIAL_SHAPE] = np.asarray(converted_image.shape)

        # Well, now channel for now.
        meta_dict[MetaKeys.ORIGINAL_CHANNEL_DIM] = "no_channel"

        return converted_image, meta_dict
