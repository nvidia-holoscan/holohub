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
import os
import time
import glob
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Type, Union

import numpy as np
from holoscan.core import Fragment, Operator, OperatorSpec

from operators.medical_imaging.core import AppContext, Image, IOType
from operators.medical_imaging.utils.importutil import optional_import


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



class MonaiBundleInferenceOperator(Operator):
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
        input_keys: List[str],
        output_keys: List[str],
        bundle_path: Union[Path, str],
        **kwargs,
    ):
        """Create an instance of this class, associated with an Application/Fragment.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            app_context (AppContext): Object holding the I/O and model paths, and potentially loaded models.
            input_keys (List[str]): Define the inputs' name.
            output_keys (List[str]): Defines the outputs' name.
            bundle_path (Union[Path, str]): Known path to the bundle file.
        """

        self._executing = False
        self._lock = Lock()

        self._bundle_path = bundle_path
        self._workflow = self._create_bundle_workflow()
        if not self._workflow:
            raise AttributeError(f"Cannot create MONAIBundleInferenceOperator from path {self._bundle_path}")

        self._device = self._workflow.device
        self._input_keys = input_keys
        self._output_keys = output_keys

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

    def setup(self, spec: OperatorSpec):
        [spec.input(v) for v in self._input_keys]
        [spec.output(v) for v in self._output_keys]

    def compute(self, op_input, op_output, context):
        """Infers with the input(s) and saves the prediction result(s) to output

        Args:
            op_input (InputContext): An input context for the operator.
            op_output (OutputContext): An output context for the operator.
            context (ExecutionContext): An execution context for the operator.
        """

        inputs: Any = {}  # Use type Any to quiet MyPy type checking complaints.
        for name in self._intput_keys:
            # Input MetaTensor creation is based on the same logic in monai LoadImage
            # value: NdarrayOrTensor  # MyPy complaints
            value = op_input.receive(name)
            inputs[name] = value
            # Named metadata dict not needed any more, as it is in the MetaTensor

        self._workflow.dataflow.clear()
        self._workflow.dataflow.update(inputs)
        start = time.time()
        self._workflow.run()
        logging.debug(f"Bundle inference elapsed time (seconds): {time.time() - start}")

        for name in self._outputs.keys():
            op_output.emit(self._workflow.dataflow[name], name)
