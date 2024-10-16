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


__all__ = ["MonaiBundleInferenceOperator"]

BUNDLE_PROPERTIES_HOLOSCAN = {
    "bundle_root": {
        "description": "root path of the bundle.",
        "required": True,
        "id": "bundle_root"
    },
    "device": {
        "description": "target device to execute the bundle workflow.",
        "required": True,
        "id": "device"
    },
    "dataflow": {
        "description": "dataflow to execute the bundle workflow.",
        "required": True,
        "id": "dataflow"
    },
    "version": {
        "description": "bundle version",
        "required": True,
        "id": "_meta_::version"
    },
    "channel_def": {
        "description": "channel definition for the prediction",
        "required": False,
        "id": "_meta_::network_data_format::outputs::pred::channel_def"
    },
    "type": {
        "description": "data type of the input image",
        "required": False,
        "id": "_meta_::network_data_format::outputs::pred::type"
    }
}


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
        workflow_kwargs: Dict,
        **kwargs,
    ):
        """Create an instance of this class, associated with an Application/Fragment.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            app_context (AppContext): Object holding the I/O and model paths, and potentially loaded models.
            input_keys (List[str]): Define the inputs' name.
            output_keys (List[str]): Defines the outputs' name.
            workflow_kwargs (Dict): Kwargs to initialize a MONAI bundle workflow.
        """

        self._executing = False
        self._lock = Lock()

        self._workflow = self._create_bundle_workflow(workflow_kwargs)
        if not self._workflow:
            raise AttributeError(f"Cannot create MONAIBundleInferenceOperator from kwargs {workflow_kwargs}")

        self._input_keys = input_keys
        self._output_keys = output_keys

        self._fragment = fragment  # In case it is needed.
        self.app_context = app_context

        super().__init__(fragment, *args, **kwargs)

    def _create_bundle_workflow(self, workflow_kwargs):
        """
        Create the MONAI bundle workflow to perform bundle run.
        The workflow object is created from the user defined workflow args.
        """
        workflow = create_workflow(**workflow_kwargs)
        return workflow

    def _get_inference_config(self):
        """Get the inference config file."""
        inference_config_list = glob.glob(os.path.join(self.bundle_path, "configs", "inference.*"))
        return inference_config_list[0] if inference_config_list else None

    def __getattr__(self, name):
        if name in BUNDLE_PROPERTIES_HOLOSCAN:
            return self._workflow.get(name)
    
    def __setattr__(self, name, value):
        if name in BUNDLE_PROPERTIES_HOLOSCAN:
            self._workflow.set(name, value)
        else:
            return super().__setattr__(name, value)

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

        self._workflow.initialize()
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
