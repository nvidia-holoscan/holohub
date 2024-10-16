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
import time
from typing import Any, Dict, List

from holoscan.core import Fragment, Operator, OperatorSpec
from operators.medical_imaging.utils.importutil import optional_import


MONAI_UTILS = "monai.utils"

ensure_tuple, _ = optional_import(MONAI_UTILS, name="ensure_tuple")
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

    def __init__(
        self,
        fragment: Fragment,
        *args,
        input_keys: List[str],
        output_keys: List[str],
        workflow_kwargs: Dict,
        **kwargs,
    ):
        """Create an instance of this class, associated with an Application/Fragment.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            input_keys (List[str]): Define the inputs' name.
            output_keys (List[str]): Defines the outputs' name.
            workflow_kwargs (Dict): Kwargs to initialize a MONAI bundle workflow.
        """

        self._workflow = self._create_bundle_workflow(workflow_kwargs)
        if not self._workflow:
            raise AttributeError(f"Cannot create MONAIBundleInferenceOperator from kwargs {workflow_kwargs}")

        self._input_keys = input_keys
        self._output_keys = output_keys
        self._fragment = fragment  # In case it is needed.

        super().__init__(fragment, *args, **kwargs)

    def _create_bundle_workflow(self, workflow_kwargs):
        """
        Create the MONAI bundle workflow to perform bundle run.
        The workflow object is created from the user defined workflow args.
        """
        workflow = create_workflow(**workflow_kwargs)
        return workflow

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
            value = op_input.receive(name)
            inputs[name] = value

        self._workflow.dataflow.clear()
        self._workflow.dataflow.update(inputs)
        start = time.time()
        self._workflow.run()
        logging.debug(f"Bundle inference elapsed time (seconds): {time.time() - start}")

        for name in self._outputs.keys():
            op_output.emit(self._workflow.dataflow[name], name)
