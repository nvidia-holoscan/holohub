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
from typing import Callable, Dict, Hashable, List

from holoscan.core import Fragment, Operator, OperatorSpec
from operators.medical_imaging.utils.importutil import optional_import

Compose, _ = optional_import("monai.transforms", name="compose")

__all__ = ["MonaiTransformOperator"]


class MonaiTransformOperator(Operator):
    """This tranform operator can be used to adapt input/output data streams of a MONAI bundle operator
    from/to difference sources.

    When building a holoscan application, there might be difference data sources in a flow. For example,
    there might be data in numpy format, torch Tensor format or other formats. And the MONAI bundle operator
    can only support a subset of the whole data format. What's more, the MONAI bundle operator's output also may
    not be suitable for every downstream operator. For example, the operator's output might be CxHxW foramt. Whereas
    the downstream operator needs a HxWxC format.

    In order to make MONAI bundle more flexiable and adaptable, this MonaiTransformOperator can be inserted into
    a stream to adapt the input/output of a MONAI bundle operator.

    Input keys, output keys and corresponding transforms should be specified during the initialization of this operator.
    Then during the computation, all transforms will be excuted based on the keys and the output will be emited through
    given output keys.
    """
    OP_IN_NAME = "data_in"
    OP_OUT_NAME = "data_out"

    def __init__(
        self,
        fragment: Fragment,
        *args,
        input_keys: List[Hashable],
        output_keys: List[Hashable],
        transforms: List[Callable],
        dict_input: bool = True,
        compose_kwargs: Dict = {},
        **kwargs,
    ):
        """Create an instance of this class, associated with an Fragment.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            input_keys (List[Hashable]): Define the inputs' name.
            output_keys (List[Hashable]): Defines the outputs' name.
            transforms (List[Callable]): transform instances to process data.
            dict_input: whether to process dict input, default to True. If set to False, the input_keys
                        and outpout_keys will not be used. Input transforms should be non-dict transforms.
                        The compute method only takes one input and emits one output.
            compose_kwargs: kwargs to initialize the compose transform.
        """

        self._transforms = transforms
        if not self._transforms:
            raise AttributeError(f"Cannot create a transform operator from given transforms {self._transforms}")

        self._input_keys = input_keys
        self._output_keys = output_keys
        self._compose_kwargs = compose_kwargs
        self._dict_input = dict_input
        self._fragment = fragment  # In case it is needed.

        super().__init__(fragment, *args, **kwargs)

    def _compose_transforms(self):
        """
        Compose input transforms to process input data.
        """
        return Compose(self._transforms, **self._compose_kwargs)

    def _get_inference_config(self):
        """Get the inference config file."""
        inference_config_list = glob.glob(os.path.join(self.bundle_path, "configs", "inference.*"))
        return inference_config_list[0] if inference_config_list else None

    def setup(self, spec: OperatorSpec):
        if self._dict_input:
            [spec.input(v) for v in self._input_keys]
            [spec.output(v) for v in self._output_keys]
        else:
            spec.input(self.OP_IN_NAME)
            spec.output(self.OP_OUT_NAME)

    def _receive_inputs(self, op_input):
        """Receive difference inputs."""
        if self._dict_input:
            inputs = {}
            for name in self._intput_keys:
                value = op_input.receive(name)
                inputs[name] = value
        else:
            inputs = op_input.receive(self.OP_IN_NAME)
        return inputs

    def _send_outputs(self, outputs, op_output):
        """Outputs difference outputs"""
        if self._dict_input:
            for name in self._outputs.keys():
                op_output.emit(outputs[name], name)
        else:
            op_output.emit(outputs, self.OP_OUT_NAME)

    def compute(self, op_input, op_output, context):
        """Transform the input to output

        Args:
            op_input (InputContext): An input context for the operator.
            op_output (OutputContext): An output context for the operator.
            context (ExecutionContext): An execution context for the operator.
        """

        compose_transform = self._compose_transforms()
        inputs = self._receive_inputs(op_input)
        outputs = compose_transform(inputs)
        self._send_outputs(outputs, op_output)
        logging.debug(f"Bundle inference elapsed time (seconds): {time.time() - start}")
