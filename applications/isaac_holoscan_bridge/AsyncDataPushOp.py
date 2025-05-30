# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import queue

from holoscan.core import Operator, OperatorSpec, Tensor


class AsyncDataPushOp(Operator):
    """An asynchronous operator that allows pushing data from external sources into a Holoscan pipeline.

    This operator implements a producer-consumer pattern where data can be pushed from outside
    the pipeline using the push_data() method, and the operator will emit this data through
    its output port when available.

    The operator uses a queue for thread-safe data transfer between the external source and
    the pipeline.

    Attributes:
        _queue: A thread-safe queue for storing data to be processed.
    """

    def __init__(self, fragment, outputs: list[str], max_queue_size: int = 0, *args, **kwargs):
        """Initialize the AsyncDataPushOp.

        Args:
            fragment: The Holoscan fragment this operator belongs to.
            outputs: List of output port names.
            max_queue_size: Maximum size of the internal queue. If 0, the queue size is unlimited.
            *args: Additional positional arguments passed to the parent Operator.
            **kwargs: Additional keyword arguments passed to the parent Operator.
        """
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._outputs = outputs
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Set up the operator's input and output ports.

        Args:
            spec: The OperatorSpec object used to define the operator's interface.
        """
        for output in self._outputs:
            spec.output(output)

    def compute(self, op_input, op_output, context):
        """Process and emit data when available.

        This method waits for data to be pushed via push_data(), then emits it through
        the output port. It blocks until data becomes available in the queue.
        Note: the emitted data is a dictionary, the key is the output port name and the value is the
        data.

        Args:
            op_input: The input data (not used in this operator).
            op_output: The output port to emit data through.
            context: The execution context.
        """
        data_dict = self._queue.get()

        for output_name, data in data_dict.items():
            # Check if data is a array and emit as Tensor, else emit as is
            if (
                hasattr(data, "__cuda_array_interface__")
                or hasattr(data, "__dlpack__")
                or hasattr(data, "__dlpack_device__")
                or hasattr(data, "__array_interface__")
            ):
                op_output.emit({output_name: Tensor.as_tensor(data)}, output_name)
            else:
                op_output.emit({output_name: data}, output_name)

    def push_data(self, data_dict: dict):
        """Push data into the operator for processing.

        This method is called from outside the pipeline to provide data for processing.
        It adds the data to the internal queue, which will be processed by the compute()
        method.

        Args:
            data_dict: A dictionary of data to be processed and emitted through the output ports.

        Note:
            If the queue is full (when max_queue_size > 0), this method will block
            until space becomes available.
        """
        self._queue.put(data_dict)
