"""
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

"""This module provides a decorator API for creating Python Operators

.. autosummary::

    holoscan.decorator.create_op
    holoscan.decorator.Input
    holoscan.decorator.Output
"""

import ast
import inspect
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import cupy as cp
import numpy as np

from holoscan.conditions import BooleanCondition
from holoscan.core import (
    ConditionBase,
    ConditionType,
    IOSpec,
    Operator,
    OperatorSpec,
    ResourceBase,
)
from holoscan.core._core import Fragment as FragmentBase
from holoscan.core._core import Tensor as TensorBase

__all__ = ["Input", "Output", "create_op"]


def _is_tensor_like(obj):
    return (
        (hasattr(obj, "__dlpack__") and hasattr(obj, "__dlpack_device__"))
        or hasattr(obj, "__cuda_array_interface__")
        or hasattr(obj, "__array_interface__")
    )


def _as_python_tensor(tensor):
    if hasattr(tensor, "__array_interface__") or (
        hasattr(tensor, "__dlpack_device__") and tensor.__dlpack_device__()[0] == 1
    ):
        return np.asarray(tensor)
    else:
        return cp.asarray(tensor)


@dataclass
class Input:
    """Class for specifying an input port and how the received value maps to a function's arguments.

    Parameters
    ----------
    name : str
        The name of the input port.
    arg_map: str or dict[str, str]
        If `arg_map` is a str, the Python object received by the input port is passed to the
        function argument specified by `arg_map`. If `arg_map` is a dict, the input is assumed to be
        a TensorMap (dictionary of tensors). In this case the keys of the dict are the tensor names
        and the values are the names of the function arguments that the tensors map to.
    size: int | holoscan.core.IOSpec.IOSize, optional
        The size of the queue for the input port.
        By default, `IOSpec.SIZE_ONE` (== `IOSpec.IOSize(1)`) is used.
        If `IOSpec.ANY_SIZE` is used, it defines multiple receivers internally for the input port.
        Otherwise, the size of the input queue is set to the specified value, and the message
        available condition for the input port is set with `min_size` equal to the same value.

        The following size constants are supported:
        - ``IOSpec.ANY_SIZE``: Any size.
        - ``IOSpec.PRECEDING_COUNT``: Number of preceding connections.
        - ``IOSpec.SIZE_ONE``: The queue size is 1.

        Please refer to the [Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_operator.html#receiving-any-number-of-inputs-python)
        to see how to receive any number of inputs in Python.
    policy : holoscan.core.IOSpec.QueuePolicy, optional
        The queue policy for the input port.
        The queue policy to set. Valid values are:

        - QueuePolicy.POP : If the queue is full, pop the oldest item, then add the new one.
        - QueuePolicy.REJECT : If the queue is full, reject (discard) the new item.
        - QueuePolicy.FAULT : If the queue is full, log a warning and reject the new item.
    condition_type : holoscan.core.ConditionType, optional
        The condition type for the input port.
    condition_kwargs : dict[str, Any], optional
        The keywords passed onto the condition specified by `condition_type`.
    connector_type : holoscan.core.IOSpec.ConnectorType, optional
        The connector type for the input port.
    connector_kwargs : dict[str, Any], optional
        The keywords passed onto the connector specified by `connector_type`.
    """

    name: str
    arg_map: str | dict[str, str] | None = ()
    size: int | IOSpec.IOSize = IOSpec.SIZE_ONE
    policy: IOSpec.QueuePolicy | None = None
    condition_type: ConditionType | None = None
    condition_kwargs: dict[str, Any] = field(default_factory=dict)
    connector_type: IOSpec.ConnectorType | None = None
    connector_kwargs: dict[str, Any] = field(default_factory=dict)

    def create_input(self, spec: OperatorSpec) -> IOSpec:
        if isinstance(self.size, int):
            self.size = IOSpec.IOSize(self.size)
        elif not isinstance(self.size, IOSpec.IOSize):
            raise ValueError(f"Invalid size: {self.size}")

        if self.policy is not None and not isinstance(self.policy, IOSpec.QueuePolicy):
            raise ValueError(f"Invalid policy: {self.policy}")

        iospec = spec.input(self.name, size=self.size, policy=self.policy)

        if self.condition_type is not None:
            iospec = iospec.condition(self.condition_type, **self.condition_kwargs)
        if self.connector_type is not None:
            iospec = iospec.connector(self.connector_type, **self.connector_kwargs)


@dataclass
class Output:
    """Class for specifying an output port and how one or more of a functions returned value(s) map
    to it.

    Parameters
    ----------
    name : str
        The name of the output port.
    tensor_names: str, tuple(str) or None
        If None, whatever Python object the func outputs is emitted on the output port. If a tuple
        of strings is provided it is assumed that the func returns a dictionary of tensors. The
        names in the tuple specify which tensors in the dict will be transmitted on the output
        port. There is no need to specify `tensor_names` if all tensors in a dict returned by the
        function are to be transmitted. In the case of a single tensor name, a string can be
        provided instead of a tuple.
    size: int | holoscan.core.IOSpec.IOSize, optional
        The size of the queue for the output port.
        By default, `IOSpec.SIZE_ONE` (== `IOSpec.IOSize(1)`) is used.
    policy : holoscan.core.IOSpec.QueuePolicy, optional
        The queue policy for the output port.
        The queue policy to set. Valid values are:

        - QueuePolicy.POP : If the queue is full, pop the oldest item, then add the new one.
        - QueuePolicy.REJECT : If the queue is full, reject (discard) the new item.
        - QueuePolicy.FAULT : If the queue is full, log a warning and reject the new item.
    condition_type : holoscan.core.ConditionType, optional
        The condition type for the input port.
    condition_kwargs : dict[str, Any], optional
        The keywords passed onto the condition specified by `condition_type`.
    connector_type : holoscan.core.IOSpec.ConnectorType, optional
        The connector type for the input port.
    connector_kwargs : dict[str, Any], optional
        The keywords passed onto the connector specified by `connector_type`.
    """

    name: str
    tensor_names: str | tuple[str] | None = ()
    size: int | IOSpec.IOSize = IOSpec.SIZE_ONE
    policy: IOSpec.QueuePolicy | None = None
    condition_type: ConditionType | None = None
    condition_kwargs: dict[str, Any] = field(default_factory=dict)
    connector_type: IOSpec.ConnectorType | None = None
    connector_kwargs: dict[str, Any] = field(default_factory=dict)

    def create_output(self, spec: OperatorSpec) -> IOSpec:
        if isinstance(self.size, int):
            self.size = IOSpec.IOSize(self.size)
        elif not isinstance(self.size, IOSpec.IOSize):
            raise ValueError(f"Invalid size: {self.size}")

        if self.policy is not None and not isinstance(self.policy, IOSpec.QueuePolicy):
            raise ValueError(f"Invalid policy: {self.policy}")

        iospec = spec.output(self.name, size=self.size, policy=self.policy)
        if self.condition_type is not None:
            iospec = iospec.condition(self.condition_type, **self.condition_kwargs)
        if self.connector_type is not None:
            iospec = iospec.connector(self.connector_type, **self.connector_kwargs)


def _as_input(input_: str | Input):
    """Cast str to Output object."""
    if isinstance(input_, str):
        return Input(input_, arg_map=input_)
    elif not isinstance(input_, Input):
        return ValueError("`inputs` must be a single port name or Input object or a tuple of these")
    return input_


def _as_output(output: str | Output):
    """Cast str to Output object."""
    if isinstance(output, str):
        return Output(output)
    elif not isinstance(output, Output):
        return ValueError(
            "`outputs` must be a single port name or Output object or a tuple of these"
        )
    return output


def _has_function_returns_value(func):
    """Check if the provided function has any return statements returning a value."""

    class ReturnVisitor(ast.NodeVisitor):
        def __init__(self):
            self.returns_value = False

        def visit_Return(self, node):  # noqa: N802
            # check if the return statement has a value
            if node.value is not None:
                self.returns_value = True
                return

            self.generic_visit(node)

        def visit_ClassDef(self, node):  # noqa: N802, ARG002
            return

        def visit_FunctionDef(self, node):  # noqa: N802, ARG002
            return

        def visit_AsyncFunctionDef(self, node):  # noqa: N802, ARG002
            return

        def visit(self, node):
            if self.returns_value:
                return
            super().visit(node)

    # parse the source code into an AST
    source_code = inspect.getsource(func)
    # deindent the text if it is indented
    source_code = textwrap.dedent(source_code)
    tree = ast.parse(source_code)
    # initialize the visitor
    visitor = ReturnVisitor()
    # walk the AST
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
            visitor.generic_visit(node)
            break
    return visitor.returns_value


def create_op(
    function_or_class: type | Callable[..., Any] | None = None,
    inputs: str | Input | Sequence[str | Input] = (),
    outputs: str | Output | Sequence[str | Output] = (),
    cast_tensors: bool = True,
    op_param: str | None = None,
) -> Callable:
    """Decorator for creating an operator from a function or a class.

    When the decorator is used on a class, the class must have a `__call__` method that will be
    used as the operator function.

    inputs : str, Input, or Tuple[str | Input], optional
        If a str is provided, it is assumed to be the name of the input port and that the function
        has a variable matching that port name to which the object received on the port will be
        connected. If the port name does not match the name of the variable in the function
        signature, or if there are multiple tensors to be mapped to multiple objects, use an Input
        argument. A tuple of str or Input objects can be provided to specify multiple input ports.
        The default of an empty tuple corresponds to no input ports.
    outputs : str, Output, or Tuple[str | Output], optional
        If a str is provided, any value returned by the function will be emitted on an output port
        of that name. If a tuple of multiple str is provided and the function returns a tuple, then
        the tuple elements will be emitted from each output port in the order at which they are
        defined. In this case, the number of output ports should match the length of the output
        tuple. Finally, an Output object can be provided in the case that the function returns a
        dictionary of output arrays that should be split across multiple ports.
    cast_tensors : bool, optional
        If True, automatically cast any tensor-like input to a NumPy or CuPy array (for host and
        device tensors, respectively). If set to False, these will be left as `holoscan.Tensor` and
        the user will have to cast to the desired type within the body of the decorated function or
        class.
    op_param : str, optional
        If provided, adds this parameter name to the function signature which will
        contain a reference to the operator instance. This allows the function to
        access operator methods and attributes.

    Notes
    -----
    Another case where using `Input` or `Output` objects is necessary is if the user wishes to
    override the default connector or condition types for the port.
    """
    # used to store the class object if the decorator is used on a class
    class_obj = None
    # used to determine if the decorator was used without args
    is_without_args = function_or_class is not None

    # convert scalars to tuple
    if isinstance(inputs, str | Input):
        inputs = (inputs,)
    # convert any str in the tuple to an Input object
    inputs = tuple(_as_input(i) for i in inputs)

    if isinstance(outputs, str | Output):
        outputs = (outputs,)
    # convert any str in the tuple to an Output object
    outputs = tuple(_as_output(o) for o in outputs)

    if not isinstance(outputs, tuple):
        raise ValueError(
            "`outputs` must be a single port name or Output object or a tuple of these"
        )

    if op_param is not None and not isinstance(op_param, str):
        raise TypeError(f"op_param must be a string or None, got {type(op_param)}")

    def decorator(func_or_cls):
        nonlocal function_or_class, class_obj

        def make_class(*args, **kwargs):
            if "fragment" in kwargs:
                fragment = kwargs.pop("fragment")
            elif args and isinstance(args[0], FragmentBase):
                fragment, args = args[0], args[1:]
            else:
                raise ValueError(
                    "fragment must be provided via kwarg or as the first positional argument"
                )

            # frame = inspect.currentframe()
            # args_names, _, _, locals_dict = inspect.getargvalues(frame)
            # print(f"{args_names=}, {locals_dict=}")

            class DynamicOp(Operator):
                def __init__(
                    self,
                    fragment: FragmentBase,
                    *args,
                    inputs,
                    outputs,
                    cast_tensors=cast_tensors,
                    op_param=op_param,
                    **kwargs,
                ):
                    self.func = func_or_cls
                    self.input_objs = inputs
                    self.output_objs = outputs
                    self.is_generator = inspect.isgeneratorfunction(self.func)
                    self.gen_obj = None
                    self.cast_tensors = cast_tensors
                    self.op_param = op_param

                    # remove conditions and resources from *args
                    condition_args = tuple(a for a in args if isinstance(a, ConditionBase))
                    resource_args = tuple(a for a in args if isinstance(a, ResourceBase))
                    args = tuple(a for a in args if not isinstance(a, ConditionBase | ResourceBase))
                    self.func_args = args

                    # add a boolean condition to prevent triggering if the function is a generator
                    # and the iteration is complete
                    if self.is_generator:
                        condition_args = condition_args + (
                            BooleanCondition(fragment, name="_generator_func"),
                        )

                    # set name kwarg to self.func.__name__ if not provided
                    name = kwargs.pop("name", self.func.__name__)

                    argspec = inspect.getfullargspec(self.func)

                    # remove self from argspec.args if the decorator is used on a class
                    if class_obj:
                        argspec = argspec._replace(args=argspec.args[1:])
                        self.class_obj = class_obj

                    self.func_argspec = argspec

                    # populate inputs and outputs with defaults if decorator was used without args
                    if is_without_args:
                        self.input_objs = tuple(Input(name, arg_map=name) for name in argspec.args)
                        # configure the output port if the function contains return statements
                        # (in this case, the port name will be left empty)
                        if _has_function_returns_value(function_or_class):
                            self.output_objs = tuple((Output(""),))

                    # populate all arguments not provided with defaults
                    if argspec.kwonlydefaults is not None:
                        for k in argspec.kwonlyargs:
                            if k not in kwargs and k in argspec.kwonlydefaults:
                                kwargs[k] = argspec.kwonlydefaults[k]

                    # store a list of what ports map to what function arguments
                    self.input_mappings = {}
                    for input_obj in self.input_objs:
                        # store what argument(s) this input maps to
                        self.input_mappings[input_obj.name] = input_obj.arg_map

                    # sets self.dynamic_kwargs and self.fixed_kwargs
                    self._set_fixed_and_dynamic_kwargs(kwargs)

                    # get the type annotations dict for the function (not currently used)
                    # self.func_annotations = inspect.get_annotations(self.func)
                    self.func_annotations = self.func.__annotations__

                    super().__init__(fragment, *condition_args, *resource_args, name=name)

                def _set_fixed_and_dynamic_kwargs(self, kwargs):
                    """Split provided kwargs into those which are "fixed" and those which are
                    "dynamic".

                    Here "dynamic" refers to function arguments that are obtained from input
                    ports. The keys for self.dynamic_kwargs are determined here, but the values
                    are initialized to None. Actual values get set during each `compute` call.

                    "fixed" refers to other keyword arguments to the function that don't change
                    across calls.
                    """
                    self.dynamic_kwargs = {}
                    for input_map in self.input_mappings.values():
                        if isinstance(input_map, str):
                            self._add_dynamic_arg(input_map, kwargs)
                        elif isinstance(input_map, dict):
                            for arg_name in input_map.values():
                                self._add_dynamic_arg(arg_name, kwargs)
                    self.fixed_kwargs = kwargs

                    # add the operator instance to the kwargs if op_param was specified
                    if self.op_param:
                        self.fixed_kwargs[self.op_param] = self

                    # store any positional args with specified defaults in fixed_kwargs instead
                    argspec = self.func_argspec
                    if argspec.defaults is not None:
                        n_default_positional = len(argspec.defaults)
                        if n_default_positional > 0:
                            self.func_args = self.func_args[:-n_default_positional]
                        n_required_positional = len(argspec.args) - len(argspec.defaults)
                        for k, v in zip(
                            argspec.args[n_required_positional:], argspec.defaults, strict=False
                        ):
                            # don't overwrite any kwargs that were provided
                            if k not in self.fixed_kwargs:
                                self.fixed_kwargs[k] = v

                    # Now that all args with defaults are in self.fixed_kwargs we can check if any
                    # of the required arguments were not specified
                    required_args = set(argspec.args) | set(argspec.kwonlyargs)
                    if argspec.kwonlydefaults is not None:
                        required_args -= set(argspec.kwonlydefaults.keys())
                    for arg in required_args:
                        if arg not in self.fixed_kwargs and arg not in self.dynamic_kwargs:
                            raise ValueError(f"required argument, '{arg}', has not been specified")

                def _add_dynamic_arg(self, arg_name, kwargs):
                    """helper function for _set_fixed_and_dynamic_kwargs"""
                    if arg_name in self.dynamic_kwargs:
                        raise ValueError(
                            "duplicate specification of mapping to function kwarg: '{arg_name}'"
                        )
                    self.dynamic_kwargs[arg_name] = None
                    try:
                        kwargs.pop(arg_name)
                    except KeyError as e:
                        argspec = self.func_argspec
                        if arg_name not in argspec.kwonlyargs + argspec.args:
                            msg = (
                                f"Provided func does not have an arg or kwarg named '{arg_name}'."
                                " The provided wrapped function has"
                                f" positional args: {argspec.args}"
                                f" and keyword-only args: {argspec.kwonlyargs}"
                            )
                            raise KeyError(msg) from e

                # # not used by the Application, but can be useful to test the call
                # def __call__(self, *args, **kwargs):
                #     print(f"{self.msg=}")
                #     return self.func(*self.func_args, *args, **self.fixed_kwargs, **kwargs)

                def setup(self, spec: OperatorSpec):
                    for input_obj in self.input_objs:
                        input_obj.create_input(spec)

                    self.output_tensor_map = {}
                    for output_obj in self.output_objs:
                        output_obj.create_output(spec)
                        if isinstance(output_obj.tensor_names, str):
                            output_obj.tensor_names = (output_obj.tensor_names,)
                        self.output_tensor_map[output_obj.name] = tuple(output_obj.tensor_names)

                def compute(self, op_input, op_output, context):
                    for port_name, arg_map in self.input_mappings.items():
                        # print(f"input {port_name=}, {arg_map=}")
                        msg = op_input.receive(port_name)
                        if isinstance(arg_map, str):
                            # print(f"{msg=}")
                            if isinstance(msg, dict):
                                try:
                                    # try tensor based on matching name
                                    msg = msg[arg_map]
                                except KeyError as e:
                                    # use tensor regardless of name if only one is present
                                    tensors = tuple(
                                        v for k, v in msg.items() if isinstance(v, TensorBase)
                                    )
                                    if len(tensors) == 1:
                                        msg = tensors[0]
                                    elif len(tensors) > 1:
                                        raise ValueError(
                                            "More than one tensor found in port, but none has "
                                            f"name {arg_map}"
                                        ) from e

                            # cast holoscan.Tensor to cp.asarray(Tensor) here or require the user
                            # to do it in the provided func?
                            if self.cast_tensors and isinstance(msg, TensorBase):
                                msg = _as_python_tensor(msg)

                            self.dynamic_kwargs[arg_map] = msg
                        elif isinstance(arg_map, dict):
                            for tensor_name, arg_name in arg_map.items():
                                try:
                                    val = msg[tensor_name]
                                except KeyError as e:
                                    raise KeyError(
                                        f"key with name '{tensor_name}' not found in input dict"
                                    ) from e
                                if self.cast_tensors and isinstance(val, TensorBase):
                                    val = _as_python_tensor(val)
                                self.dynamic_kwargs[arg_name] = val

                    if self.is_generator:
                        if self.gen_obj is None:
                            out = self.func(
                                *self.func_args, **self.fixed_kwargs, **self.dynamic_kwargs
                            )
                            self.gen_obj = out
                        try:
                            out = next(self.gen_obj)
                        except StopIteration:
                            # disable the condition to prevent further calls
                            self.conditions["_generator_func"].disable_tick()
                            return
                    else:
                        out = self.func(*self.func_args, **self.fixed_kwargs, **self.dynamic_kwargs)

                    # if the output is a tuple and there is >1 port, we distribute the outputs
                    if isinstance(out, tuple) and (len(self.output_tensor_map) > 1):
                        # for tuple case, each port should correspond to each output tuple element
                        if any([len(names) > 1 for names in self.output_tensor_map.values()]):
                            raise ValueError(
                                "The function output was found to be a tuple type, but each "
                                "output tuple element must have its own port. In other words, "
                                "the `outputs` argument of `create_op` should be a tuple of port "
                                "names equal in length to the returned tuple."
                            )
                        # Make sure check that the output tuple length and number of ports match
                        if len(out) != len(self.output_tensor_map):
                            raise ValueError(
                                f"The number of output tuple elements and number of tensors must "
                                f"match.\n"
                                f"Output tuple length = {len(out)}\n"
                                f"Number of output tensors = {len(self.output_tensor_map)}"
                            )
                        for (port_name, tensor_names), out_element in zip(
                            self.output_tensor_map.items(), out, strict=False
                        ):
                            if _is_tensor_like(out_element):
                                name = "" if len(tensor_names) == 0 else tensor_names[0]
                                out_element = {name: out_element}
                            op_output.emit(out_element, port_name)
                        return

                    for port_name, tensor_names in self.output_tensor_map.items():
                        if tensor_names is None or len(tensor_names) == 0:
                            if _is_tensor_like(out):
                                # emit as dict of tensor-like objects
                                out = {"": out}
                            op_output.emit(out, port_name)
                        elif len(tensor_names) == 1:
                            name = tensor_names[0]
                            if _is_tensor_like(out):
                                # emit as dict of tensor-like objects
                                out = {name: out}
                                op_output.emit(out, port_name)
                            else:
                                if name not in out:
                                    raise ValueError(
                                        f"tensor with name '{name}' not found in function output"
                                    )
                                op_output.emit({name: out[name]}, port_name)
                        else:
                            out_tensors = {}
                            for name in tensor_names:
                                if name not in out:
                                    raise ValueError(
                                        f"tensor with name '{name}' not found in function output"
                                    )
                                out_tensors[name] = out[name]
                            # print(f"outputting tensors named: {tuple(out_tensors.keys())} on
                            # port {port_name}")
                            # print(f"tensormap emit of {out_tensors=}")
                            op_output.emit(out_tensors, port_name)

            op = DynamicOp(
                fragment, *args, inputs=inputs, outputs=outputs, op_param=op_param, **kwargs
            )

            def _to_camel_case(name):
                """Convert name to camel case"""
                parts = name.split("_")
                return "".join(p.capitalize() for p in parts)

            # manually update instead of using functools.update_wrapper(op, func_or_cls) because:
            #    - don't want to overwrite __doc__ with func.__doc__
            #    - want to use name instead of func.__name__
            if class_obj:
                class_name = class_obj.__class__.__name__
                op.__name__ = class_name + "Op" if not class_name.endswith("Op") else class_name
            else:
                op.__name__ = _to_camel_case(func_or_cls.__name__) + "Op"
            op.__qualname__ = op.__name__
            op.__module__ = func_or_cls.__module__
            return op

        def init_class(*args, **kwargs):
            nonlocal class_obj, function_or_class
            # create an instance of the class (using function_or_class as the class)
            class_obj = function_or_class(*args, **kwargs)
            # use the class's __call__ method as the operator function
            if not callable(class_obj):
                raise ValueError(
                    f"{function_or_class} must have a __call__ method to be used as an operator"
                )
            function_or_class = class_obj.__call__
            return decorator(function_or_class)

        if func_or_cls is None:
            return decorator

        # check if the decorator was used on a class first
        if inspect.isclass(func_or_cls):  # if isinstance(func_or_cls, type):
            function_or_class = func_or_cls
            return init_class

        if callable(func_or_cls):
            return make_class

        raise Exception(f"Invalid usage of decorator for {func_or_cls}")

    return decorator(function_or_class)
