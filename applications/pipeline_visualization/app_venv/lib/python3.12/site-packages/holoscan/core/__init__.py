# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""This module provides a Python API for the core C++ API classes.

The `Application` class is the primary class that should be derived from to
create a custom application.

.. autosummary::

    holoscan.core.Application
    holoscan.core.Arg
    holoscan.core.ArgContainerType
    holoscan.core.ArgElementType
    holoscan.core.ArgList
    holoscan.core.ArgType
    holoscan.core.AsyncDataLoggerResource
    holoscan.core.AsyncQueuePolicy
    holoscan.core.CLIOptions
    holoscan.core.Clock
    holoscan.core.ClockInterface
    holoscan.core.Component
    holoscan.core.ComponentSpec
    holoscan.core.Condition
    holoscan.core.ConditionBase
    holoscan.core.ConditionType
    holoscan.core.Config
    holoscan.core.DataFlowMetric
    holoscan.core.DataFlowTracker
    holoscan.core.DataLogger
    holoscan.core.DataLoggerResource
    holoscan.core.DefaultFragmentService
    holoscan.core.DLDevice
    holoscan.core.DLDeviceType
    holoscan.core.ExecutionContext
    holoscan.core.Executor
    holoscan.core.Fragment
    holoscan.core.FlowInfo
    holoscan.core.Fragment
    holoscan.core.Graph
    holoscan.core.FragmentService
    holoscan.core.InputContext
    holoscan.core.IOSpec
    holoscan.core.Message
    holoscan.core.MetadataDictionary
    holoscan.core.MetadataPolicy
    holoscan.core.MultiMessageConditionInfo
    holoscan.core.NetworkContext
    holoscan.core.Operator
    holoscan.core.OperatorBase
    holoscan.core.OperatorSpec
    holoscan.core.OperatorStatus
    holoscan.core.OutputContext
    holoscan.core.ParameterFlag
    holoscan.core.arg_to_py_object
    holoscan.core.arglist_to_kwargs
    holoscan.core.Resource
    holoscan.core.ResourceBase
    holoscan.core.SchedulingStatusType
    holoscan.core.ServiceDriverEndpoint
    holoscan.core.ServiceWorkerEndpoint
    holoscan.core.Subgraph
    holoscan.core.Tensor
    holoscan.core.Tracker
    holoscan.core.kwargs_to_arglist
    holoscan.core.py_object_to_arg
"""

import logging
import sys

# Note: Python 3.7+ expects the threading module to be initialized (imported) before additional
# threads are created (by C++ modules using pybind11).
# Otherwise you will get an assert tlock.locked() error on exit.
# (CLARAHOLOS-765)
import threading as _threading  # noqa: F401, I001
import warnings
from collections.abc import Callable

# Add ThreadPoolExecutor to imports if not already there
from concurrent.futures import ThreadPoolExecutor

# Check stack size and warn if insufficient
try:
    import resource

    # Get current stack size limit
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_STACK)

    # Recommended minimum stack size (32 MB)
    RECOMMENDED_STACK_SIZE = 32 * 1024 * 1024

    # Check if soft limit is set and is below recommended size
    if soft_limit != resource.RLIM_INFINITY and soft_limit < RECOMMENDED_STACK_SIZE:
        warnings.warn(
            f"Current stack size ({soft_limit / (1024 * 1024):.1f} MB) is below the "
            f"recommended minimum ({RECOMMENDED_STACK_SIZE / (1024 * 1024):.1f} MB). "
            f"This may cause segmentation faults or crashes. "
            f"Consider increasing the stack size with 'ulimit -s {RECOMMENDED_STACK_SIZE // 1024}'"
            f", or if using Docker, launch the container with "
            f"'--ulimit stack={RECOMMENDED_STACK_SIZE}'.",
            RuntimeWarning,
            stacklevel=1,
        )
except (ImportError, OSError, ValueError):
    # resource module not available (e.g., on Windows) or error getting limits
    pass

# Import statements for the C++ API classes
from ..graphs._graphs import FragmentGraph, OperatorGraph
from ._core import Application as _Application
from ._core import (
    Arg,
    ArgContainerType,
    ArgElementType,
    ArgList,
    ArgType,
    AsyncDataLoggerResource,
    AsyncQueuePolicy,
    CLIOptions,
    Clock,
    ClockInterface,
    Component,
    ConditionBase,
    ConditionType,
    Config,
    DataFlowMetric,
    DataFlowTracker,
    DataLogger,
    DataLoggerResource,
    DistributedAppService,
    DLDevice,
    DLDeviceType,
    Executor,
    FlowInfo,
    FragmentService,
    IOSpec,
    Message,
    MetadataDictionary,
    MetadataPolicy,
    MultiMessageConditionInfo,
    NetworkContext,
    OperatorBase,
    OperatorStatus,
    ParameterFlag,
    ResourceBase,
    Scheduler,
    SchedulingStatusType,
    ServiceDriverEndpoint,
    ServiceWorkerEndpoint,
    arg_to_py_object,
    arglist_to_kwargs,
    kwargs_to_arglist,
    py_object_to_arg,
)
from ._core import DefaultFragmentService as _DefaultFragmentService
from ._core import Fragment as _Fragment
from ._core import PyComponentSpec as ComponentSpec
from ._core import PyExecutionContext as ExecutionContext
from ._core import PyInputContext as InputContext
from ._core import PyOperatorSpec as OperatorSpec
from ._core import PyOutputContext as OutputContext
from ._core import PyRegistryContext as _RegistryContext
from ._core import PyTensor as Tensor
from ._core import Subgraph as _Subgraph
from ._core import register_types as _register_types

# Get a logger instance for this module
logger = logging.getLogger(__name__)

Graph = OperatorGraph  # define alias for backward compatibility

__all__ = [
    "Application",
    "Arg",
    "ArgContainerType",
    "ArgElementType",
    "ArgList",
    "ArgType",
    "AsyncDataLoggerResource",
    "AsyncQueuePolicy",
    "CLIOptions",
    "Clock",
    "ClockInterface",
    "Component",
    "ComponentSpec",
    "ConditionType",
    "Condition",
    "ConditionBase",
    "Config",
    "DataFlowMetric",
    "DataFlowTracker",
    "DataLogger",
    "DataLoggerResource",
    "DefaultFragmentService",
    "DistributedAppService",
    "DLDevice",
    "DLDeviceType",
    "ExecutionContext",
    "Executor",
    "FlowInfo",
    "Fragment",
    "FragmentGraph",
    "FragmentService",
    "Graph",
    "InputContext",
    "IOSpec",
    "Message",
    "MetadataDictionary",
    "MetadataPolicy",
    "MultiMessageConditionInfo",
    "NetworkContext",
    "Operator",
    "OperatorBase",
    "OperatorSpec",
    "OperatorStatus",
    "OperatorGraph",
    "OutputContext",
    "ParameterFlag",
    "Resource",
    "ResourceBase",
    "Scheduler",
    "SchedulingStatusType",
    "ServiceDriverEndpoint",
    "ServiceWorkerEndpoint",
    "START_OPERATOR_NAME",
    "Subgraph",
    "Tensor",
    "Tracker",
    "arg_to_py_object",
    "arglist_to_kwargs",
    "io_type_registry",
    "kwargs_to_arglist",
    "py_object_to_arg",
]

# define aliases for backwards compatibility
_Condition = ConditionBase
_Operator = OperatorBase
_Resource = ResourceBase


# Define custom __repr__ method for MetadataDictionary
def metadata_repr(self):
    items = {k: v for k, v in self.items()}
    return f"{items}"


# need these imports for ThreadPool return type of Fragment.make_thread_pool to work
from ..gxf._gxf import GXFResource as _GXFResource  # noqa: E402, F401, I001
from ..resources import ThreadPool as _ThreadPool  # noqa: E402, F401, I001

MetadataDictionary.__repr__ = metadata_repr

# Defines the special operator name used to initiate application execution.
# The GXF framework requires entity names to not begin with double underscores,
# so this distinctive name pattern is chosen to prevent naming collisions.
# This constant mirrors the C++ definition of `holoscan::kStartOperatorName`
# found in holoscan/core/fragment.hpp
START_OPERATOR_NAME = "<|start|>"


class Application(_Application):
    def __init__(self, argv=None, *args, **kwargs):
        # If no arguments are provided, instead of letting the C++ API initialize the application
        # from the command line (through '/proc/self/cmdline'), we initialize the application
        # with the command line arguments retrieved from the Python interpreter.
        # This is because the C++ API will not be able to discard arguments that are not meant for
        # the Python application.
        # For example, if the user runs the application with the following
        # command line arguments:
        #   /usr/bin/python3 -m pytest -v -k test_init /workspace/holoscan-sdk/public/python/tests
        # then the C++ API will get the following arguments:
        #   ['/usr/bin/python3', '-m', 'pytest', '-v', '-k', 'test_init',
        #    '/workspace/holoscan-sdk/public/python/tests']
        # whereas the Python interpreter (sys.argv) will get the following arguments:
        #   ['/usr/lib/python3/dist-packages/pytest.py', '-v', '-k', 'test_init',
        #    '/workspace/holoscan-sdk/public/python/tests']
        # For the reason above, we initialize the application with the arguments:
        #   [sys.executable, *sys.argv]
        # which will be equivalent to the following command line arguments:
        #   ['/usr/bin/python3', '/usr/lib/python3/dist-packages/pytest.py', '-v', '-k',
        #    'test_init', '/workspace/holoscan-sdk/public/python/tests']
        # and ``Application().argv`` will return the same arguments as ``sys.argv``.

        if not argv:
            argv = [sys.executable, *sys.argv]

        # It is recommended to not use super()
        # (https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python)
        _Application.__init__(self, argv, *args, **kwargs)
        self._async_executor = None
        self._async_executor_lock = _threading.Lock()
        self._start_op = None

    def run_async(self):
        """Run the application asynchronously using a shared executor.

        This method uses a shared ThreadPoolExecutor associated with this
        Application instance. The executor is created on the first call.
        Call `shutdown_async_executor()` when done with async runs
        to clean up resources.

        Returns
        -------
        future : ``concurrent.futures.Future`` object
        """
        # Ensure only one thread creates the executor
        with self._async_executor_lock:
            if self._async_executor is None:
                # Create the executor ONCE
                self._async_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=f"HoloscanApp_{self.name}_Async"
                )

        # Submit the job to the shared executor
        return self._async_executor.submit(self.run)

    def shutdown_async_executor(self, wait=True):
        """Shuts down the shared asynchronous executor.

        Call this method when the application instance is no longer needed
        and asynchronous runs initiated by `run_async` should terminate.

        Parameters
        ----------
        wait : bool
            If True (default), wait for running tasks to complete before shutting down.
            If False, shut down immediately.
        """
        # Use the lock to prevent race conditions with run_async
        with self._async_executor_lock:
            if self._async_executor is not None:
                # Shutting down async executor
                self._async_executor.shutdown(wait=wait)
                self._async_executor = None

    def start_op(self):
        """Get or create the start operator for this application.

        This operator is nothing but the first operator that was added to the application.
        It has the name of `<|start|>` and has a condition of `CountCondition(1)`.
        This Operator is used to start the execution of the application.
        Entry operators who want to start the execution of the application should connect to this
        operator.

        If this method is not called, no start operator is created.
        Otherwise, the start operator is created if it does not exist, and the start operator is
        returned.

        Returns
        -------
        Operator
            The start operator instance. If it doesn't exist, it will be created with
            a CountCondition(1).
        """
        from ..conditions import CountCondition  # noqa: PLC0415

        if not self._start_op:
            self._start_op = Operator(self, CountCondition(self, 1), name=START_OPERATOR_NAME)
            self.add_operator(self._start_op)
        return self._start_op

    # If we created a context via `gxf.context_create` then we would need to
    # call `gxf.context_destroy` in a destructor. However, in the __init__
    # here, the C++ API creates the GXFExecutor and its context and the
    # C++ object will also take care of the context deletion.

    # def __del__(self):
    #    context_destroy(self._context)

    def __del__(self):
        # This is best-effort cleanup, not guaranteed to be called reliably.
        # Avoid potentially blocking calls or complex logic here.
        if self._async_executor is not None:
            # Non-blocking shutdown is safer in __del__ if possible,
            # but might leave work unfinished or resources dangling longer.
            # Using wait=False might be preferable here, but check implications.
            try:
                self._async_executor.shutdown(wait=False)  # Try non-blocking first
            except Exception as e:
                logger.error(
                    f"Error during __del__ executor shutdown for Application {self.name}: {e}",
                    exc_info=True,
                )
            finally:
                self._async_executor = None


# copy docstrings defined in core_pydoc.hpp
Application.__doc__ = _Application.__doc__
Application.__init__.__doc__ = _Application.__init__.__doc__


class Fragment(_Fragment):
    def __init__(self, app=None, name="", *args, **kwargs):
        if app is not None and not isinstance(app, _Application):
            raise ValueError(
                "The first argument to a Fragment's constructor must be the Application "
                "to which it belongs."
            )
        # It is recommended to not use super()
        # (https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python)
        _Fragment.__init__(self, self, *args, **kwargs)

        self.name = name
        self.application = app
        # Set the fragment config to the application config.
        if app:
            self.config(app.config())
        self._async_executor = None
        self._async_executor_lock = _threading.Lock()
        self._start_op = None
        # Initialize the Python service registry for PyFragment
        self._python_service_registry = {}

    def compose(self):
        pass

    def run_async(self):
        """Run the fragment asynchronously using a shared executor.

        This method uses a shared ThreadPoolExecutor associated with this
        Application instance. The executor is created on the first call.
        Call `shutdown_async_executor()` when done with async runs
        to clean up resources.

        Returns
        -------
        future : ``concurrent.futures.Future`` object
        """
        # Ensure only one thread creates the executor
        with self._async_executor_lock:
            if self._async_executor is None:
                # Create the executor ONCE
                self._async_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=f"HoloscanFragment_{self.name}_Async"
                )

        # Submit the job to the shared executor
        return self._async_executor.submit(self.run)

    def shutdown_async_executor(self, wait=True):
        """Shuts down the shared asynchronous executor.

        Call this method when the application instance is no longer needed
        and asynchronous runs initiated by `run_async` should terminate.

        Parameters
        ----------
        wait : bool
            If True (default), wait for running tasks to complete before shutting down.
            If False, shut down immediately.
        """
        # Use the lock to prevent race conditions with run_async
        with self._async_executor_lock:
            if self._async_executor is not None:
                # Shutting down async executor
                self._async_executor.shutdown(wait=wait)
                self._async_executor = None

    def start_op(self):
        """Get or create the start operator for this fragment.

        This operator is nothing but the first operator that was added to the fragment.
        It has the name of `<|start|>` and has a condition of `CountCondition(1)`.
        This Operator is used to start the execution of the fragment.
        Entry operators who want to start the execution of the fragment should connect to this
        operator.

        If this method is not called, no start operator is created.
        Otherwise, the start operator is created if it does not exist, and the start operator is
        returned.

        Returns
        -------
        Operator
            The start operator instance. If it doesn't exist, it will be created with
            a CountCondition(1).
        """
        from ..conditions import CountCondition  # noqa: PLC0415

        if not self._start_op:
            self._start_op = Operator(self, CountCondition(self, 1), name=START_OPERATOR_NAME)
            self.add_operator(self._start_op)
        return self._start_op

    def __del__(self):
        # This is best-effort cleanup, not guaranteed to be called reliably.
        # Avoid potentially blocking calls or complex logic here.
        if getattr(self, "_async_executor", None):
            # Non-blocking shutdown is safer in __del__ if possible,
            # but might leave work unfinished or resources dangling longer.
            # Using wait=False might be preferable here, but check implications.
            try:
                self._async_executor.shutdown(wait=False)  # Try non-blocking first
            except Exception as e:
                logger.error(
                    f"Error during __del__ executor shutdown for Fragment {self.name}: {e}",
                    exc_info=True,
                )
            finally:
                self._async_executor = None


# copy docstrings defined in core_pydoc.hpp
Fragment.__doc__ = _Fragment.__doc__
Fragment.__init__.__doc__ = _Fragment.__init__.__doc__


class Subgraph(_Subgraph):
    def __init__(self, fragment: _Fragment | _Subgraph, instance_name: str):
        if not isinstance(fragment, (_Fragment, _Subgraph)):
            raise ValueError(
                "The first argument to a Subgraph's constructor must be the Fragment "
                "(Application) or Subgraph to which it belongs."
            )

        # It is recommended to not use super()
        # (https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python)
        _Subgraph.__init__(self, self, fragment, instance_name)

        # store Fragment as an attribute so it is accessible from Operator constructor, etc.
        if isinstance(fragment, _Subgraph):
            fragment = fragment.fragment

        self.fragment = fragment

        # Compose immediately after Python object is fully constructed
        # This matches the behavior of Fragment::make_subgraph() in C++
        if not self.is_composed():
            self.compose()
            self.set_composed(True)

    def compose(self):
        pass

    def add_flow(
        self,
        upstream: OperatorBase,
        downstream: OperatorBase,
        port_pairs: set[tuple[str, str]] | None = None,
        connector_type: IOSpec.ConnectorType | None = None,
    ):
        """
        Add a flow between components within this Subgraph.

        This method delegates to the fragment's add_flow method, providing a convenient
        way to connect operators and subgraphs within a Subgraph's compose() method.

        Parameters
        ----------
        upstream : Operator or Subgraph
            The upstream component
        downstream : Operator or Subgraph
            The downstream component
        port_pairs : set of tuple of str, optional
            Port connections as (upstream_port, downstream_port) pairs
        connector_type : IOSpec.ConnectorType, optional
            The connector type to use for the connection
        """

        # Make sure operator names are updated to the qualified name via `add_operator`
        #
        # Note: The name will already be prefixed for operators where a Subgraph was passed as the
        # first argument to the constructor, but it is still safe to call `add_operator` as it
        # guards against adding a second copy of the prefix. This will allow robustly making sure
        # the prefix is added even if the Operator was constructed by passing the Fragment as the
        # first argument to the constructor or if the Python bindings of a C++ operator did not use
        # the `get_fragment_ptr_name_pair` utility function to handle the prefixing automatically
        # during operator construction.
        if isinstance(upstream, OperatorBase):
            self.add_operator(upstream)
        if isinstance(downstream, OperatorBase):
            self.add_operator(downstream)

        # Call the PyFragment::add_flow method to handle the connections. This ensures that the
        # overloads handling additions to the PyFragment's python_operator_registry_ are used.
        if connector_type is not None:
            if port_pairs is not None:
                self.fragment.add_flow(upstream, downstream, port_pairs, connector_type)
            else:
                self.fragment.add_flow(upstream, downstream, connector_type)
        else:
            if port_pairs is not None:
                self.fragment.add_flow(upstream, downstream, port_pairs)
            else:
                self.fragment.add_flow(upstream, downstream)

    def set_dynamic_flows(self, op: OperatorBase, func: Callable):
        """Set a callback function to define dynamic flows for an operator at runtime.

        This method allows operators to modify their connections with other operators during execution.
        The callback function is called after the operator executes and can add dynamic flows using
        the operator's `add_dynamic_flow` methods.

        Parameters
        ----------
        op : holoscan.core.Operator
            The operator for which to set dynamic flows.
        dynamic_flow_func : callable
            The callback function that defines the dynamic flows. Takes an operator as input and returns
            ``None``.
        """
        self.fragment.set_dynamic_flows(op, func)


# copy docstrings defined in core_pydoc.hpp
Subgraph.__doc__ = _Subgraph.__doc__
Subgraph.__init__.__doc__ = _Subgraph.__init__.__doc__


class Operator(OperatorBase):
    _readonly_attributes = [
        "fragment",
        "conditions",
        "resources",
        "operator_type",
        "description",
    ]

    def __setattr__(self, name, value):
        if name in self._readonly_attributes:
            raise AttributeError(f'cannot override read-only property "{name}"')
        super().__setattr__(name, value)

    def __init__(self, fragment, *args, **kwargs):
        if not isinstance(fragment, (_Fragment, _Subgraph)):
            raise ValueError(
                "The first argument to an Operator's constructor must be the Fragment "
                "(Application) or Subgraph to which it belongs."
            )
        # It is recommended to not use super()
        # (https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python)
        OperatorBase.__init__(self, self, fragment, *args, **kwargs)
        # Create a PyOperatorSpec object and pass it to the C++ API
        spec = OperatorSpec(fragment=self.fragment, op=self)
        self.spec = spec
        # Call setup method in PyOperator class
        self.setup(spec)

    def setup(self, spec: OperatorSpec):
        """Default implementation of setup method."""
        pass

    def initialize(self):
        """Default implementation of initialize"""
        pass

    def start(self):
        """Default implementation of start"""
        pass

    def compute(self, op_input, op_output, context):
        """Default implementation of compute"""
        pass

    def stop(self):
        """Default implementation of stop"""
        pass


# copy docstrings defined in core_pydoc.hpp
Operator.__doc__ = OperatorBase.__doc__
Operator.__doc__.replace(
    "Base class representing either a wrapped C++ operator or native Python operator.",
    "Native Python operator class.",
)
Operator.__init__.__doc__ = OperatorBase.__init__.__doc__
Operator.__init__.__doc__.replace(
    "Base class representing either a wrapped C++ operator or native Python operator.",
    "Native Python operator class.",
)


class Condition(ConditionBase):
    _readonly_attributes = [
        "fragment",
        "condition_type",
        "description",
    ]

    def __setattr__(self, name, value):
        if name in self._readonly_attributes:
            raise AttributeError(f'cannot override read-only property "{name}"')
        super().__setattr__(name, value)

    def __init__(self, fragment, *args, **kwargs):
        if not isinstance(fragment, (_Fragment, _Subgraph)):
            raise ValueError(
                "The first argument to an Operator's constructor must be the Fragment "
                "(Application) or Subgraph to which it belongs."
            )

        # Extract receiver/transmitter kwargs for automatic argument detection
        # This enables user-supplied condition detection in the C++ backend

        # It is recommended to not use super()
        # (https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python)
        ConditionBase.__init__(self, self, fragment, *args, **kwargs)

        # Create a PyComponentSpec object and pass it to the C++ API
        spec = ComponentSpec(fragment=self.fragment, component=self)
        self.spec = spec
        # Call setup method in PyCondition class
        self.setup(spec)

        # If a "receiver_name" or "transmitter_name" kwarg was passed in so the
        # Operator::find_ports_used_by_condition_args() will avoid adding a default condition to
        # that port, then we should also add a C++ Parameter with that name to avoid a warning
        # being raised about specifying an argument for which there is no corresponding parameter
        # defined.
        receiver_name = kwargs.get("receiver_name")
        transmitter_name = kwargs.get("transmitter_name")
        spec_repr = repr(spec)
        # add Parameter for "receiver_name" if it doesn't already exist
        if receiver_name is not None and "receiver_name" not in spec_repr:
            spec.param("receiver_name", receiver_name)
        # add Parameter for "transmitter_name" if it doesn't already exist
        if transmitter_name is not None and "transmitter_name" not in spec_repr:
            spec.param("transmitter_name", transmitter_name)

    def setup(self, spec: ComponentSpec):
        """Default implementation of setup method."""
        pass

    def initialize(self):
        """Default implementation of initialize"""
        pass

    def update_state(self, timestamp):
        """Default implementation of update_state

        Parameters
        ----------
        timestamp : int
            The timestamp at which the update_state method was called.

        Notes
        -----
        This method is always called by the underlying GXF framework immediately before the
        `Condition.check` method. In some cases, the `Condition.on_execute` method may also wish
        to call this method.
        """
        pass

    def check(self, timestamp: int) -> tuple[SchedulingStatusType, int | None]:
        """Default implementation of check.

        Parameters
        ----------
        timestamp : int
            The timestamp at which the check method is called. This method is called by the
            underlying GXF framework to determine whether an operator is ready to execute.

        Returns
        -------
        status_type: SchedulingStatusType
            The current status of the operator. See the documentation on native condition
            creation for explanations of the various status types.
        target_timestamp: int or None
            Specifies a specific target timestamp at which the operator is expected to be ready.
            This should only be provided if relevant (it helps the underlying framework avoid
            overhead of repeated checks before the target time).

        Notes
        -----
        The method should return SchedulingStatusType.READY when the desired condition has been met.

        The operator will always execute with this default implementation that always execute with
        this default implementation.
        """
        return SchedulingStatusType.READY, None

    def on_execute(self, timestamp):
        """Default implementation of on_execute

        Parameters
        ----------
        timestamp : int
            The timestamp at which the on_execute method was called.

        Notes
        -----
        This method is called by the underlying GXF framework immediately after the
        `Operator.compute` call for the operator to which the condition has been assigned.
        """
        pass


# copy docstrings defined in core_pydoc.hpp
Condition.__doc__ = ConditionBase.__doc__
Condition.__doc__.replace(
    "Base class representing either a wrapped C++ condition or native Python condition.",
    "Native Python condition class.",
)
Condition.__init__.__doc__ = ConditionBase.__init__.__doc__
Condition.__init__.__doc__.replace(
    "Base class representing either a wrapped C++ condition or native Python condition.",
    "Native Python condition class.",
)


class Resource(ResourceBase):
    _readonly_attributes = [
        "fragment",
        "resource_type",
        "description",
    ]

    def __setattr__(self, name, value):
        if name in self._readonly_attributes:
            raise AttributeError(f'cannot override read-only property "{name}"')
        super().__setattr__(name, value)

    def __init__(self, fragment, *args, **kwargs):
        if not isinstance(fragment, (_Fragment, _Subgraph)):
            raise ValueError(
                "The first argument to an Operator's constructor must be the Fragment "
                "(Application) or Subgraph to which it belongs."
            )

        # It is recommended to not use super()
        # (https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python)
        ResourceBase.__init__(self, self, fragment, *args, **kwargs)
        # Create a PyComponentSpec object and pass it to the C++ API
        spec = ComponentSpec(fragment=self.fragment, component=self)
        self.spec = spec
        # Call setup method in PyResource class
        self.setup(spec)

    def setup(self, spec: ComponentSpec):
        """Default implementation of setup method."""
        pass


# copy docstrings defined in core_pydoc.hpp
Resource.__doc__ = ResourceBase.__doc__
Resource.__doc__.replace(
    "Base class representing either a wrapped C++ resource or native Python resource.",
    "Native Python resource class.",
)
Resource.__init__.__doc__ = ResourceBase.__init__.__doc__
Resource.__init__.__doc__.replace(
    "Base class representing either a wrapped C++ resource or native Python resource.",
    "Native Python resource class.",
)


class DefaultFragmentService(_DefaultFragmentService):
    """Base class for fragment services in Python.

    Provides default implementations of virtual methods to avoid
    infinite recursion issues with pybind11 trampolines.
    """

    def __init__(self, resource=None, *args, **kwargs):
        """Initialize the fragment service.

        Parameters
        ----------
        resource : Resource, optional
            The underlying resource for this service.
        """
        # Call the C++ base class constructor
        if resource is not None:
            _DefaultFragmentService.__init__(self, resource, *args, **kwargs)
        else:
            _DefaultFragmentService.__init__(self, *args, **kwargs)
        self._resource_ref = resource

    def resource(self, new_resource=None):
        """Get or set the underlying Resource associated with this service.

        This method is called by the C++ backend.

        Parameters
        ----------
        new_resource : Resource or None
            If provided, sets the resource. If None, acts as getter.

        Returns
        -------
        Resource or None
            The associated resource when called as a getter.
        """
        if new_resource is not None:
            self._resource_ref = new_resource
            # We also need to call the C++ base class's resource setter
            super().resource(new_resource)
        return self._resource_ref


class Tracker:
    """Context manager to add data flow tracking to an application."""

    def __init__(
        self,
        app,
        *,
        filename=None,
        num_buffered_messages=100,
        num_start_messages_to_skip=10,
        num_last_messages_to_discard=10,
        latency_threshold=0,
        is_limited_tracking=False,
    ):
        """
        Parameters
        ----------
        app : holoscan.core.Application
            on which flow tracking should be applied.
        filename : str or None, optional
            If none, logging to file will be disabled. Otherwise, logging will
            write to the specified file.
        num_buffered_messages : int, optional
            Controls the number of messages buffered between file writing when
            `filename` is not ``None``.
        num_start_messages_to_skip : int, optional
            The number of messages to skip at the beginning of the execution. This does not affect
            the log file or the number of source messages metric.
        num_last_messages_to_discard : int, optional
            The number of messages to discard at the end of the execution. This does not affect
            the log file or the number of source messages metric.
        latency_threshold : int, optional
            The minimum end-to-end latency in milliseconds to account for in the end-to-end
            latency metric calculations.
        is_limited_tracking : bool, optional
            If true, the tracking is limited to root and leaf nodes, minimizing the timestamps by
            avoiding intermediate operators.
        """
        self.app = app

        # Check the number of fragment nodes to see if it is a distributed app.
        # Use compose_graph(), not compose() to protect against repeated compose() calls.
        self.app.compose_graph()
        self.is_distributed_app = len(app.fragment_graph.get_nodes()) > 0

        self.enable_logging = filename is not None
        if self.enable_logging:
            self.logging_kwargs = dict(
                filename=filename,
                num_buffered_messages=num_buffered_messages,
            )
        self.tracker_kwargs = dict(
            num_start_messages_to_skip=num_start_messages_to_skip,
            num_last_messages_to_discard=num_last_messages_to_discard,
            latency_threshold=latency_threshold,
            is_limited_tracking=is_limited_tracking,
        )

    def __enter__(self):
        if self.is_distributed_app:
            self.trackers = self.app.track_distributed(**self.tracker_kwargs)
            for tracker in self.trackers.values():
                if self.enable_logging:
                    tracker.enable_logging(**self.logging_kwargs)
            return self.trackers
        else:
            self.tracker = self.app.track(**self.tracker_kwargs)
            if self.enable_logging:
                self.tracker.enable_logging(**self.logging_kwargs)
            return self.tracker

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.enable_logging:
            if self.is_distributed_app:
                for tracker in self.trackers.values():
                    tracker.end_logging()
            else:
                self.tracker.end_logging()


_registry_context = _RegistryContext()
io_type_registry = _registry_context.registry()

_register_types(io_type_registry)
