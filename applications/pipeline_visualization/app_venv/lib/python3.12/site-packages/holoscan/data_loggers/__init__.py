"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""This module provides a Python API to underlying C++ API Operators.

.. autosummary::

    holoscan.data_loggers.AsyncConsoleLogger
    holoscan.data_loggers.BasicConsoleLogger
    holoscan.data_loggers.GXFConsoleLogger
    holoscan.data_loggers.SimpleTextSerializer
"""

# Eager import of SimpleTextSerializer symbol required by console logger constructors
from holoscan.data_loggers.basic_console_logger import SimpleTextSerializer

# Define data logger modules and classes for lazy loading
_DATA_LOGGER_MODULES = {
    "basic_console_logger": ["BasicConsoleLogger", "GXFConsoleLogger"],
    "async_console_logger": ["AsyncConsoleLogger"],
}
_DATA_LOGGERS = [item for sublist in _DATA_LOGGER_MODULES.values() for item in sublist]
_LOADED_DATA_LOGGERS = {}
# expose both data logger modules and data logger classes for consistency with operators
__all__ = list(_DATA_LOGGER_MODULES.keys()) + _DATA_LOGGERS


# Autocomplete
def __dir__():
    return __all__


# Lazily load modules and classes
def __getattr__(attr):
    # Get submodule, import if needed
    def getsubmodule(name):
        import importlib
        import sys

        module_name = f"{__name__}.{name}"
        if module_name in sys.modules:  # cached
            module = sys.modules[module_name]
        else:
            module = importlib.import_module(module_name)  # import
            sys.modules[module_name] = module  # cache
        return module

    # Return submodule
    if attr in _DATA_LOGGER_MODULES:
        return getsubmodule(attr)

    # Return cached operator class
    if attr in _LOADED_DATA_LOGGERS:
        return _LOADED_DATA_LOGGERS[attr]

    # Get new operator class
    if attr in _DATA_LOGGERS:
        # Search for submodule that holds it
        for module_name, values in _DATA_LOGGER_MODULES.items():
            if attr in values:
                operator = getattr(getsubmodule(module_name), attr)  # retrieve from submodule
                _LOADED_DATA_LOGGERS[attr] = operator  # cache
                return operator

    raise AttributeError(f"module {__name__} has no attribute {attr}")
