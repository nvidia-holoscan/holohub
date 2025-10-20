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

import importlib
from typing import Any, TypeVar

T = TypeVar("T")


class LazyRaise:
    """A proxy object that raises the stored exception when accessed."""

    def __init__(self, exception: Exception):
        self._exception = exception

    def __getattr__(self, name: str) -> None:
        """Raises the stored exception when any attribute is accessed."""
        raise self._exception

    def __call__(self, *args, **kwargs) -> None:
        """Raises the stored exception when the object is called."""
        raise self._exception

    def __getitem__(self, item) -> None:
        """Raises the stored exception when indexed."""
        raise self._exception

    def __iter__(self) -> None:
        """Raises the stored exception when iterated."""
        raise self._exception

    def __repr__(self) -> str:
        """Returns a string representation of the stored exception."""
        return f"LazyRaise({self._exception})"


def lazy_import(module_name: str) -> Any:
    """
    Import an optional module specified by module_name.

    If the module cannot be imported, returns a proxy object that raises
    the ImportError when accessed.

    Args:
        module_name: name of the module to be imported.

    Returns:
        The imported module or a proxy object that raises ImportError when accessed
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as import_error:
        return LazyRaise(import_error)
