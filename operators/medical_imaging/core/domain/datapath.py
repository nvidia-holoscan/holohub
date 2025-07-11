#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path
from typing import Dict, Optional, Union

from operators.medical_imaging.exceptions import IOMappingError as IOMappingError
from operators.medical_imaging.exceptions import ItemNotExistsError as ItemNotExistsError

from .domain import Domain as Domain


class DataPath(Domain):
    def __init__(
        self, path: Union[str, Path], read_only: bool = False, metadata: Optional[Dict] = None
    ):
        """Initializes a DataPath object.

        Args:
            path (Union[str, Path]): Path to the data file/directory.
            read_only (bool): True if the the file/directory path cannot be modified.
            metadata (Optional[Dict]): A metadata.
        """
        super().__init__(metadata=metadata)
        self._path: Path = Path(path)
        self._read_only: bool = read_only

    @property
    def path(self):
        """Returns the path of the data file/directory."""
        return self._path

    @path.setter
    def path(self, val):
        if self._read_only:
            raise IOMappingError("This DataPath is read-only.")
        self._path = Path(val)

    def to_absolute(self):
        """Convert the internal representation of the path to an absolute path."""
        if not self._path.is_absolute():
            self._path = self._path.absolute()


class NamedDataPath(Domain):
    """A data path dictionary with name as key and data path as value.

    This class is used to store data paths and the provided name of each data path is unique.

    A data path for a name is accessible by calling the `get()` method with the name.

    If only one data path is available and the name is not specified, the data path is returned.
    """

    def __init__(self, paths: Dict[str, DataPath], metadata: Optional[Dict] = None):
        super().__init__(metadata=metadata)
        self._paths = paths

    def get(self, name: Optional[str] = "") -> DataPath:
        if name not in self._paths:
            if name == "" and len(self._paths) == 1:
                return next(iter(self._paths.values()))
            else:
                raise IOMappingError(
                    f"{name!r} is not a valid name. It should be one of ({', '.join(self._paths.keys())})."
                )
        else:
            datapath = self._paths.get(name)
            if not datapath:
                raise ItemNotExistsError(f"A DataPath instance for {name!r} does not exist.")
            return datapath
