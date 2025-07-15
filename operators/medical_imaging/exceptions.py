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


class MedImagingError(Exception):
    """Base class for exceptions in this module."""

    pass


class ItemAlreadyExistsError(MedImagingError):
    """Raises when an item already exists in the container."""

    pass


class ItemNotExistsError(MedImagingError):
    """Raises when an item does not exist in the container."""

    pass


class IOMappingError(MedImagingError):
    """Raises when IO mapping is missing or invalid."""

    pass


class UnknownTypeError(MedImagingError):
    """Raises when unknown/wrong type/name is specified."""

    pass


class WrongValueError(MedImagingError):
    """Raises when wrong value is specified."""

    pass


class UnsupportedOperationError(MedImagingError):
    """Raises when unsupported operation is requested."""

    pass
