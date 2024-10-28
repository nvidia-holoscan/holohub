#!/bin/bash
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

from typing import Any, Union

from operators.medical_imaging.utils.importutil import optional_import

from .domain import Domain

DataElement_, dataelement_ok_ = optional_import("pydicom", name="DataElement")
# Dynamic class is not handled so make it Any for now: https://github.com/python/mypy/issues/2477
DataElement: Any = DataElement_ if dataelement_ok_ else Any
Dataset_, dataset_ok_ = optional_import("pydicom", name="Dataset")
# Dynamic class is not handled so make it Any for now: https://github.com/python/mypy/issues/2477
Dataset: Any = Dataset_ if dataset_ok_ else Any
Tag_, tag_ok_ = optional_import("pydicom.tag", name="Tag")
# Dynamic class is not handled so make it Any for now: https://github.com/python/mypy/issues/2477
Tag: Any = Tag_ if tag_ok_ else Any


class DICOMSOPInstance(Domain):
    """This class represents a SOP Instance.

    An attribute can be looked up with a slice ([group_number, element number]).
    """

    def __init__(self, native_sop):
        super().__init__(None)
        self._sop: Any = native_sop

    def get_native_sop_instance(self):
        return self._sop

    def __getitem__(self, key: Union[int, slice, Tag]) -> Union[Dataset, DataElement]:
        return self._sop.__getitem__(key)

    def get_pixel_array(self):
        return self._sop.pixel_array

    def __str__(self):
        result = "---------------" + "\n"

        return result
