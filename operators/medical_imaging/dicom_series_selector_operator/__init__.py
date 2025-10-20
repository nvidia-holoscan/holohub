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

"""
DICOM Series Selector Operator Package.

This package provides the DICOMSeriesSelectorOperator for selecting specific DICOM series from a set of studies.
The operator enables filtering and selection of relevant DICOM series for downstream processing.

.. autosummary::
    :toctree: _autosummary

    DICOMSeriesSelectorOperator
"""

from .dicom_series_selector_operator import DICOMSeriesSelectorOperator

__all__ = ["DICOMSeriesSelectorOperator"]
