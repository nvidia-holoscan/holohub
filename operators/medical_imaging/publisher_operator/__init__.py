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
Publisher Operator Package.

This package provides the PublisherOperator for publishing medical imaging data.
The operator enables flexible publishing of processed medical imaging data for visualization, storage, or further analysis.

.. autosummary::
    :toctree: _autosummary

    PublisherOperator
"""

from .publisher_operator import PublisherOperator

__all__ = ["PublisherOperator"]
