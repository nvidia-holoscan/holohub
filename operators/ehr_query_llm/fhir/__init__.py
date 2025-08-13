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

"""
FHIR Utilities Package

This package provides utilities for working with FHIR (Fast Healthcare Interoperability Resources)
data including query construction, response handling, resource sanitization, and token management.
"""

from .ehr_query import FHIRQuery
from .ehr_response import FHIRQueryResponse
from .exceptions import FHIRQueryError, InvalidRequestBodyError
from .resource_sanitizer import FHIRResourceSanitizer
from .token_provider import TokenProvider

__all__ = [
    "FHIRQuery",
    "FHIRQueryResponse",
    "FHIRQueryError",
    "InvalidRequestBodyError",
    "FHIRResourceSanitizer",
    "TokenProvider",
]
