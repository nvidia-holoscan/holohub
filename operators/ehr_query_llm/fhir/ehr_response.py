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

import json
from typing import Dict, Union


class FHIRQueryResponse:
    """
    Encapsulates a FHIR query response.
    """

    @property
    def request_id(self) -> str:
        """The original request ID"""
        return self._request_id

    @property
    def patient_resources(self) -> str:
        """
        dictionary containing all matching patient and their medical records;
             key=Patient ID/MRN
             value=list of Python dictionary objects where each dict object represents a FHIR resource object
        """
        return self._patient_resources

    def __init__(self, request_id, patient_resources):
        self._request_id = request_id
        self._patient_resources = patient_resources

    def to_json(self):
        return json.dumps(vars(self))

    @staticmethod
    def from_json(json_dct: Union[str, Dict]):
        if isinstance(json_dct, str):
            json_dct = json.loads(json_dct)

        return FHIRQueryResponse(
            json_dct.get("_request_id", ""),
            json_dct.get("_patient_resources", ""),
        )
