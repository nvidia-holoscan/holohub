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
import uuid
from typing import Dict, List, Optional, Union


class FHIRQuery:
    """
    Helper class for performing an FHIR query.

    identifier: Patient.identifier (MRN, PatientID)
    patient_name: Patient.name
    patient_dob: Patient.birthdate: must be ISO-8601 format. e.g. YYYY-MM-DD
    resources_to_retrieve: limits the type of FHIR resources associated for the patient
    start_date: limits the start date of the records
    end_date: limits the end date of the records
    """

    def __init__(
        self,
        request_id: Optional[str] = str(uuid.uuid4()),
        identifier: Optional[str] = "",
        patient_name: Optional[str] = "",
        patient_dob: Optional[str] = "",
        resources_to_retrieve: Optional[List[str]] = [
            "Observation",
            "ImagingStudy",
            "FamilyMemberHistory",
            "Condition",
            "DiagnosticReport",
            "DocumentReference",
        ],
        start_date: Optional[str] = "",
        end_date: Optional[str] = "",
    ):
        self.request_id = request_id
        self.patient_name = patient_name
        self.patient_dob = patient_dob
        self.identifier = identifier
        self.resources_to_retrieve = resources_to_retrieve
        self.start_date = start_date
        self.end_date = end_date

    def __str__(self) -> str:
        values = []
        if self.identifier:
            values.append(f"\tMRN: {self.identifier}")
        if self.patient_name:
            values.append(f"\tName: {self.patient_name}")
        if self.patient_dob:
            values.append(f"\tDOB: {self.patient_dob}")
        if self.resources_to_retrieve:
            values.append(f"\tResources: {self.resources_to_retrieve}")
        if self.start_date:
            values.append(f"\tStart Date: {self.start_date}")
        if self.end_date:
            values.append(f"\tEnd Date: {self.end_date}")

        return "\n".join(values)

    def to_json(self):
        return json.dumps(vars(self))

    def get_everything_fitler(self, patient_url):
        url = f"{patient_url}/$everything?"

        if self.resources_to_retrieve:
            url += "_type=" + ",".join(self.resources_to_retrieve) + "&"

        if self.start_date:
            url += f"start={self.start_date}&"

        if self.end_date:
            url += f"end={self.end_date}&"

        return url

    def get_patient_query(self):
        parameters = []

        if self.patient_name:
            for name in self.patient_name.split(" "):
                parameters.append(f"name={name}")
        if self.identifier:
            parameters.append(f"identifier={self.identifier}")
        if self.patient_dob:
            parameters.append(f"birthdate={self.patient_dob}")

        return "/Patient?" + "&".join(parameters)

    def _add_to_dict_if_not_empty(self, query: dict, value: str, key: str):
        if value and not value.isspace():
            query[key] = value

    @staticmethod
    def from_json(json_dct: Union[str, Dict]):
        if isinstance(json_dct, str):
            json_dct = json.loads(json_dct)

        return FHIRQuery(
            json_dct.get("request_id", ""),
            json_dct.get("identifier", ""),
            json_dct.get("patient_name", ""),
            json_dct.get("patient_dob", ""),
            json_dct.get("resources_to_retrieve", ""),
            json_dct.get("start_date", ""),
            json_dct.get("end_date", ""),
        )
