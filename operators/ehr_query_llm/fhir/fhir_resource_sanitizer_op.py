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


import logging
from time import perf_counter as pc


from holoscan.core import (
    ConditionType,
    Fragment,
    Operator,
    OperatorSpec,
)

from operators.ehr_query_llm.fhir.ehr_response import FHIRQueryResponse
from operators.ehr_query_llm.fhir.resource_sanitizer import FHIRResourceSanitizer


class FhirResourceSanitizerOp(Operator):
    """
    Named inputs:
        requests: a FHIRQueryResponse object

    Named output:
        out: a FHIRQueryResponse object with sanitized medical records
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        fhir_endpoint: str = "http://localhost:8080/",
        **kwargs,
    ):
        """An operator that queries FHIR service based on information received via
           messaging queue.

        Args:
            fhir_endpoint (str): FHIR endpoint
        Raises:
            ValueError: if queue_policy is out of range.
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

        self._fhir_endpoint = fhir_endpoint
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("records")
        spec.output("out").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        """
        Sanitizes a given FHIR resource.
        If the input content equals "__COMPLETE__" then the store patient records are emitted

        """

        start = pc()
        sanitized_patient_records = {}
        patient_records = op_input.receive("records")
        for patient in patient_records.patient_resources.keys():
            sanitized_patient_records[patient] = []
            for record in patient_records.patient_resources[patient]:
                try:
                    sanitized_record = FHIRResourceSanitizer.sanitize(record)
                    if sanitized_record:
                        sanitized_patient_records[patient].append(sanitized_record)
                except NotImplementedError as e:
                    self._logger.warning(e)

        op_output.emit(
            FHIRQueryResponse(patient_records.request_id, sanitized_patient_records).to_json(),
            "out",
        )
        end = pc()
        self._logger.info(
            f"{patient_records.request_id}: FHIR sanitize op elapsed: {end - start} seconds"
        )
