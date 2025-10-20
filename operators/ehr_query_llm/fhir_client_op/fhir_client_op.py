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
from typing import Dict, List

import requests
from holoscan.core import ConditionType, Fragment, Operator, OperatorSpec

from operators.ehr_query_llm.fhir.ehr_query import FHIRQuery
from operators.ehr_query_llm.fhir.ehr_response import FHIRQueryResponse
from operators.ehr_query_llm.fhir.exceptions import InvalidRequestBodyError
from operators.ehr_query_llm.fhir.token_provider import TokenProvider

# WARNING: Disable validation warnings for self-signed server certificates.
#          Use this for trusted servers in demo/dev only
requests.packages.urllib3.disable_warnings()


class FhirClientOperator(Operator):
    """
    Named inputs:
        request: a JSON representation of the EHRQuery object

    Named output:
        out: a FHIRQueryResponse object containing the original request ID and all matching patient and their medical records;
             key=Patient ID/MRN
             value=list of Python dictionary objects where each dict object represents a FHIR resource object
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        fhir_endpoint: str = "http://localhost:8080/",
        token_provider: TokenProvider = None,
        verify_cert: bool = True,
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
        self._token_provider = token_provider  # If None, assume no auth token required.
        self._verify_cert = verify_cert  # True to verify server cert

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("request")
        spec.output("out").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        """
        Pulls the next message in the queue, performs a QIDO query and emits all study instance UIDs
        found in the response.

        Raises:
            InvalidRequestBodyError: when the message received from message queue is malformed.
        """
        try:
            self._logger.debug("FHIR Client op processing request...")
            request_str = op_input.receive("request")
            query_parameters = FHIRQuery.from_json(request_str)
        except Exception as ex:
            raise InvalidRequestBodyError(request_str, ex)

        start = pc()
        try:
            patient_urls = self._find_patients(query_parameters)
            patient_resources = {}
            for id, url in patient_urls.items():
                resources = []
                self._fetch_patient_resources(query_parameters, id, url, resources)
                patient_resources[id] = resources

            if patient_resources:
                self._logger.info(
                    f"{query_parameters.request_id}: Found {len(patient_resources.keys())} patient(s) with {sum(len(item) for item in patient_resources.values())} matching FHIR resources."
                )
                op_output.emit(
                    FHIRQueryResponse(query_parameters.request_id, patient_resources), "out"
                )

        except Exception as ex:
            self._logger.error(
                f"{query_parameters.request_id}: Error performing FHIR query", str(ex)
            )

        end = pc()
        self._logger.info(
            f"{query_parameters.request_id}: FHIR query elapsed: {end - start} seconds"
        )

    def _find_patients(self, query_parameters: FHIRQuery) -> Dict[str, str]:
        request_url = self._fhir_endpoint + query_parameters.get_patient_query()

        self._logger.debug(f"{query_parameters.request_id}: Querying patient from {request_url}")
        response = requests.get(
            request_url,
            headers=self._create_headers(),
            verify=self._verify_cert,
        )
        response.raise_for_status()
        data = response.json()

        patient_urls = {}
        if data and "entry" in data:
            entries = data.get("entry")

            for entry in entries:
                if "resource" in entry:
                    resource = entry.get("resource")
                    if "id" in resource:
                        id = resource.get("id")
                if "fullUrl" in entry:
                    request_url = query_parameters.get_everything_filter(entry.get("fullUrl"))

                if id and request_url:
                    patient_urls[id] = request_url
                elif id:
                    self._logger.warn(
                        f"{query_parameters.request_id}: Patient entry missing 'fullUrl': {entry}"
                    )
                else:
                    self._logger.warn(
                        f"{query_parameters.request_id}: Patient entry missing 'id': {entry}"
                    )
        else:
            self._logger.warn(f"{query_parameters.request_id}: No matching patient found!")

        return patient_urls

    def _fetch_patient_resources(
        self, query_parameters: FHIRQuery, id: str, url: str, entries: List[str]
    ):
        self._logger.debug(
            f"{query_parameters.request_id}: Fetching resources for patient {id} from {url}"
        )
        try:
            response = requests.get(
                url,
                headers=self._create_headers(),
                verify=self._verify_cert,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 500 and len(entries) > 0:
                return
            raise e

        data = response.json()

        if "entry" in data:
            entries.extend(data["entry"])

        if "link" in data:
            next_page = next(
                (item for item in data.get("link") if item.get("relation") == "next"), None
            )
            if next_page:
                self._fetch_patient_resources(query_parameters, id, next_page["url"], entries)

    def _create_headers(self):
        """Populates the header for the FHIR requests.

        Content type is expected to be fixed FHIR JSON, while the authorization header is
        populated if the object to acquire the token has been provided.
        """
        req_headers = {"content-type": "application/fhir+json"}
        if self._token_provider:
            req_headers["Authorization"] = self._token_provider.authorization_header

        return req_headers
