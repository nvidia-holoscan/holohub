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

import signal
import sys
import time
import zmq
from threading import Thread

from operators.ehr_query_llm.fhir.ehr_query import FHIRQuery
from operators.ehr_query_llm.fhir.ehr_response import FHIRQueryResponse
from operators.ehr_query_llm.message_handling import MessageReceiver

topic_request = "ehr-request"
topic_response = "ehr-response"
busy = False
option = None

patient_name = "John Doe"
resource_id_patient = "108915cc-b9f2-47db-ad4b-0143151f4f61"
start_date_default = "2000-08-31"
end_date_Default = "2020-08-31"
start_date_procedures = "2024-03-01"

# Flag to enable saving the last non-empty query response
saving_query_response = False

def print_patient(patient_id, records):
    print(f"==> Found {len(records)} records for Patient ID: {patient_id}:")

    resrouce_type_counts = {}
    for record in records:
        if "resource" in record and "resourceType" in record["resource"]:
            resource_type = record["resource"]["resourceType"]
            if resource_type in resrouce_type_counts:
                resrouce_type_counts[resource_type] += 1
            else:
                resrouce_type_counts[resource_type] = 1
        else:
            show_prompt()

    for key in sorted(resrouce_type_counts):
        print(f"     {key}: {resrouce_type_counts[key]}")

    print("\n")


def print_latest_diagnostic_report(resources):
    print("=====================================================")
    patient_resources = resources["108915cc-b9f2-47db-ad4b-0143151f4f61"]
    procedures = [
        x["resource"]
        for x in patient_resources
        if x["resource"]["resourceType"] == "Procedure"
        and x["resource"]["id"] == "b3906aa2-27a9-4d7d-b458-e288d13231f0"
    ]

    diagnostic_reports = [
        x["resource"]
        for x in patient_resources
        if x["resource"]["resourceType"] == "DiagnosticReport"
        and x["resource"]["id"] == "350f79b7-5cfc-4a1a-bbf5-557d7db53659"
    ]

    if len(diagnostic_reports) == 0:
        raise Exception("No reports found for patient")

    diagnostic_report = diagnostic_reports[0]
    print(f"Report ID: {diagnostic_report['id']}")
    print(f"Date: {diagnostic_report['date']}")
    if "conclusion" in diagnostic_report:
        print(f"Conclusion: {diagnostic_report['conclusion']}")
    for report in diagnostic_report["reports"]:
        print("=====================================================")
        print(f"Report: {report['data']}")
        print("=====================================================")
    for notes, image in diagnostic_report["key_images"]:
        print("=====================================================")
        print(f"Notes: {notes}")
        print(f"Image: {image}")
        print("=====================================================")

def data_handler():
    global busy
    global option
    receiver = MessageReceiver(topic_response, "tcp://localhost:5601")
    while True:
        response = receiver.receive_json()

        # Only saving the last none empty message received, if needed
        if saving_query_response:
            if isinstance(response, str) or (isinstance(response, dict) and response.keys()):
                with open("fhir_response_jason.txt", "w+") as f:
                    f.write(response)

        fhir_response = FHIRQueryResponse.from_json(response)
        if len(fhir_response.patient_resources) > 0:
            if option == "4":
                try:
                    print_latest_diagnostic_report(fhir_response.patient_resources)
                except Exception as e:
                    print(e)
            else:
                for key in fhir_response.patient_resources:
                    print_patient(key, fhir_response.patient_resources[key])

            busy = False
            show_prompt()
        time.sleep(1)


def sender():
    global busy
    global option

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5600")
    while True:
        option = input()

        if not busy:
            busy = True
            if option == "1":
                query = FHIRQuery(
                    patient_name=patient_name,
                    resources_to_retrieve=[],
                )
            elif option == "2":
                query = FHIRQuery(
                    patient_name=patient_name,
                    start_date=start_date_default,
                    end_date=end_date_Default,
                    resources_to_retrieve=[],
                )
            elif option == "4":
                query = FHIRQuery(
                    identifier=resource_id_patient,
                    start_date=start_date_procedures,
                    resources_to_retrieve=["Procedure", "DiagnosticReport"],
                )
            else:
                query = FHIRQuery(
                    patient_name=patient_name, start_date=start_date_default, end_date=end_date_Default,
                )

            print("Sending query:")
            print(query)
            print("")
            socket.send_multipart([topic_request.encode("utf-8"), query.to_json().encode("utf-8")])


def stop(signum=None, frame=None):
    print("Stopping...")
    sys.exit()

def show_prompt():
    print(
        f"""Enter an option:
        [1] all records
        [2] {start_date_default} to {end_date_Default}
        [3] {start_date_default} to {end_date_Default} Observation, ImagingStudy, FamilyMemberHistory, Condition, DiagnosticReport, DocumentReference (default)
        [4] Procedure & DiagnosticReport for {patient_name}
        """
    )


if __name__ == "__main__":
    global t_receiver
    global t_sender
    signal.signal(signal.SIGINT, stop)

    t_receiver = Thread(target=data_handler)
    t_receiver.daemon = True
    t_receiver.start()
    t_sender = Thread(target=sender)
    t_sender.daemon = True
    t_sender.start()
    show_prompt()
    t_receiver.join()
    t_sender.join()
