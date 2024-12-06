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

# BE SURE TO HAVE FHIR APP RUNNING BEFORE STARTING THIS SCRIPT

import logging
import signal
import sys
import time
from threading import Thread
from typing import Optional

import zmq
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

from operators.ehr_query_llm.fhir.ehr_query import FHIRQuery
from operators.ehr_query_llm.fhir.ehr_response import FHIRQueryResponse
from operators.ehr_query_llm.message_handling import MessageReceiver

# FHIR global vars
topic_request = "ehr-request"
topic_response = "ehr-response"
busy = False
resources_to_retrieve = ["Observation", "FamilyMemberHistory", "Condition", "AllergyIntolerance"]

# EHR Patient ID for John Doe
mrn = "108915cc-b9f2-47db-ad4b-0143151f4f61"

# LLM summary global vars
ehr_data = None

# Stores EHR requests initiated from this instance
ehr_request_ids = []

# Vector db global vars
EHR_FINETUNED_MODEL = "/workspace/volumes/models/bge-large-ehr-finetune"
CACHE_FOLDER = "/workspace/volumes/models"
PERSISTENT_FOLDER = "/workspace/holohub/applications/ehr_query_llm/lmm/rag/ehr/db"


def get_ehr_data(
    allow_reqested_only: Optional[bool] = True,
):
    global ehr_data
    global ehr_request_ids

    receiver = MessageReceiver(topic_response, "tcp://localhost:5601")
    while True:
        response = receiver.receive_json(blocking=True)
        try:
            ehr_response = FHIRQueryResponse.from_json(response)
            logging.info(f"Got a response for request, id: {ehr_response.request_id}")
            if ehr_response.request_id.casefold() not in ehr_request_ids:
                # For now, only support request/response scenario
                # In the future, may need to support unsolicilated messages
                logging.warning(
                    f"Processing response for a untracked request, id {ehr_response.request_id}"
                )
                if allow_reqested_only:
                    continue
            else:
                ehr_request_ids.remove(ehr_response.request_id.strip())
                logging.info(f"Removed processed tracked request, id {ehr_response.request_id}")

            if len(ehr_response.patient_resources) > 0:
                for key in ehr_response.patient_resources:
                    logging.info(f"Documents retrieved: {len(ehr_response.patient_resources[key])}")
                    ehr_data = ehr_response.patient_resources
                return
        except Exception as ex:
            logging.error(f"Error parsing FHIR response {ex}")
        time.sleep(1)


def send_request():
    global ehr_request_ids
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5600")
    # Give ZeroMQ time to bind TODO determine why this is needed
    time.sleep(1)
    logging.info("Sending request to FHIR server...")
    query = FHIRQuery(identifier=mrn, resources_to_retrieve=resources_to_retrieve)
    ehr_request_ids.append(query.request_id.casefold())
    logging.info(f"Sent request with id {query.request_id}")
    socket.send_multipart([topic_request.encode("utf-8"), query.to_json().encode("utf-8")])


def stop(signum=None, frame=None):
    logging.info("Stopping...")
    sys.exit()


def create_db(documents):
    """
    Creates a Vector DB using the provided documents
    """
    model_name = EHR_FINETUNED_MODEL
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder=CACHE_FOLDER,
    )
    chroma_db = Chroma.from_documents(
        documents, embedding_model, persist_directory=PERSISTENT_FOLDER
    )
    chroma_db.persist()


def update_ehr_dict(flattened_ehr, summary, date_str):
    """
    If the summary doesn't exist yet, create a new entry with the provided date. Otherwise,
    append the date to the existing entry.
    """
    if date_str:
        if summary not in flattened_ehr:
            flattened_ehr[summary] = [date_str[:10]]
        else:
            flattened_ehr[summary].append(date_str[:10])
    else:
        if summary not in flattened_ehr:
            flattened_ehr[summary] = ["N/A"]
    return flattened_ehr


def flatten_ehr(ehr_dict):
    """
    Used to flatten the parsed EHR resources and make it easier for the LLM to understand.
    Additionally, if duplicate summaries are found, this function only keeps the original summary and appeneds
    the duplicated entry's date to the list of dates for the given summary. (This dramatically reduces the number of documents)
    """
    flattened_ehr = {}
    num_entry_missing_categories = 0
    num_entry_no_date = 0

    for _, resources in ehr_dict.items():
        for entry in resources:
            resource = entry["resource"]
            # Parse Condition
            if resource["resourceType"] == "Condition":
                summary = f"{resource['clinical_status']} condition: {resource['condition']}"
                flattened_ehr = update_ehr_dict(flattened_ehr, summary, resource["recorded_date"])
            # Parse Observation
            elif resource["resourceType"] == "Observation":

                # Collect stats of entries missing key attributes
                if "categories" not in [x.lower() for x in resource.keys()]:
                    num_entry_missing_categories += 1
                    resource["categories"] = ""
                    logging.warning(
                        f"Missing 'categories' attribute in resource entry: {entry.get('fullUrl', '')}"
                    )
                if "date" not in [x.lower() for x in resource.keys()]:
                    num_entry_no_date += 1
                    resource["date"] = ""
                    logging.warning(
                        f"Missing 'date' attribute in resource entry: {entry.get('fullUrl', '')}"
                    )

                # Separate long surveys into individual QA pairs
                if resource["categories"] == "survey" and isinstance(resource["observation"], list):
                    for qa_pair in resource["observation"]:
                        summary = f"{resource['status']} {resource['categories']}:\n\tsurvey response: {qa_pair[0] if isinstance(qa_pair, list) else qa_pair}"
                        flattened_ehr = update_ehr_dict(flattened_ehr, summary, resource["date"])
                # Parse single observations individually
                else:
                    summary = f"{resource['status']} {resource['categories']}:\n\tobservation:"
                    if isinstance(resource["observation"], list):
                        for item in resource["observation"]:
                            summary += f"\n\t\t{item}"
                    else:
                        summary += f"\n\t\t{resource['observation']}"
                    flattened_ehr = update_ehr_dict(flattened_ehr, summary, resource["date"])
            # Parse FamilyMemberHistory
            elif resource["resourceType"] == "FamilyMemberHistory":
                summary = f"The {'deceased' if resource['deceased'] else 'living'} {'/'.join(resource['relationship'])} has {' ,'.join(resource['conditions'])}"
                # TODO Update FHIR parsing to include date
                flattened_ehr = update_ehr_dict(flattened_ehr, summary, None)
            elif resource["resourceType"] == "AllergyIntolerance":
                summary = f"{resource['clinical_status'].capitalize()} allergy intolerance:\n\tAllergy: {', '.join([allergy['display'] for allergy in resource['allergy_intolerance']])}\n\tCriticality: {resource['criticality']}"
                flattened_ehr = update_ehr_dict(flattened_ehr, summary, None)

    return flattened_ehr


def create_db_docs(flattened_ehr):
    """
    Creates LangChain Documents from the flattened EHR entries.
    This function takes entries with multiple date entries and represents them as a range
    """
    documents = []
    i = 0
    for summary, dates in flattened_ehr.items():
        num_dates = len(dates)
        if num_dates == 1:
            summary += f"\n\tDate: {dates[0]}"
        else:
            dates.sort()
            summary += f"\n\tDate range: {dates[0]} - {dates[-1]}"
            summary += f"\n\tTotal occurrences: {num_dates}"

        new_doc = Document(page_content=summary, seq_num=i)
        documents.append(new_doc)
        i += 1

    return documents


def main():
    logging.basicConfig(level=logging.INFO)
    global ehr_data
    global t_receiver
    global t_sender
    signal.signal(signal.SIGINT, stop)

    t_receiver = Thread(target=get_ehr_data)
    t_receiver.daemon = True
    t_receiver.start()
    t_sender = Thread(target=send_request)
    t_sender.daemon = True
    t_sender.start()
    t_receiver.join()

    flattened_ehr = flatten_ehr(ehr_data)
    documents = create_db_docs(flattened_ehr)
    logging.info(f"Total DB documents: {len(flattened_ehr)}")
    logging.info("Creating EHR vector db...")
    start_time = time.time()
    create_db(documents)
    end_time = time.time()
    total_time = end_time - start_time
    logging.info("Done!\n")
    logging.info(f"Total time to build db: {total_time}")


if __name__ == "__main__":
    main()
