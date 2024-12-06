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
import os
import time

from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

# The collection name for the set of docs in the vector database.
# Magic string not to be used, as it often causes issues.
COLLECTION_NAME = "ehr_rag"

# Path to the EHR data as input
EHR_DATA_JSON = "/workspace/holohub/applications/ehr_query_llm/lmm/rag/ehr/ehr_data.json"

# Path to the downloaded fine tuned model
EHR_FINETUNED_MODEL = "/workspace/volumes/models/bge-large-ehr-finetune"
CACHE_FOLDER = "/workspace/volumes/models"

# Persistent storage folder for the Vector DB
PERSISTENT_FOLDER = "/workspace/holohub/applications/ehr_query_llm/lmm/rag/ehr/db"


def get_ehr_data():
    with open(EHR_DATA_JSON) as f:
        ehr_data = json.load(f)
    return ehr_data


def create_db(documents, persist_directory):
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

    ids = [str(i) for i in range(1, len(documents) + 1)]

    # Delete the vector db if it already exists
    if os.path.exists(persist_directory + "choma.sqlite3"):
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
        )
        for id, document in zip(ids, documents):
            db.update_document(id, document)
    else:
        chroma_db = Chroma.from_documents(
            documents,
            embedding_model,
            persist_directory=persist_directory,
            collection_name=COLLECTION_NAME,
            ids=ids,
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

    Also, this function assumes all the referenced keys exist, and if not, will simply throw up and let the user know!
    better than suppressing the errors of missing key/value pairs, for this program's intended use.
    """
    flattened_ehr = {}
    num_entry_missing_categories = 0
    num_entry_no_date = 0

    for key, resources in ehr_dict.items():
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
                    print(
                        f"Missing 'categories' attribute in resource entry: {entry.get('fullUrl', '')}"
                    )
                if "date" not in [x.lower() for x in resource.keys()]:
                    num_entry_no_date += 1
                    resource["date"] = ""
                    print(f"Missing 'date' attribute in resource entry: {entry.get('fullUrl', '')}")

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
                # print(resource)
                summary = f"{resource['clinical_status'].capitalize()} allergy intolerance:\n\tAllergy: {', '.join([allergy['display'] for allergy in resource['allergy_intolerance']])}\n\tCriticality: {resource['criticality']}"
                flattened_ehr = update_ehr_dict(flattened_ehr, summary, None)

    if num_entry_missing_categories > 0:
        print(
            f"Total number of resource entries missing categories attribute: {num_entry_missing_categories}"
        )
    if num_entry_no_date > 0:
        print(f"Total number of resource entries missing date attribute: {num_entry_no_date}")

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


def create_ehr_database():
    persist_directory = PERSISTENT_FOLDER
    ehr_data = get_ehr_data()
    flattened_ehr = flatten_ehr(ehr_data)
    documents = create_db_docs(flattened_ehr)
    print(f"Total DB documents: {len(flattened_ehr)}")
    print("Creating EHR vector db...")
    start_time = time.time()
    create_db(documents, persist_directory)
    end_time = time.time()
    total_time = end_time - start_time
    print("Done!\n")
    print(f"Total time to build db: {total_time}")
    # Only keep one decimal place
    total_time = round(total_time, 1)
    return total_time


if __name__ == "__main__":
    create_ehr_database()
