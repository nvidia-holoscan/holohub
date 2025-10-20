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

import json
import random
import signal
import sys
import time
import urllib
import uuid
from threading import Thread

import requests
import yaml
import zmq
from llama_index.finetuning import EmbeddingQAFinetuneDataset, SentenceTransformersFinetuneEngine
from llama_index.schema import MetadataMode, TextNode
from tqdm import tqdm

from operators.ehr_query_llm.fhir.ehr_query import FHIRQuery
from operators.ehr_query_llm.message_handling import MessageReceiver

# Need to run:
# /opt/nvidia/holoscan/llama.cpp/build/bin/server -m  /workspace/volumes/models/mixtral-slimorca-8x7b.Q5_K_M.gguf --host 0.0.0.0 -ngl 1000 -c 3000

# FHIR global vars
topic_request = "ehr-request"
topic_response = "ehr-response"
busy = False
resources_to_retrieve = ["Observation", "FamilyMemberHistory", "Condition"]
ehr_data = None

# Vector db global vars
embedding_model_cache = "/workspace/volumes/models"
persist_directory = "/workspace/holohub/applications/ehr_query_llm/lmm/rag/ehr"

# LLM Global Vars
inference_server_url = "http://0.0.0.0:8080/v1"


def get_ehr_data():
    global ehr_data
    receiver = MessageReceiver(topic_response, "tcp://localhost:5601")
    while True:
        response = receiver.receive_json()
        if len(response) > 0:
            for key in response:
                print(f"Documents retrieved: {len(response[key])}")
                ehr_data = response
            return
        time.sleep(1)


def send_request():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5600")
    # Give ZeroMQ time to bind TODO determine why this is needed
    time.sleep(1)
    print("Sending request to FHIR server...")
    query = FHIRQuery(
        start_date="1950-01-01", end_date="2023-01-01", resources_to_retrieve=resources_to_retrieve
    )
    socket.send_multipart([topic_request.encode("utf-8"), query.to_json().encode("utf-8")])


def stop(signum=None, frame=None):
    print("Stopping...")
    sys.exit()


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
    for key, resources in ehr_dict.items():
        for entry in resources:
            resource = entry["resource"]

            # Parse Condition
            if resource["resourceType"] == "Condition":
                summary = f"{resource['clinical_status']} condition: {resource['condition']}"
                flattened_ehr = update_ehr_dict(flattened_ehr, summary, resource["recorded_date"])
            # Parse Observation
            elif resource["resourceType"] == "Observation":
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

    return flattened_ehr


def stream_llm_response(prompt, grammar, temperature=0):
    # Llama-specific request data
    request_data = {
        "prompt": prompt,
        "temperature": 0.4,
        "stop": ["</s>", "Response:"],
        "n_keep": -1,
        "grammar": grammar,
        "stream": True,
        "n_predict": 512,
    }
    resData = requests.request(
        "POST",
        urllib.parse.urljoin(inference_server_url, "/completion"),
        data=json.dumps(request_data),
        stream=True,
    )

    response = ""
    for line in resData.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            next_token = json.loads(decoded_line[6:]).get("content")
            response += next_token

    print(f"LLM Rsponse: {response}\n\n")
    return json.loads(response.replace("\n", "\\n"))["Question"]


def generate_qa_embedding_pairs(
    nodes: [TextNode],
    qa_generate_prompt_tmpl: str,
    grammar: str,
) -> EmbeddingQAFinetuneDataset:
    """Generate examples given a set of nodes."""
    node_dict = {node.node_id: node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes}

    queries = {}
    relevant_docs = {}
    for node_id, text in tqdm(node_dict.items()):
        try:
            query = qa_generate_prompt_tmpl + f"{text.replace(' Text: ', '')}\n## Response:\n"

            print(f"Prompt to LLM:\n{query}")
            response = stream_llm_response(query, grammar)

            questions = [response]

            for question in questions:
                question_id = str(uuid.uuid4())
                queries[question_id] = question
                relevant_docs[question_id] = [node_id]
        except Exception as e:
            print(f"Error generating question for node {node_id}: {e}")

    # construct dataset
    return EmbeddingQAFinetuneDataset(
        queries=queries, corpus=node_dict, relevant_docs=relevant_docs
    )


def create_db_docs(flattened_ehr):
    """
    Identical to the create_db_docs function in create_ehr_db.py. However, this
    function uses Llama_Index TextNodes instead of LangChain Documents.
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

        new_doc = TextNode(text=summary, id_=i)
        documents.append(new_doc)
        i += 1

    return documents


def main():
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
    print(f"Total DB documents: {len(flattened_ehr)}")

    random.shuffle(documents)

    train_nodes = documents[: int(len(documents) * 0.9)]
    val_nodes = documents[int(len(documents) * 0.9) :]

    with open("finetune_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        grammar = config["grammar"]
        prompt = config["prompt"]
        epochs = config["epochs"]

    train_dataset = generate_qa_embedding_pairs(
        train_nodes, qa_generate_prompt_tmpl=prompt, grammar=grammar
    )
    train_dataset.save_json("train_dataset.json")

    val_dataset = generate_qa_embedding_pairs(
        val_nodes, qa_generate_prompt_tmpl=prompt, grammar=grammar
    )
    val_dataset.save_json("val_dataset.json")

    train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
    val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")

    finetune_engine = SentenceTransformersFinetuneEngine(
        train_dataset,  # Dataset to be trained on
        model_id="BAAI/bge-large-en-v1.5",  # HuggingFace reference to base embeddings model
        model_output_path=f"/workspace/volumes/models/Bge-large-EHR-finetune-{epochs}_epochs",  # Output directory for fine-tuned embeddings model
        val_dataset=val_dataset,  # Dataset to validate on
        epochs=epochs,  # Number of Epochs to train for
    )

    finetune_engine.finetune()

    finetune_engine.get_finetuned_model()


if __name__ == "__main__":
    main()
