# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import urllib.parse
from time import sleep

import requests
from build_holoscan_db import CHROMA_DB_PATH
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

# Most Llama-2 models are trained with a context length of 4096 tokens
# 1 token = ~4 characters, so 3300 * 4 provides plenty of room.
MAX_TOKENS = 3300 * 4
# Empirically found to be the cutoff of a specific questions vs. generic comments about previous answer
# This ensures no documents are returned for comments such as "Rewrite that code in one block"
SEARCH_THRESHOLD = 0.35
NUM_HOLOSCAN_DOCS = 7
LLAMA_SERVER = "http://127.0.0.1:8080"
SERVER_TIMEOUT = 60  # Timeout in seconds to connect to llama.cpp

system_prompt = (
    "You are NVIDIA-GPT, an expert at all things NVIDIA who knows "
    "the Holoscan user guide, as well as examples from Holohub and the api from the SDK. "
    "You are an assistant who answers questions step-by-step and always provides your "
    "reasoning so you have the correct result. Answer the questions based on the provided "
    "context and augment with your general knowledge where appropriate. Reformat the provided "
    "code examples as necessary since they were retrieved with a web scrape. "
    "Under no circumstances will you make up Holoscan API functions or functionality that does not "
    "exist! Do not conflate Holoscan Python API with Holoscan C++ API. You ALWAYS end your response "
    "with '</s>'. Below is NVIDIA Holoscan SDK documentation to assist you in answering questions:"
)


class LLM:
    def __init__(self) -> None:
        _wait_for_server()
        self.db = self._get_database()
        self.prev_docs = []

    def answer_question(self, chat_history):
        question = chat_history[-1][0]
        docs = self.db.similarity_search_with_score(query=question, k=NUM_HOLOSCAN_DOCS)
        # Filter out poor matches from vector db
        docs = list(
            map(lambda lc_doc: lc_doc[0], filter(lambda lc_doc: lc_doc[1] < SEARCH_THRESHOLD, docs))
        )
        # If filter removes documents, add previous documents
        if len(docs) < NUM_HOLOSCAN_DOCS:
            docs += self.prev_docs[
                : NUM_HOLOSCAN_DOCS - len(docs)
            ]  # Get first docs (highest similarity score)
        self.prev_docs = docs  # Save document list

        # Create a prompt to send to the llm (Remove greeting and question)
        llama_prompt = _to_llama_prompt(chat_history[1:-1], question, docs)
        response = self._stream_ai_response(llama_prompt, chat_history)

        for chunk in response:
            yield chunk

    def _stream_ai_response(self, llama_prompt, chat_history):
        # Llama-specific request data
        request_data = {
            "prompt": llama_prompt,
            "temperature": 0,
            "stop": ["</s>"],
            "n_keep": -1,
            "stream": True,
        }
        resData = requests.request(
            "POST",
            urllib.parse.urljoin(LLAMA_SERVER, "/completion"),
            data=json.dumps(request_data),
            stream=True,
        )

        chat_history[-1][1] = ""
        for line in resData.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                next_token = json.loads(decoded_line[6:]).get("content")
                chat_history[-1][1] += next_token
                yield chat_history

    def _get_database(self):
        model_name = "BAAI/bge-large-en"
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity

        # Construct embedding model and cache to local './models' dir
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder="./models",
        )
        chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)

        return chroma_db


def _to_llama_prompt(history, question, docs):
    """
    Function that takes the chat history, current question, and the documents
    from the vector db and creates a single string to prompt the Llama model with
    """

    # Phind v2's prompt prefixes (Note these are dependent on the model used)
    user_prefix = "### User Message:"
    bot_prefix = "### Assistant:"
    bot_rule_prefix = "### System Prompt:"

    # Explain the context of the information being provided
    opening_prompt = (
        f"Below is a chat between a user '{user_prefix}', and you, "
        "the AI assistant '{bot_prefix}'. You follow the given rule "
        "'{bot_rule_prefix}' no matter what."
    )

    # Combine all the vector db docs into a single string
    docs = "\n\n".join(list(map(lambda lc_doc: lc_doc.page_content, docs)))
    # Add the system prompt with the vector db docs
    opening_prompt += f"\n\n{bot_rule_prefix}\n{system_prompt}\n\n{docs}"
    # Define the final portion of the prompt
    ending_prompt = (
        f"\n\n{user_prefix}\nUsing the previous conversation history, "
        "the provided NVIDIA Holoscan SDK documentation, AND your own expert knowledge, answer "
        "the following question (include markdown code snippets for coding questions and do not "
        f"acknowledge that documentation was provided to you):\n{question}"
    )

    # Loop over the chat history and convert it to a single string
    msg_hist = ""
    for msg_pair in history:
        if msg_pair[0]:
            msg_hist += f"\n\n{user_prefix}\n{msg_pair[0]}"
        if msg_pair[1]:
            msg_hist += f"\n\n{bot_prefix}\n{msg_pair[1]}</s>"

    len_prompt = len(msg_hist) + len(opening_prompt) + len(ending_prompt)

    # Truncate previous conversation history if MAX_TOKENS exceeded
    if len_prompt > MAX_TOKENS:
        excess_tokens = len_prompt - MAX_TOKENS
        msg_hist = msg_hist[excess_tokens:]
        last_msg_idx = msg_hist.find("\n\n" + user_prefix)
        bot_idx = msg_hist.find("\n\n" + bot_prefix)
        # Truncate to the last user or bot message, which ever allows for a
        # longer chat history
        if bot_idx < last_msg_idx:
            last_msg_idx = bot_idx
        msg_hist = msg_hist[last_msg_idx:]

    # Create the final prompt
    prompt = opening_prompt + msg_hist + ending_prompt + f"\n\n{bot_prefix}\n"
    return prompt


def _wait_for_server():
    """
    Method that attempts to connect to the llama.cpp server
    for up to SERVER_TIMEOUT until throwing an exception
    """
    attempts = 0
    while attempts < SERVER_TIMEOUT / 5:
        try:
            response = requests.get(LLAMA_SERVER)
            # Check for a successful response status code (e.g., 200 OK)
            if response.status_code == 200:
                print("Connected to Llama.cpp server")
                return
        except requests.ConnectionError:
            sleep(5)
        attempts += 1
    raise requests.ConnectionError("Unable to connected to Llama.cpp server")
