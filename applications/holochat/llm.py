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
import os
from time import sleep
from types import SimpleNamespace

import openai
import requests
import tiktoken
import yaml
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

# Format logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


class LLM:
    def __init__(self, is_local=False, is_mcp=False) -> None:
        load_dotenv()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.yaml")
        yaml_config = yaml.safe_load(open(config_path))
        self.config = SimpleNamespace(**yaml_config)
        self._logger = logging.getLogger(__name__)
        # Load the vector db
        self.db = self._get_database()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.model = self.config.nim_model
        base_url = self.config.nim_url
        api_key = os.environ.get("NVIDIA_API_KEY", "N/A")
        if is_local:
            # Ensure the Llama.cpp server is running
            self._wait_for_server()
            self._logger.info("Using local Llama.cpp server")
            self.model = self.config.local_model
            base_url = self.config.local_llm_url
        elif is_local:
            self._logger.info("Using NVIDIA NIM API")
            assert (
                api_key != "N/A"
            ), "NVIDIA_API_KEY environment variable not set, please set it in .env file"

        if not is_mcp:
            # Create OpenAI client
            self.llm_client = openai.OpenAI(base_url=base_url, api_key=api_key)

        # Calculate the base prompt length for the system and user prompts
        self._base_prompt_length = self.calculate_token_usage(
            self.config.system_prompt + self.config.user_prompt
        )
        self.prev_docs = []

    def answer_question(self, chat_history):
        """
        Function that takes the chat history and returns the response from the LLM
        """
        question = chat_history[-1][0]
        # Get the most similar documents from the vector db
        docs = self.db.similarity_search_with_score(query=question, k=self.config.num_docs)
        # Filter out poor matches from vector db
        docs = list(
            map(
                lambda lc_doc: lc_doc[0],
                filter(lambda lc_doc: lc_doc[1] < self.config.search_threshold, docs),
            )
        )
        # If filter removes documents, add previous documents
        if len(docs) < self.config.num_docs:
            docs += self.prev_docs[
                : self.config.num_docs - len(docs)
            ]  # Get first docs (highest similarity score)
        self.prev_docs = docs  # Save document list

        # Create a ChatML messages to send to the llm (Remove greeting and question)
        messages = self.get_chat_messages(chat_history[1:-1], question, docs)
        response = self._stream_ai_response(messages, chat_history)
        # Stream the response
        for chunk in response:
            yield chunk

    def _stream_ai_response(self, messages, chat_history):
        """
        Function that streams the response from the LLM using OpenAI's API
        """
        completion = self.llm_client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=1024, stream=True
        )

        chat_history[-1][1] = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                next_token = chunk.choices[0].delta.content
                if next_token:
                    chat_history[-1][1] += next_token
                yield chat_history

    def calculate_token_usage(self, text):
        """
        Calculates the number of tokens used by the given text
        """
        return len(self.tokenizer.encode(text))

    def _get_database(self):
        """
        Function that retrieves the Holoscan vector DB
        """
        self._logger.info("Retrieving Holoscan embeddings vector DB...")
        model_name = "BAAI/bge-large-en"
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_cache_path = os.path.join(current_dir, self.config.model_cache_dir)
        # Construct embedding model and cache to local './models' dir
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=model_cache_path,
        )
        chroma_db_path = os.path.join(current_dir, self.config.chroma_db_dir)
        chroma_db = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)

        return chroma_db

    def get_chat_messages(self, history, question, docs):
        """
        Function that takes the chat history, current question, and the documents
        from the vector db and creates list of ChatML messages to send to the LLM
        """
        question_length = self.calculate_token_usage(question)
        prompt_length = self._base_prompt_length + question_length

        # Add relevant documents to the prompt until max_prompt_tokens - 600 is reached
        used_docs = []
        for doc in docs:
            doc_length = self.calculate_token_usage(doc.page_content)
            # Leave room for the chat history
            if prompt_length + doc_length < self.config.max_prompt_tokens - 600:
                prompt_length += doc_length
                used_docs.append(doc)
            else:
                break

        # Combine all the vector db docs into a single string
        doc_str = "\n\n".join(list(map(lambda lc_doc: lc_doc.page_content, used_docs)))

        # Create the system prompt
        system_prompt = f"{self.config.system_prompt}\n{doc_str}"
        messages = [{"role": "system", "content": system_prompt}]

        # Add the chat history to the prompt until max_prompt_tokens is reached
        for msg_pair in history[::-1]:
            if msg_pair[1]:
                msg_length = self.calculate_token_usage(msg_pair[1])
                if prompt_length + msg_length < self.config.max_prompt_tokens:
                    prompt_length += msg_length
                    messages.insert(1, {"role": "assistant", "content": msg_pair[1]})
                else:
                    break
            if msg_pair[0]:
                msg_length = self.calculate_token_usage(msg_pair[0])
                if prompt_length + msg_length < self.config.max_prompt_tokens:
                    prompt_length += msg_length
                    messages.insert(1, {"role": "user", "content": msg_pair[0]})
                else:
                    break

        # Remove the last user prompt if it is the last message (need alternating ChatML roles)
        if len(messages) > 1:
            if messages[-1]["role"] == "user":
                messages.pop()

        # Add the current user prompt to the prompt
        complete_user_prompt = f"{self.config.user_prompt}\n{question}"
        messages.append({"role": "user", "content": complete_user_prompt})

        self._logger.info(f"Num tokens in prompt: {prompt_length}")
        return messages

    def _wait_for_server(self):
        """
        Method that attempts to connect to the llama.cpp server
        for up to server_timeout until throwing an exception
        """
        attempts = 0
        while attempts < self.config.server_timeout / 5:
            try:
                response = requests.get(self.config.local_llm_url.replace("/v1", ""))
                # Check for a successful response status code (e.g., 200 OK)
                if response.status_code == 200:
                    self._logger.info("Connected to Llama.cpp server")
                    return
            except requests.ConnectionError:
                sleep(5)
            attempts += 1
        raise requests.ConnectionError("Unable to connected to Llama.cpp server")
