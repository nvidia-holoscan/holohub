# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

from .base_agent import Agent


class EHRAgent(Agent):
    def __init__(self, settings_path, response_handler):
        super().__init__(settings_path, response_handler)
        # Pass the specific key for this agent's settings
        self.load_settings(settings_path)
        self.rag_prompt = self.agent_settings.get("rag_prompt").strip()
        self.rag_grammar = self.agent_settings.get("rag_grammar")
        self.db = self._get_ehr_database()

    def _get_ehr_database(self):
        model_name = self.agent_settings.get("embedding_model")
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity

        # Construct embedding model and cache to local '/workspace/volumes/models' dir
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder="/workspace/volumes/models",
        )
        chroma_db = Chroma(
            persist_directory=self.agent_settings.get("db_path"),
            embedding_function=embedding_model,
            collection_name="ehr_rag",
        )

        chroma_db = chroma_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.agent_settings["k"],
                "lambda_mult": self.agent_settings["lambda_mult"],
                "fetch_k": self.agent_settings["fetch_k"],
            },
        )
        return chroma_db

    def generate_prompt(self, text, agent_prompt, chat_history):
        """
        Generate a prompt for the LLM based on the given text and agent configuration,
        including the retrieved EHR documents.
        """
        # Get the relevant EHR docs using mmr
        lc_documents = self.db.get_relevant_documents(text)

        documents = "\n\n".join(list(map(lambda doc: doc.page_content, lc_documents)))
        system_prompt = f"{self.bot_rule_prefix}\n{agent_prompt.format(documents=documents)}\n{self.end_token}\n"
        user_prompt = f"{self.user_prefix}\n{text}\n{self.end_token}\n"
        # Calculate the token usage of the system and user prompts
        _ = self.calculate_token_usage(system_prompt + user_prompt)
        # Create the chat history component without exceeding the maximum prompt tokens
        # chat_prompt = self.create_conversation_str(chat_history, token_usage)
        prompt = system_prompt + user_prompt
        prompt += f"{self.bot_prefix}\n"

        return prompt, lc_documents

    def generate_rag_prompt(self, text):
        """
        Generate a prompt for the LLM based on the given text and agent configuration,
        including the retrieved EHR documents.
        """
        # Get the relevant EHR docs using mmr
        prompt = f"{self.bot_rule_prefix}\n{self.rag_prompt}\n{self.end_token}\n"
        prompt += f"{self.user_prefix}\n{text}{self.end_token}\n"
        prompt += f"{self.bot_prefix}\n"
        return prompt

    def get_rag_query(self, text):
        """
        Processes a request related to the patient
        """
        rag_prompt = self.generate_rag_prompt(text)
        # Send the prompt to the LLM and get the response
        self._logger.debug(f"EHR RAG query Agent Prompt:\n{rag_prompt}")

        response = self.stream_response(rag_prompt, grammar=self.rag_grammar, display_output=False)

        # Escape newlines so json.loads() can be used
        response = response.replace("\n", "\\n")
        query = json.loads(response)["Possible EHR"]
        return query

    def process_request(self, text, chat_history):
        """
        Processes a request related to the patient
        """
        # Uncomment to create a RAG query that represents synthetic EHR data that would
        # answer the user's question

        self._logger.debug(f"ASR Text Received:\n{text}\n")
        # self._logger.debug(f"Rag query generated:\n{query}\n")
        # Generate prompt
        prompt, documents = self.generate_prompt(text, text, chat_history)
        self._logger.debug("Documents retrieved:")
        for doc in documents:
            self._logger.debug(f"\n{doc.page_content}\n")
        # Send the prompt to the LLM and get the response
        self._logger.debug(f"EHR RAG Agent Prompt:\n{prompt}")

        response = self.stream_response(prompt, grammar=self.grammar)

        # Escape newlines so json.loads() can be used
        response = response.replace("\n", "\\n")

        # Return the complete response
        return response, documents
