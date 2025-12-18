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
import shutil
import tempfile
from time import sleep
from types import SimpleNamespace

import requests
import tiktoken
import yaml
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

# Format logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


class LLM:
    def __init__(self, is_local=False, is_mcp=False) -> None:
        load_dotenv()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.yaml")
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)
        self.config = SimpleNamespace(**yaml_config)
        self._logger = logging.getLogger(__name__)
        self._app_dir = current_dir
        self._cache_root = self._get_writable_cache_root()
        self._configure_hf_cache_env(self._cache_root)
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
        elif not is_mcp:
            self._logger.info("Using NVIDIA NIM API")
            assert (
                api_key != "N/A"
            ), "NVIDIA_API_KEY environment variable not set, please set it in .env file"

        if not is_mcp:
            # Create OpenAI client
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)

        # Calculate the base prompt length for the system and user prompts
        self._base_prompt_length = self.calculate_token_usage(
            self.config.system_prompt + self.config.user_prompt
        )
        self.prev_docs = []

    def _coerce_to_text(self, content) -> str:
        """
        Coerce Gradio Chatbot message content into a plain string.

        Gradio can pass content as:
        - str
        - list/tuple of strings (or richer multimodal parts)
        - dicts for richer message parts
        """
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore")

        # Multimodal content often comes as a list of parts
        if isinstance(content, (list, tuple)):
            parts = []
            for item in content:
                if item is None:
                    continue
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    # Common keys used by multimodal/text parts
                    for k in ("text", "content", "value", "message"):
                        if k in item and item[k] is not None:
                            parts.append(self._coerce_to_text(item[k]))
                            break
                    else:
                        parts.append(str(item))
                    continue
                parts.append(str(item))
            return "\n".join([p for p in parts if p])

        if isinstance(content, dict):
            for k in ("text", "content", "value", "message"):
                if k in content and content[k] is not None:
                    return self._coerce_to_text(content[k])
            return str(content)

        return str(content)

    def answer_question(self, chat_history):
        """
        Takes Gradio 6.x Chatbot "messages" history and streams an updated history.

        Expected format:
        - [{"role": "user"|"assistant", "content": "..."}, ...]
          (also supports Gradio ChatMessage-like objects via duck-typing)
        """
        if not chat_history:
            return

        # Normalize ChatMessage-like objects to dicts and validate shape.
        chat_history = self._normalize_gradio_messages(chat_history)

        question = self._extract_last_user_message(chat_history)
        if not question:
            return

        docs = self._retrieve_docs(question)

        pairs = self._messages_to_pairs(chat_history)
        # Drop a leading assistant-only message (welcome banner) if present
        if pairs and pairs[0][0] is None:
            pairs = pairs[1:]
        # Drop the last pair which contains the current question (no assistant answer yet)
        history_pairs = pairs[:-1] if pairs else []

        messages = self.get_chat_messages(history_pairs, question, docs)
        yield from self._stream_ai_response_messages(messages, chat_history)

    def _retrieve_docs(self, question: str):
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
            docs += self.prev_docs[: self.config.num_docs - len(docs)]
        self.prev_docs = docs  # Save document list
        return docs

    def _extract_last_user_message(self, chat_history):
        for msg in reversed(chat_history):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return self._coerce_to_text(msg.get("content"))
        return ""

    def _normalize_gradio_messages(self, chat_history):
        """
        Normalize a Gradio Chatbot history into a list of {"role","content"} dicts.

        This supports dicts and ChatMessage-like objects with .role/.content.
        """
        normalized = []
        for msg in chat_history:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
            else:
                # Duck-type Gradio ChatMessage
                role = getattr(msg, "role", None)
                content = getattr(msg, "content", None)

            if role not in ("user", "assistant", "system"):
                raise TypeError(
                    "chat_history must be Gradio Chatbot 'messages' format: "
                    "each message needs role in {'user','assistant','system'} and content."
                )
            normalized.append({"role": role, "content": self._coerce_to_text(content)})
        return normalized

    def _messages_to_pairs(self, chat_history):
        """
        Convert [{"role","content"}, ...] into [[user, assistant], ...] pairs.

        This preserves an initial assistant-only message as [None, content].
        """
        pairs = []
        current_user = None
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                # Flush any unfinished user message
                if current_user is not None:
                    pairs.append([current_user, None])
                current_user = content
            elif role == "assistant":
                if current_user is None:
                    pairs.append([None, content])
                else:
                    pairs.append([current_user, content])
                    current_user = None
        if current_user is not None:
            pairs.append([current_user, None])
        return pairs

    def _stream_ai_response_messages(self, messages, chat_history):
        """
        Streams the LLM response and yields updated Gradio-style messages history.
        """
        completion = self.llm_client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=1024, stream=True
        )

        # Ensure there's an assistant message to stream into
        if (
            not chat_history
            or not isinstance(chat_history[-1], dict)
            or chat_history[-1].get("role") != "assistant"
        ):
            chat_history.append({"role": "assistant", "content": ""})
        else:
            chat_history[-1]["content"] = chat_history[-1].get("content") or ""

        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                next_token = chunk.choices[0].delta.content
                if next_token:
                    chat_history[-1]["content"] += next_token
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
        # Always prefer a writable cache location for HF/SentenceTransformers assets.
        # Some deployments mount /workspace read-only, which breaks caching under the repo.
        model_cache_path = self._ensure_writable_dir(
            os.path.join(self._cache_root, "models"),
            fallback=os.path.join(tempfile.gettempdir(), "holochat", "models"),
        )
        # Construct embedding model and cache
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=model_cache_path,
        )

        chroma_db_path = os.path.join(self._app_dir, self.config.chroma_db_dir)
        chroma_db_path = self._ensure_writable_chroma_dir(chroma_db_path)
        chroma_db = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)

        return chroma_db

    def _get_writable_cache_root(self) -> str:
        """
        Returns a writable cache root directory for runtime downloads/copies.

        Priority:
        - HOLOCHAT_CACHE_DIR env var
        - XDG_CACHE_HOME/holochat
        - ~/.cache/holochat
        - /tmp/holochat
        """
        candidates = []
        env_dir = os.environ.get("HOLOCHAT_CACHE_DIR")
        if env_dir:
            candidates.append(env_dir)
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            candidates.append(os.path.join(xdg, "holochat"))
        candidates.append(os.path.join(os.path.expanduser("~"), ".cache", "holochat"))
        candidates.append(os.path.join(tempfile.gettempdir(), "holochat"))

        for d in candidates:
            try:
                os.makedirs(d, exist_ok=True)
                test_path = os.path.join(d, ".write_test")
                with open(test_path, "w") as f:
                    f.write("ok")
                os.remove(test_path)
                return d
            except Exception:
                continue

        # As a last resort, use a temp dir (should always be writable)
        return tempfile.mkdtemp(prefix="holochat-")

    def _configure_hf_cache_env(self, cache_root: str) -> None:
        """
        Point HuggingFace/SentenceTransformers caches at a writable directory.
        This prevents permission errors when the repo is mounted read-only.
        """
        hf_home = os.path.join(cache_root, "hf")
        os.makedirs(hf_home, exist_ok=True)
        # Force override: containers/environments may pre-set these to read-only paths.
        os.environ["HF_HOME"] = hf_home
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "transformers")
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(hf_home, "sentence-transformers")

    def _ensure_writable_dir(self, preferred: str, fallback: str) -> str:
        """
        Ensure we have a writable directory. If preferred isn't writable, use fallback.
        """
        for d in (preferred, fallback):
            try:
                os.makedirs(d, exist_ok=True)
                # Don't rely on os.access() alone; in some container setups it can be misleading.
                test_path = os.path.join(d, ".write_test")
                with open(test_path, "w") as f:
                    f.write("ok")
                os.remove(test_path)
                return d
            except Exception:
                continue
        # If both fail, use cache root (should be writable by construction)
        os.makedirs(self._cache_root, exist_ok=True)
        return self._cache_root

    def _ensure_writable_chroma_dir(self, chroma_dir: str) -> str:
        """
        Chroma will write to its persist_directory (sqlite/journal/temp files).
        If the configured directory is not writable (e.g. mounted read-only),
        copy the existing DB to a writable location under the cache root.
        """
        # If it's writable, use it as-is.
        try:
            os.makedirs(chroma_dir, exist_ok=True)
            if os.access(chroma_dir, os.W_OK):
                return chroma_dir
            else:
                raise PermissionError(f"Chroma directory '{chroma_dir}' is not writable.")
        except Exception as e:
            self._logger.debug(
                f"Failed to prepare Chroma directory '{chroma_dir}'. "
                "Using cache root for fallback Chroma directory. "
                f"Error: {e}"
            )

        # Fallback: copy to cache root
        fallback_dir = os.path.join(self._cache_root, "chroma", "holoscan")
        os.makedirs(os.path.dirname(fallback_dir), exist_ok=True)

        # Copy only if source exists; if it doesn't, we'll just use an empty writable dir.
        if os.path.isdir(chroma_dir):
            try:
                shutil.copytree(chroma_dir, fallback_dir, dirs_exist_ok=True)
            except Exception as e:
                self._logger.warning(
                    f"Failed to copy Chroma DB from {chroma_dir} to {fallback_dir}: {e}"
                )
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir

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
