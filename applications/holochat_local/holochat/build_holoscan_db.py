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

import os
import re

import chromadb
import sklearn
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader  # for loading the pdf
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from utils import clone_repository, get_source_chunks

repos = ["holoscan-sdk", "holohub"]
docs = ["/holochat/docs/Holoscan_SDK_User_Guide_v0.6.0.pdf"]

file_types = [".md", ".py", ".cpp", ".yaml"]

CHROMA_DB_PATH = f'/holochat/embeddings/{os.path.basename("holoscan")}'


def main():
    content_lists = {file_type: [] for file_type in file_types}
    total_files = 0

    # Loop over each repo and create a Document for each file found
    for repo in repos:
        clone_repository(repo, "")
        for file_type in file_types:
            for root, dirs, files in os.walk(f"/tmp/{repo}"):
                for file in files:
                    if file.lower().endswith(file_type):
                        total_files += 1
                        print(f"Processing file: {file}")
                        print(f"Total files: {total_files}")
                        with open(os.path.join(root, file), "r", errors="ignore") as f:
                            content = f.read()
                            content_lists[file_type].append(
                                Document(
                                    page_content=content,
                                    metadata={"source": os.path.join(root, file)},
                                )
                            )

    # Loop over the user guide and create a Document for each page
    content_lists[".pdf"] = []
    for doc in docs:
        loader = PyPDFLoader(doc)
        pages = loader.load_and_split()
        print("doc length: ", len(pages))
        for page in pages:
            page_content = page.page_content
            # Remove line numbers for code
            page_content = re.sub(
                r"^\d+(?!\.)",
                lambda match: " " * len(match.group(0)),
                page_content,
                flags=re.MULTILINE,
            )
            # Remove unnecessary text
            page_content = re.sub(
                r".*(Holoscan SDK User Guide, Release|Chapter|(continued from previous page|(continues on next page))).*\n?",
                "",
                page_content,
            )
            content_lists[".pdf"].append(
                Document(page_content=page_content, metadata={"userguide": doc})
            )

    # Dictionary used to map file type to language
    ext_to_language = {
        ".py": "python",
        ".cpp": "cpp",
        ".md": "markdown",
        ".yaml": None,
        ".pdf": None,
    }
    source_chunks = []
    content_len = 0

    for file_ext, content in content_lists.items():
        content_len += len(content)
        source_chunks += get_source_chunks(content, ext_to_language[file_ext])

    model_name = "BAAI/bge-large-en"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity

    # Create local embedding model cached at ./models
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder="./models",
    )

    print(f"Length of content: {content_len}")
    print(f"Total number of files to process: {total_files}")
    print(f"Number of source chunks: {len(source_chunks)}")
    print(f"Building Holoscan Embeddings Chroma DB at {CHROMA_DB_PATH}...")
    print("Building Chroma DB (This may take a few minutes)...")
    chroma_db = Chroma.from_documents(
        source_chunks, embedding_model, persist_directory=CHROMA_DB_PATH
    )
    print("Done!")
    chroma_db.persist()


if __name__ == "__main__":
    main()
