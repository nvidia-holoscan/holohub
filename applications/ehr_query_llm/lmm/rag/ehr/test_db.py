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

# This script helps quickly validate and test new Vector DB's

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

Target_Collection_Name = "ehr_rag"  # must match that used in DB creation
CHROMA_DB_PATH = "/workspace/holohub/applications/ehr_query_llm/lmm/rag/ehr/db"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
CACHE_FOLDER = "/workspace/volumes/models"
SEARCH_THRESHOLD = 1
NUM_DOCS = 15
LAMBDA_MULT = 0  # 0 is diverse, 1 is least diverse


def get_ehr_database():
    model_name = MODEL_NAME
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity

    # Construct embedding model and cache to local './models' dir
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder=CACHE_FOLDER,
    )
    chroma_db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_model,
        collection_name=Target_Collection_Name,
    )
    return chroma_db


chroma_db = get_ehr_database()

print(f"DB loaded for collection: {chroma_db._collection.name}")
print(f"# docs in the collection: {chroma_db._collection.count()}")

chroma_db = chroma_db.as_retriever(
    search_type="mmr", search_kwargs={"k": NUM_DOCS, "lambda_mult": LAMBDA_MULT}
)
print("Ready for input!")


while True:
    print("\nPlease type in your question (or just return to terminate):")
    query = input()
    if not query:
        break

    docs = chroma_db.get_relevant_documents(query)
    # docs = chroma_db.similarity_search_with_score(
    #     query=query, k=NUM_DOCS, distance_metric="cos"
    # )
    print("------------------------------------------")
    for doc in docs:
        # print(f"Score: {score}")
        # doc = json.loads(doc.page_content)
        # print(f"Summary: {doc['summary']}")
        # print(f"Keywords: {doc['keyWords']}")
        # print(f"date: {doc['date']}")
        print(doc.page_content)
        print()
