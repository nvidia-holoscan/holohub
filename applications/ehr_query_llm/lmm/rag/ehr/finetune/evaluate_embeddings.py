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

import argparse
from pathlib import Path

import pandas as pd
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.embeddings import resolve_embed_model
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from llama_index.schema import TextNode
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from tqdm import tqdm


def evaluate_comprehensive(dataset, model_id, name):
    """
    Uses InformationRetrievalEvaluator to get a comprehensive suite of evaluation metrics.
    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)
    """
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs, name=name)
    model = SentenceTransformer(model_id)
    output_path = "results/"
    Path(output_path).mkdir(exist_ok=True, parents=True)
    return evaluator(model, output_path=output_path)


def evaluate(
    dataset,
    embed_model,
    adapter=None,
    top_k=10,
    verbose=False,
):
    """
    Evaluates the given dataset using top_k hit rate
    """
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    print(f"Using embed model {embed_model}")
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)
    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(nodes, service_context=service_context, show_progress=True)
    if adapter:
        print(f"Using adapter {adapter}")
        index.service_context.embed_model = adapter
    retriever = index.as_retriever(similarity_top_k=top_k)

    eval_results = []
    for query_id, query in tqdm(queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

        eval_result = {
            "is_hit": is_hit,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results.append(eval_result)
    return eval_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation script of Embedding models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--hit-rate-only", action="store_true", help="Run testing for hit rate only"
    )
    parser.add_argument(
        "--comprehensive-only",
        action="store_true",
        help="Run full sentence_transformers embedding test suite",
    )
    parser.add_argument(
        "--model-1",
        type=str,
        help="The file path that contains the first embedding model",
        default="/workspace/volumes/models/BAAI_bge-large-en-v1.5",
    )
    parser.add_argument(
        "--model-2",
        type=str,
        help="The file path that contains the second embedding model",
        default="/workspace/volumes/models/Bge-large-EHR-finetune-7_epochs",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")

    if not args.hit_rate_only:

        print("Model 1 dot_score-MAP@100:")
        print(evaluate_comprehensive(val_dataset, args.model_1, name="model_1"))
        print("Model 2 dot_score-MAP@100:")
        print(evaluate_comprehensive(val_dataset, args.model_2, name="model_2"))
        print("See ./results for more comprehensive evaluation results")

    if not args.comprehensive_only:
        model_1 = resolve_embed_model("local:" + args.model_1)
        model_2 = resolve_embed_model("local:" + args.model_2)

        val_results_finetuned_wo_wrapper = evaluate(val_dataset, model_1)
        df_finetune_wo_wrapper = pd.DataFrame(val_results_finetuned_wo_wrapper)
        hit_rate_finetune_wo_wrapper = df_finetune_wo_wrapper["is_hit"].mean()
        print("Model 1 hit rate:")
        print(hit_rate_finetune_wo_wrapper)

        val_results_finetuned_wo_wrapper = evaluate(val_dataset, model_2)
        df_finetune_wo_wrapper = pd.DataFrame(val_results_finetuned_wo_wrapper)
        hit_rate_finetune_wo_wrapper = df_finetune_wo_wrapper["is_hit"].mean()
        print("Model 2 hit rate:")
        print(hit_rate_finetune_wo_wrapper)


if __name__ == "__main__":
    main()
