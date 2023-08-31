import torch
import requests
import json
import urllib.parse
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from build_holoscan_db import CHROMA_DB_PATH

system_prompt = """You are NVIDIA-GPT, an expert at all things NVIDIA who knows the Holoscan user guide, as well as examples from Holohub and the api from the SDK. 
You are an assistant who answers questions step-by-step and always provides your reasoning so you have the correct result.
Answer the questions based on the provided context, make sure to use only real Holoscan API in code examples, and augment with your general knowledge where appropriate.
Under no circumstances will you make up Holoscan API functions or functionality that does not exist! Do not conflate Holoscan Python API with Holoscan C++ API. You ALWAYS end your response with '</s>'
"""


class LLM:
    def __init__(self) -> None:
        self.retriever =  self._get_retriever()

    def answer_question(self, chat_history):
        question = chat_history[-1][0]
        docs = self.retriever.get_relevant_documents(question)

        llama_prompt = to_llama_prompt(chat_history[1:-1], question, docs)
        response = self._stream_ai_response(llama_prompt, chat_history)

        for chunk in response:
            yield chunk


    def _stream_ai_response(self, llama_prompt, chat_history):
            request_data = {
                "prompt": llama_prompt,
                "temperature": 0,
                "stop": [
                    "</s>"
                ],
                "n_keep": -1,
                "stream": True
            }
            resData = requests.request("POST", urllib.parse.urljoin("http://127.0.0.1:8080", "/completion"), data=json.dumps(request_data), stream=True)
            
            chat_history[-1][1] = ''
            for line in resData.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    next_token = json.loads(decoded_line[6:]).get("content")
                    chat_history[-1][1] += next_token
                    yield chat_history


    def _get_retriever(self):
        model_name = "BAAI/bge-large-en"
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

        embedding_model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
        retriever = chroma_db.as_retriever()
        retriever.search_kwargs['distance_metric'] = 'cos'
        retriever.search_kwargs['fetch_k'] = 7
        retriever.search_kwargs['maximal_marginal_relevance'] = True
        retriever.search_kwargs['k'] = 7

        return retriever


def to_llama_prompt(history, question, docs):
    """ An attempt to mirror Alpaca-style prompting as closely as possible: https://github.com/arielnlee/Platypus/blob/main/templates/alpaca.json
    """
    user_prefix = '### Input:'
    bot_prefix = '### Response:'
    bot_rule_prefix = '### Instruction:'


    prompt = f"Below is a chat between a user '{user_prefix}', and you, the AI assistant '{bot_prefix}'. You follow the given rule '{bot_rule_prefix}' no matter what."

    for msg_pair in history:
        if msg_pair[0]:
            prompt += f'\n\n{user_prefix}\n{msg_pair[0]}'
        if msg_pair[1]:
            prompt += f'\n\n{bot_prefix}\n{msg_pair[1]}</s>'

    docs = "\n\n".join(list(map(lambda lc_doc: lc_doc.page_content, docs)))

    prompt += f'\n\n{bot_rule_prefix}\n{system_prompt}'
    prompt += f'\n\n{user_prefix}\n{docs}\n\n\nUsing the previous conversation history, the provided NVIDIA Holoscan SDK documentation, AND your own expert knowledge, answer the following question (include markdown code snippets for coding questions and do not acknowledge that I provided documentation):\n{question}'
    prompt += f'\n\n{bot_prefix}\n'

    return prompt
