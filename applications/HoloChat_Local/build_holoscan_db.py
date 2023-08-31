import os
from dotenv import load_dotenv
import sklearn
import torch
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader # for loading the pdf


from utils import get_source_chunks, clone_repository

load_dotenv()

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

repos = ['holoscan-sdk', 'holohub']
docs = ['docs/Holoscan_SDK_User_Guide_0.5.1.pdf']

file_types = ['.md','.py','.cpp','.yaml']

CHROMA_DB_PATH = f'./embeddings/{os.path.basename("holoscan")}'


def main():
    content_lists = {file_type: [] for file_type in file_types}
    total_files = 0
    for repo in repos:
        clone_repository(repo, GITHUB_TOKEN)
        for file_type in file_types:
            for root, dirs, files in os.walk(f'./{repo}'):
                for file in files:
                    if file.lower().endswith(file_type):
                        total_files += 1
                        print(f'Processing file: {file}')
                        print(f'Total files: {total_files}')
                        with open(os.path.join(root, file), 'r', errors='ignore') as f:
                            content = f.read()
                            content_lists[file_type].append(Document(page_content=content, metadata={"source": os.path.join(root, file)}))

    content_lists['.pdf'] =  []
    for doc in docs:
        loader = PyPDFLoader(doc)
        pages = loader.load_and_split()
        print('doc length: ', len(pages))
        for page in pages:
            total_files += 1
            content_lists['.pdf'].append(Document(page_content=page.page_content, metadata={"userguide": doc}))

    ext_to_language = {'.py': 'python', '.cpp': 'cpp', '.md': 'markdown', '.yaml': None, '.pdf': None}
    source_chunks = []
    content_len = 0

    for file_ext, content in content_lists.items():
        content_len += len(content)
        source_chunks += get_source_chunks(content, ext_to_language[file_ext])


    model_name = "BAAI/bge-large-en"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


    print(f"Length of content: {content_len}")
    print(f"Total number of files to process: {total_files}")
    print(f"Number of source chunks: {len(source_chunks)}")
    print(f'Building Holoscan Embeddings Chroma DB at {CHROMA_DB_PATH}...')
    print('Building Chroma DB (This may take a few minutes)...')
    chroma_db = Chroma.from_documents(source_chunks, embedding_model, persist_directory=CHROMA_DB_PATH)
    print('Done!')
    chroma_db.persist()


if __name__ == "__main__":
    main()