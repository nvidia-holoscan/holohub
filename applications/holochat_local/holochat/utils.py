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

import base64
import fnmatch
import time

import git
import requests
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm


def clone_repository(repo, token):
    """
    Used to clone nvidia-holoscan repos
    """
    print(f"Cloning repository: {repo}")
    time.sleep(1)
    try:
        git.Repo.clone_from(
            f"https://github.com/nvidia-holoscan/{repo}.git",
            f"/tmp/{repo}",
            env={"GIT_ASKPASS": "", "GIT_USERNAME": "", "GIT_PASSWORD": token},
        )
        print(f"Cloned repository: {repo}")
    except Exception as e:
        print(f"Failed to clone repository: {repo}. Error: {e}")


def clone_general_repository(repo, token):
    """
    Used to clone general repos
    """
    print(f"Cloning repository: {repo}")
    time.sleep(1)
    try:
        git.Repo.clone_from(
            f"https://github.com/{repo}.git",
            f"/tmp/{repo}",
            env={"GIT_ASKPASS": "", "GIT_USERNAME": "", "GIT_PASSWORD": token},
        )
        print(f"Cloned repository: {repo}")
    except Exception as e:
        print(f"Failed to clone repository: {repo}. Error: {e}")


def process_git_repo(repo, token):
    url = f"https://api.github.com/repos/nvidia-holoscan/{repo}/git/trees/main?recursive=1"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        return content["tree"]
    else:
        raise ValueError(f"Fetching repository {repo} unsuccessful!: {response.status_code}")


def get_files(files, type):
    contents = []
    for file in files:
        if file["type"] == "blob" and fnmatch.fnmatch(file["path"], "*" + type):
            response = requests.get(file["url"])
            time.sleep(1)
            if response.status_code == 200:
                content = response.json()["content"]
                decoded_content = base64.b64decode(content).decode("utf-8")
                print("Fetching Content from ", file["path"])
                contents.append(
                    Document(page_content=decoded_content, metadata={"source": file["path"]})
                )
            else:
                print(f"Error downloading file {file['path']}: {response.status_code}")
    return contents


def get_source_chunks(all_contents, file_type=None, chunk_size=1500, chunk_overlap=150):
    """
    Method that splits Documents into chunks for storage. If the language is supported,
    it is split according to the syntax of that language (Ex: not splitting python
    functions in the middle)
    """
    if file_type in ["python", "cpp", "markdown"]:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=file_type, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    print(f"Turning {file_type} text into chunks ...")
    source_chunks = []
    for source in tqdm(all_contents, desc="Processing files..."):
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))
    return source_chunks
