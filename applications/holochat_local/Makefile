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

# Target that runs all commands in order to run HoloChat
.PHONY: run_holochat
run_holochat: build_llamaCpp build_db download_llama start_holochat

# Creates container used for HoloChat and compiles Llama.cpp
.PHONY: build_llamaCpp
build_llamaCpp:
	docker build --ulimit memlock=-1 --ulimit stack=67108864 -t holochat .

# Creates the vector database used by HoloChat
.PHONY: build_db
build_db:
	mkdir -p holochat/embeddings
	mkdir -p holochat/models
	wget -nc -P ./holochat/docs/ https://developer.download.nvidia.com/assets/Clara/Holoscan_SDK_User_Guide_v0.6.0.pdf
	[ -f ./holochat/embeddings/holoscan/chroma.sqlite3 ] || \
	docker run --rm \
		--gpus all \
		--ipc=host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-v ./holochat:/holochat \
		-w /holochat \
		holochat \
		python3 build_holoscan_db.py \

# Downloads the Llama-2 model used by HoloChat
.PHONY: download_llama
download_llama:
	mkdir -p holochat/docs
	wget -nc -P ./holochat/models https://api.ngc.nvidia.com/v2/models/nvidia/clara-holoscan/phind-codellama-34b-v2-q5_k_m/versions/2.0/zip
	unzip ./holochat/models/zip -d ./holochat/models/

# Runs HoloChat inside the pytorch container
.PHONY: start_holochat
start_holochat:
	docker run --rm -it \
	-p 7860:7860 \
	-p 8080:8080 \
	--gpus all \
	--ipc=host \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	-v ./holochat:/holochat \
	-w /holochat \
	holochat \
	bash -c "/workspace/llama.cpp/build/bin/server \
			 -m /holochat/models/phind-codellama-34b-v2.Q5_K_M.gguf \
			 --host 0.0.0.0 \
			 -ngl 1000 \
			-c 4096 \
			--alias llama_2 \
			& python3 -u chatbot.py"