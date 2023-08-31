# 🦙 Holochat-local 🦙 

Holochat-local is an AI-driven chatbot, built on top of a local Llama-2 model running on IGX Orin. The chatbot leverages vector databases to generate human-like responses and write code.

## Hardware Requirements: 👉💻
- [NVIDIA IGX Orin](https://www.nvidia.com/en-us/edge-computing/products/igx/) with:
  - RTX A6000 dGPU
  - 500 GB SSD

## Dependencies: 📦
- [NVIDIA Drivers](https://www.nvidia.com/download/index.aspx)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) >= 11.8
- Python >= 3.10
- [`build-essential`](https://packages.ubuntu.com/focal/build-essential) apt package (gcc, g++, etc.)
- [Cmake](https://apt.kitware.com/) >= 3.17

## Running the local Llama-2 model on IGX Orin:
* Follow all of the set-up instructions in the ['Llama chat tutorial'](https://gitlab-master.nvidia.com/nigeln/llama_chat_tutorial) until you reach ['Setting up a local OpenAI Server'](https://github.com/nvidia-holoscan/holohub/tree/main/tutorials/local-llama#setting-up-a-local-openai-server-%EF%B8%8F) (once you are running the Llama.cpp server example return here.)
  * *Note:* You don't need to use the base Llama-2 70B provided. You can use any GGML Llama-2 model in `/TheBloke`'s [model repo](https://huggingface.co/TheBloke). If VRAM space is an issue, use a 7B or 13B model. If you have the space for a 70B model, [Platypus2-70B-Instruct-GGML](https://huggingface.co/TheBloke/Platypus2-70B-Instruct-GGML/tree/main) is a quantized version of the #1 model on the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) that provides great results.
  * ⚠️ The current prompts have been optimized on the 'Platypus2-70B-Instruct-GGML', if you are conducting VDR, this should be the model that you use.
## Installation:

1. Clone the repository

`git clone https://gitlab-master.nvidia.com/nigeln/holochat-local.git`
`cd holochat-local`

2. Install dependencies

`pip install -r requirements.txt`

3. Download Pytorch 2.1.0 built from source
    * If you already have 'git lfs' ignore this step, you'll have the torch wheel in the repo.
      * Install 'git lfs': `curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash`
      * Finish the install: `git lfs install`
      * Download the torch wheel: `git lfs pull`
        * This Wheel was built from source on IGX Orin for CUDA 11.8 support of Python 3.10*

4. Pip install Pytorch

`pip install torch-2.1.0a0+git861ae39-cp310-cp310-linux_aarch64.whl`

5. Set Up Environment Variables

`cp .env.template .env`

Open .env in your preferred text editor and replace the placeholders with your Github user token

6. Build Vector Databases

`python build_holoscan_db.py`

7. Launch the Chatbot

`python chatbot.py`

The chatbot will now be running at http://127.0.0.1:7860/. Open this URL in a web browser to interact with the chatbot.
