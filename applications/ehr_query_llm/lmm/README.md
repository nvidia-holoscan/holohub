# EHR Agent Framework

The EHR Agent Framework is designed to handle and interact with EHR (Electronic Health Records) and it provides a modular and extensible system for handling various types of queries through specialized agents, with robust error handling and performance optimization features.

## Table of Contents

- [Agent Framework overview](#agent-framework-overview)
- [Setup Instructions](#setup-instructions)
- [Run Instructions](#run-instructions)
- [Offline Mode](#offline-mode)

### Agent Framework overview

The `AgentFrameworkOp` orchestrates multiple specialized agents to handle different types of queries and responses, it maintains a streaming queue for responses and it handles response states through `ResponseHandler`.
It tracks conversation history using ChatHistory class and updates history based on agent responses and ASR transcripts.

Agent Lifecycle:

- Processes requests asynchronously using threads
- Controls response streaming and muting during speech

### The Base Agent Class

This is an abstract base class implementing common agent functionality:

- Uses threading.Lock for LLM and , if needed, LMM access
- Prevents concurrent requests to language models

It loads configuration from YAML files and it handles prompt templates, tokens limits, and model URLs. The Base Agent is designed to stream responses from LLM server and it supports both text and image-based prompts while enforcing maximum prompt token limits. Throughout the agent lifecycle it maintains conversation context and it creates conversation strings within token limits.

### The Selector Agent

The Selector Agent routes user queries to the appropriate specialized agent by analyzing user input to determine the appropriate agent and return the selected agent name and corrected input text.
For response parsing, it handles JSON response format. If there are parsing failures, it logs them and it returns `None` for invalid selection.

### The EHR Builder Agent

The EHR Builder Agent handles EHR database construction on demand and it tracks and reports build time performance in the process.
For response generation, it uses custom prompt templates for EHR tasks and it returns structured JSON responses.
It also verifies build capability before execution and it reports success/failure status.

### The EHR Agent

The EHR Agent handles EHR queries and data retrieval. It uses Chroma for document storage while implementing HuggingFaceBgeEmbeddings for embeddings.
For the RAG (Retrieval-Augmented Generation) pipeline, it performs MMR (Maximal Marginal Relevance) search with configurable search parameters (`k`, `lambda_mult`, `fetch_k`).
For prompt generation, it incorporates retrieved documents into prompts and it supports both standard and RAG-specific prompts.
The EHR Agent is using CUDA for embedding computation and optimizes for cosine similarity calculations.

### Common features across agents

#### Configuration Management

- YAML-based settings
- Configurable prompts and rules
- Extensible tool support

#### Response Handling

- Streaming response support
- Mutable response states
- Structured output formatting

#### Error Management

- Connection retry logic
- Comprehensive error logging
- Graceful failure handling

#### Performance Optimization

- Thread-safe operations
- Token usage optimization
- Efficient resource management

## Setup Instructions

### Speech pipeline

**_Note_**
___
[NVIDIA Riva](https://www.nvidia.com/en-us/ai-data-science/products/riva/) provides speech and translation services for user interaction with the LLM. We recommend running Riva in the bare metal host environment outside of the development container to minimize demands on container resources. During test run, it was observed that Riva could take up around 8 GB of GPU memory, while the rest of the application around 12 GB of GPU memory.

NVIDIA Riva Version Compatibility : tested with v2.13.0 / v2.14.0.
___

Please adhere to the "Data Center" configuration specifications in the [Riva Quick Start guide](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html#data-center).

To optimize Riva installation footprint:

- Locate the `config.sh` file in the riva_quickstart_vX.XX.X directory.
- Modify the `service_enabled_*` variables as follows:

```bash
service_enabled_asr=true
service_enabled_nlp=false
service_enabled_tts=true
service_enabled_nmt=false
```

### Model acquisition

> It is recommended to create a directory called `/models` on your machine to download the LLM.

Download the [quantized Mistral 7B finetuned LLM](https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF) from HugginFace.co:

```bash
wget -nc -P <your_model_dir> https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF/resolve/main/openchat-3.5-0106.Q8_0.gguf
```

Download the BGE-large finetuned embedding model from NGC:

 ```bash
 wget https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/models/bge-large-ehr-finetune
```

Execute the following command from the Holohub root directory:

```bash
./holohub build lmm
```

## Run Instructions

### Step 1: Enabling HTTPS/SSL (only required once)

⚠️ Note: This has only been tested with Chrome and Chromium

Browsers require HTTPS to be used in order to access the client's microphone.  Hence, you'll need to create a self-signed SSL certificate and key.

This key must be placed in `/applications/ehr_query_llm/lmm/ssl`

```bash
cd <holohub root>/applications/ehr_query_llm/lmm/
mkdir ssl
cd ssl
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes -subj '/CN=localhost'
```

When you first navigate your browser to a page that uses these self-signed certificates, it will issue you a warning since they don't originate from a trusted authority. Ignore this and proceed to the web app:

<img src="./imgs/ssl_warning.jpg" width="400">

ehr_query_llm will use your default speaker and microphone. To change this go to your Ubuntu sound settings and choose the correct devices:

<img src="./imgs/ubuntu_sound_setting.png" width="400">

### Step 2: Ensure Riva server is running

The Riva server must be running to use the LLM pipeline. If it is already running you can skip this step.

```bash
cd <riva install dir>
bash riva_start.sh
```

### Step 3: Launch and Run the App

#### Step 3.1

Launch the `holohub:lmm` container:

```bash
sudo ./holohub run-container lmm --add-volume <your_model_dir>
```

- Note, if the parent directory of <your_model_dir> is not `/models` you must update the [asr_llm_tts.yaml](./asr_llm_tts.yaml) and [ehr.yaml](./agents_configs/ehr.yaml) files with the complete path to your model inside the container. You will also need to update the [run_lmm.sh](./run_lmm.sh) so the correct directory is exported in the `set_transformer_cache()` function. (You can determine these paths by looking in `/workspace/volumes` inside the launched container) or you can use the following `sed` commands:

`sed -i -e 's#^model_path:.*#model_path: /workspace/volumes/<your_model_dir>#' asr_llm_tts.yaml`

`sed -i -e 's#^model_path:.*#model_path: /workspace/volumes/<your_model_dir>#' agents_configs/ehr.yaml`

`sed -i -e 's#^export TRANSFORMERS_CACHE=.*#export TRANSFORMERS_CACHE="/workspace/volumes/<your_model_dir>"#' run_lmm.sh`

#### Step 3.2

Then run the application:

```bash
./applications/ehr_query_llm/lmm/run_lmm.sh
```

This command builds ehr_query_llm/lmm, starts an LLM api server, then launches the ehr_query_llm app.
Access the web interface at `https://127.0.0.1:8080`. Llama.cpp LLM server output is redirected to `./applications/ehr_query_llm/lmm/llama_cpp.log/`.

To interact with ehr_query_llm using voice input:

- Press and hold the space bar to activate the voice recognition feature.
- Speak your query or command clearly while maintaining pressure on the space bar.
- Release the space bar when you've finished speaking to signal the end of your input.
- ehr_query_llm will then process your speech and generate a response.

⚠️ Note: When running via VNC, you must have your keyboard focus on the VNC terminal that you are using to run ehr_query_llm in order to use the push-to-talk feature.

### Stopping Instructions

To stop the main app, simply use `ctrl+c`

To stop Riva server:

```bash
bash <Riva_install_dir>riva_stop.sh
```

### ASR_To_LLM Application arguments

The `asr_llm_tts.py` can receive the following **optional** cli argument:

`--sample-rate-hz`: The number of frames per second in audio streamed from the selected microphone.

## Offline mode

To enable offline use (no internet connection required):

1. First run the complete application as-is (This ensures all relevant models are downloaded)
2. Uncomment `set_offline_flags` at [line 52 of run_lmm.sh](./run_lmm.sh)

## Troubleshooting

### Adding Agents

An Agent is an LLM (or LMM) with a task specific "persona" - such as a EHRAgent, etc., each with their own specific task. They also have a specific prompt tailored to complete that task, pre-fix prompts specific to the model used, grammar to constrain output, as well as context length.

The AgentFrameworkOp works by using a SelectorAgent to select which Agent should be called upon based on user input.

Adding a new "agent" for ehr_query_llm involves creating a new agent .py and YAML file in the `agents` directory, and in the new .py inheriting the Agent base class `agents/base_agent.py`.

When creating a new agent .py file, you will need to define:

**Agent name**: A class name which will also need to be added to the selector agent YAML, so it knows the agent is available to be called.
**process_request**: A runtime method describing the logic of how an agent should carry out its task and send a response.

For the YAML file, the fields needed are:

**name**: This is the name of the agent, as well as what is used as the ZeroMQ topic when the agent  publishes its output. So you must make sure your listener is using this as the topic.

**user_prefix, bot_prefix, bot_rule_prefix, end_token:**: These are dependent on the particular llm or lmm being used, and help to set the correct template for the model to interact with.

**agent_prompt**: This gives the agent its "persona" - how it should behave, and for what purpose. It should have as much context as possible.

**ctx_length**: Context length for the model. This determines how much output the agent is capable of generating. Smaller values lead to faster to first token time, but can be at the sacrifice of detail and verbosity.

**grammar:** This is the BNF grammar used to constrain the models output. It can be a bit tricky to write. ChatGPT is great at writing these grammars for you if you give an example JSON of what you want. Also helpful, is the Llama.cpp [BNF grammar guide](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md).

**publish:** The only important part of this field is the "ags" sub-field. This should be a list of your arg names. This is important as this is used as the list of keys to pull the relevant args from the LMM's response, and thus ensure the relevant fields are complete for a given tool use.

For a specific example, please refer to the EHR Agent YAML file below:

```
description: This tool is used to search the patient's EHR in order to answer questions about the patient, or general questions about the patient's medical history.
user_prefix: "<|im_start|>user"
bot_prefix: "<|im_start|>assistant"
bot_rule_prefix: "<|im_start|>system"
end_token: "<|im_end|>"
agent_prompt: |
  You are NVIDIA's EHR Agent,your job is to assist surgeons as they prepare for surgery.
  You are an expert when it comes to surgery and medical topics - and answer all questions to the best of your abilities.
  The patient has signed consent for you to access and discuss all of their electronic records.
  Be as concise as you can be with your answers.

  You NEVER make-up information that isn't grounded in the provided medical documents.

  If applicable, include the relevant date (use sparingly)
  The following medical documents may be helpful to answer the surgeon's question:
  {documents}
  Use your expert knowledge and the above context to answer the surgeon.
# This is the request that the LLM replies to, where '{text}' indicates where the transcribed
# text is inserted into the prompt
request: "{text}"

ctx_length: 256

grammar: |
  space ::= " "?
  string ::= "\"" ([^"\\])* "\"" space 
  root ::= "{" space "\"name\"" space ":" space "\"EHRAgent\"" space "," space "\"response\"" space ":" space string "}" space

publish:
  ags:
    - "response"

```

With a complete YAML file, an agent should be able to use any new tool effectively. The only remaining step is ensure you have a ZeroMQ listener in the primary app with a topic that is the same as the tool's name.

The `AgentFrameworkOp` is based on a ZeroMQ publish/subscribe pattern to send and receive messages from the Message Bus. It uses the `MessageReceiver` and `MessageSender` classes implemented in the `message_handling.py` Python script. The `MessageSender` creates a ZeroMQ PUB socket that binds to port 5555, accepts connections from any interface, and is used to broadcast messages and commands from the agent framework. It uses `send_json()` to send JSON-encoded messages with topics. A 0.1-second sleep on initialization prevents the "slow joiner" problem where early messages might be lost. The `MessageReceiver`` creates a ZeroMQ SUB socket connecting to port 5560 and uses receive_json() to get messages, with configurable blocking behavior.

### EHR RAG

To test new document formats for the database use [test_db.py](./rag/ehr/test_db.py)
This will start the current Vector database in `./rag/ehr/db` and allow you to test different queries via the CLI to see what documents are returned.

When changing the Vector DB, remove the previous database first:

```bash
rm -rf ./rag/ehr/db
```

### Riva - Can't find speaker to use

This usually means that some process is using the speaker you wish to use. This could be a Riva process that didn't exit correctly, or even Outlook loaded in your browser using your speakers to play notification sounds.

First see what processes are using your speakers:

```bash
pactl list sink-inputs | grep -E 'Sink Input|application.name|client|media.name|sink: '
```

Sometimes that will give you all the information you need to kill the process responsible. If not, and the process has unfamiliar name such as "speech-dispatcher-espeak-ng" then find the responsible process ID:

```bash
pgrep -l -f <grep expression here (ex: 'speech')>
```

Once you know the PID's of the responsible process, kill them :)

```bash
kill <PID>
```
