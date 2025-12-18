# HoloChat

HoloChat is an AI-driven chatbot, built on top of a **locally hosted Code-Llama model** *OR* a remote **NIM API for Llama-3-70b**, which acts as developer's copilot in Holoscan development. The LLM leverages a vector database comprised of the Holoscan SDK repository and user guide, enabling HoloChat to answer general questions about Holoscan, as well act as a Holoscan SDK coding assistant.

<p align="center">
  <kbd style="border: 2px solid black;">
    <img src="holochat_demo.gif" alt="HoloChat Demo" />
  </kbd>
</p>

## Quick Start Guide

**Choose one of the following modes to run HoloChat:**

- **NVIDIA NIM API (cloud)**:  Uses the [NVIDIA NIM API](https://build.nvidia.com/explore/discover) with [Llama-3.3-70b-instruct](https://build.nvidia.com/meta/llama-3_3-70b-instruct) model.
  > [!IMPORTANT]
  > For cloud mode, you must create a `.env` file in the `holochat` directory with your NVIDIA API key. If you don't have an API key, you can create one on [build.nvidia.com](https://build.nvidia.com/settings/api-keys) using your NVIDIA developer account.
  >
  > ```bash
  > echo "NVIDIA_API_KEY=<api_key_here>" > ./applications/holochat/.env
  > ```

  ```bash
  ./holohub run holochat
  ```

- **Local LLM (standalone)**: Runs the LLM locally (GPU required), using [Phind-CodeLlama-34B-v2](https://huggingface.co/Phind/Phind-CodeLlama-34B-v2) model and [Llama.cpp](https://github.com/ggml-org/llama.cpp.git) for inference.

  ```bash
  ./holohub run holochat standalone
  ```

  > [!NOTE]
  > First local run will likely take ~45 minutes (container + model downloads), and expect ~28 GB of model data under `holochat/models`, so use a disk with plenty of free space.

- **MCP server (mcp)**: Runs as a [Model Context Protocol](https://modelcontextprotocol.io/) server that provides Holoscan documentation and code context to upstream LLMs like Claude

  ```bash
  ./holohub run holochat mcp
  ```

  See [MCP_MODE.md](MCP_MODE.md) for more details on using MCP mode.

## Hardware Requirements (Running Locally)

- **GPU**: NVIDIA dGPU w/ >= 28 GB VRAM
- **Memory**: \>= 28 GB of available disk memory, needed to download [fine-tuned Code Llama 34B](https://huggingface.co/TheBloke/Phind-CodeLlama-34B-v2-GGUF) and [BGE-Large](https://huggingface.co/BAAI/bge-large-en) embedding model

*Tested using [NVIDIA IGX Orin](https://www.nvidia.com/en-us/edge-computing/products/igx/) w/ RTX A6000 and Dell Precision 5820 Workstation w/ RTX A6000*

## Running Instructions

If connecting to your machine via SSH, be sure to forward the appropriate ports:

- For chatbot UI: 7860
- For local LLM: 8080
- For MCP server: 8090

```bash
ssh <user_name>@<IP address> -L 7860:localhost:7860 -L 8080:localhost:8080 -L 8090:localhost:8090
```

### Running w/ Local LLM 💻

**To build and start the app:**

```bash
./holohub run holochat standalone
```

Once the LLM is loaded on the GPU and the Gradio app is running, HoloChat should be available at <http://127.0.0.1:7860/>.

### Running w/ NIM API ☁️

To use the NIM API you must create a .env file at:

```bash
./applications/holochat/.env
```

This is where you should place your [NVIDIA API](https://build.nvidia.com/explore/discover) key.

```bash
NVIDIA_API_KEY=<api_key_here>
```

**To build and run the app:**

```bash
./holohub run holochat
```

Once the Gradio app is running, HoloChat should be available at <http://127.0.0.1:7860/>.

## Usage Notes: 🗒️

### Intended use: 🎯
  >
  >HoloChat is developed to accelerate and assist Holoscan developers’ learning and development. HoloChat serves as an intuitive chat interface, enabling users to pose natural language queries related to the Holoscan SDK. Whether seeking general information about the SDK or specific coding insights, users can obtain immediate responses thanks to the underlying Large Language Model (LLM) and vector database.
  >
  >HoloChat is given access to the Holoscan SDK repository, the HoloHub repository, and the Holoscan SDK user guide. This essentially allows users to engage in natural language conversations with these documents, gaining instant access to the information they need, thus sparing them the task of sifting through vast amounts of documentation themselves.

### Known Limitations: ⚠️🚧

Before diving into how to make the most of HoloChat, it's crucial to understand and acknowledge its known limitations. These limitations can guide you in adopting the best practices below, which will help you navigate and mitigate these issues effectively.

- **Hallucinations:** Occasionally, HoloChat may provide responses that are not entirely accurate. It's advisable to approach answers with a healthy degree of skepticism.
- **Memory Loss:** LLM's limited attention window may lead to the loss of previous conversation history. To mitigate this, consider restarting the application to clear the chat history when necessary.
- **Limited Support for Stack Traces**: HoloChat's knowledge is based on the Holoscan repository and the user guide, which lack large collections of stack trace data. Consequently, HoloChat may face challenges when assisting with stack traces.

### Best Practices: ✅👍

While users should be aware of the above limitations, following the recommended tips will drastically minimize these possible shortcomings. In general, the more detailed and precise a question is, the better the results will be. Some best practices when asking questions are:

- **Be Verbose**: If you want to create an application, specify which operators should be used if possible (HolovizOp, V4L2VideoCaptureOp, InferenceOp, etc.).
- **Be Specific**: The less open-ended a question is the less likely the model will hallucinate.
- **Specify Programming Language**: If asking for code, include the desired language (Python or C++).
- **Provide Code Snippets:** If debugging errors include as much relevant information as possible. Copy and paste the code snippet that produces the error, the abbreviated stack trace, and describe any changes that may have introduced the error.

In order to demonstrate how to get the most out of HoloChat two example questions are posed below. These examples illustrate how a user can refine their questions and as a result, improve the responses they receive:

---
**Worst👎:**
“Create an app that predicts the labels associated with a video”

**Better👌:**
“Create a Python app that takes video input and sends it through a model for inference.”

**Best🙌:**
“Create a Python Holoscan application that receives streaming video input, and passes that video input into a pytorch classification model for inference. Then, collect the model’s predicted class and use Holoviz to display the class label on each video frame.”

---
**Worst👎:**
“What os can I use?”

**Better👌:**
“What operating system can I use with Holoscan?”

**Best🙌:**
“Can I use MacOS with the Holoscan SDK?”

## Appendix

### Meta Terms of Use

By using the Code-Llama model, you are agreeing to the terms and conditions of the [license](https://ai.meta.com/llama/license/), [acceptable use policy](https://ai.meta.com/llama/use-policy/) and Meta’s [privacy policy](https://www.facebook.com/privacy/policy/).

### Implementation Details
  >
  >HoloChat operates by taking user input and comparing it to the text stored within the vector database, which is comprised of Holoscan SDK information. The most relevant text segments from SDK code and the user guide are then appended to the user's query. This approach allows the chosen LLM to answer questions about the Holoscan SDK, without being explicitly trained on SDK data.
  >
  >However, there is a drawback to this method - the most relevant documentation is not always found within the vector database. Since the user's question serves as the search query, queries that are too simplistic or abbreviated may fail to extract the most relevant documents from the vector database. As a consequence, the LLM will then lack the necessary context, leading to poor and potentially inaccurate responses. This occurs because LLMs strive to provide the most probable response to a question, and without adequate context, they hallucinate to fill in these knowledge gaps.
