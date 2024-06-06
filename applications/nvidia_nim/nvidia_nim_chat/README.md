# Chat with NVIDIA NIM

This is a sample application that shows how to use the OpenAI SDK with NVIDIA Inference Microservice (NIM). Whether you are using a NIM from [build.nvidia.com/](https://build.nvidia.com/) or a self-hosted NIM, this sample application will work for both.

### Quick Start

1. Add API key in `nvidia_nim.yaml`
2. `./dev_container build_and_run nvidia_nim_chat`

## Configuring the sample application

Use the `nvidia_nim.yaml` configuration file to configure the sample application:

### Connection Information

```
nim:
  base_url: https://integrate.api.nvidia.com/v1
  api_key:

```

`base_url`: The URL of your NIM instance. Defaults to NVIDIA hosted NIMs.
`api_key`: Your API key to access NVIDIA hosted NIMs.

### Model Information

The `models` section in the YAML file is configured with multiple NVIDIA hosted models by default. This allows you to switch between different models easily within the application by sending the prompt `/m` to the application.

Model parameters may be added or adjusted in the `models` section as well per model.

## Run the sample application

There are a couple of options to run the sample application:

### Run using Docker

To run the sample application with Docker, you must first build a Docker image that includes the sample application and its dependencies:

```
# Build the Docker images from the root directory of Holohub
./dev_container build --docker_file applications/nvidia_nim/Dockerfile
```

Then, run the Docker image:

```bash
./dev_container  launch
```

Continue to the [Start the Application](#start-the-application) section once inside the Docker container.

### Run the Application without Docker

Install all dependencies from the `requirements.txt` file:

```bash
# optionally create a virtual environment and activate it
python3 -m venv .venv
source .venv/bin/activate

# install the required packages
pip install -r applications/nvidia_nim/chat/requirements.txt
```

### Start the Application

To use the NIMs on [build.nvidia.com/](https://build.nvidia.com/), configure your API key in the `nvidia_nim.yaml` configuration file and run the sample app as follows:

note: you may also configure your api key using an environment variable.
E.g., `export API_KEY=...`

```bash
# To use NVIDIA hosted NIMs available on build.nvidia.com, export your API key first
export API_KEY=[enter your api key here]

./run launch nvidia_nim_chat
```

Have fun!


## Connecting with Locally Hosted NIMs

To use a locally hosted NIM, first download and start the NIM.
Then configure the `base_url` parameter in the `nvidia_nim.yaml` configuration file to point to your local NIM instance.

The following example shows a NIM running locally and serving its APIs and the `meta-llama3-8b-instruct` model from `http://0.0.0.0:8000/v1`.

```bash
nim:
  base_url: http://0.0.0.0:8000/v1/

models:
  llama3-8b-instruct:
    model: meta-llama3-8b-instruct # name of the model serving by the NIM
    # add/update/remove the following key/value pairs to configure the parameters for the model
    top_p: 1
    n: 1
    max_tokens: 1024
    frequency_penalty: 1.0
```