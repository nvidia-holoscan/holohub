# Chat with NVIDIA NIM

This is a sample application that shows how to use the OpenAI SDK with NVIDIA Inference Microservice (NIM). Whether you are using a NIM from [build.nvidia.com/](https://build.nvidia.com/) or a self-hosted NIM, this sample application will work for both.

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

The `models` section int the YAML file is configured with multiple NVIDIA hosted models by default. This allows you to switch between different models easily within the application by sending `/m` to the application.

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
source.venv/bin/activate

# install the required packages
pushd applications/nvidia_nim/chat
pip install -r requirements.txt
popd
```

### Start the Application

To use the NIMs on [build.nvidia.com/](https://build.nvidia.com/), configure your API key in the `nvidia_nim.yaml` configuration file and run the sample app as follows:

note: you may also configure your api key using an environment variable.
E.g., `export OPENAI_API_KEY=...`

```bash
# To use NVIDIA hosted NIMs available on build.nvidia.com, export your API key first
export OPENAI_API_KEY=[enter your api key here]

cd applications/nvidia_nim/chat
python3 app.py
```

Have fun!
