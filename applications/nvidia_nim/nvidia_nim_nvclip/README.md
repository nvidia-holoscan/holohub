# NVIDIA NV-CLIP

NV-CLIP is a multimodal embeddings model for image and text and this is a sample application that shows how to use the OpenAI SDK with NVIDIA Inference Microservice (NIM). Whether you are using a NIM from [build.nvidia.com/](https://build.nvidia.com/) or a self-hosted NIM, this sample application will work for both.

### Quick Start

1. Add API key in `nvidia_nim.yaml`
2. `./dev_container build_and_run nvidia_nim_nvclip`

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

## Run the sample application

There are a couple of options to run the sample application:

### Run using Docker

To run the sample application with Docker, you must first build a Docker image that includes the sample application and its dependencies:

```
# Build the Docker images from the root directory of Holohub
./dev_container build --docker_file applications/nvidia_nim/nvidia_nim_nvclip/Dockerfile
```

Then, run the Docker image:

```bash
./dev_container  launch
```


### Start the Application

To use the NIMs on [build.nvidia.com/](https://build.nvidia.com/), configure your API key in the `nvidia_nim.yaml` configuration file and run the sample app as follows:

note: you may also configure your api key using an environment variable.
E.g., `export API_KEY=...`

```bash
# To use NVIDIA hosted NIMs available on build.nvidia.com, export your API key first
export API_KEY=[enter your api key here]

./run launch nvidia_nim_nvclip
```

Have fun!


## Connecting with Locally Hosted NIMs

To use a locally hosted NIM, first download and start the NIM.
Then configure the `base_url` parameter in the `nvidia_nim.yaml` configuration file to point to your local NIM instance.

The following example shows a NIM running locally and serving its APIs and the `meta-llama3-8b-instruct` model from `http://0.0.0.0:8000/v1`.

```bash
nim:
  base_url: http://0.0.0.0:8000/v1/
```