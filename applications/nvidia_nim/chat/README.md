# Chat with NVIDIA NIM 

This is a sample application that shows how to use the OpenAI SDK with NVIDIA Inference Microservice (NIM). Whether you are using a NIM from [build.nvidia.com/](https://build.nvidia.com/) or a self-hosted NIM, this sample application will work for both.


## Prerequisites

Install all dependencies from the `requirements.txt` file:

```bash
# optionally create a virtual environment and activate it
python3 -m venv .venv 
source.venv/bin/activate

# install the required packages
pip install -r requirements.
```

## Run the sample application

```bash
cd applications/nvidia_nim/chat
python3 app.py
```


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

Have fun!
