# NVIDIA NV-CLIP

NV-CLIP is a multimodal embeddings model for image and text, and this is a sample application that shows how to use the OpenAI SDK with NVIDIA Inference Microservice (NIM). Whether you are using a NIM from [build.nvidia.com/](https://build.nvidia.com/) or [a self-hosted NIM](https://docs.nvidia.com/nim/nvclip/latest/getting-started.html#option-2-from-ngc), this sample application will work for both.

## Quick Start

Get your [API Key](https://docs.nvidia.com/nim/nvclip/latest/getting-started.html#generate-an-api-key) and start the sample application.

1. Enter your API key in `nvidia_nim.yaml`
2. `./dev_container build_and_run nvidia_nim_nvclip`

## Advanced

### Configuring the sample application

Use the `nvidia_nim.yaml` configuration file to configure the sample application:

### NVIDIA-Hosted NV-CLIP NIM

By default, the application is configured to use NVIDIA-hosted NV-CLIP NIM.

```
nim:
 base_url: https://integrate.api.nvidia.com/v1
 api_key:

```

`base_url`: The URL of your NIM instance. Defaults to NVIDIA-hosted NIMs.
`api_key`: Your API key to access NVIDIA-hosted NIMs.


Note: you may also configure your API key using an environment variable.
E.g., `export API_KEY=...`

```bash
# To use NVIDIA hosted NIMs available on build.nvidia.com, export your API key first
export API_KEY=[enter your API key here]
```


### Self-Hosted NIMs

To use a self-hosted NIM, refer to the [NV-CLIP](https://docs.nvidia.com/nim/nvclip/latest/getting-started.html) NIM documentation to configure and start the NIM.

Then, comment out the NVIDIA-hosted section and uncomment the self-hosted configuration section in the `nvidia_nim.yaml` file.

```bash
nim:
  base_url: http://0.0.0.0:8000/v1/
  encoding_format: float
  api_key: NA
  model: nvidia/nvclip-vit-h-14
```


### Build The Application

To run the sample application, you must first build a Docker image that includes the sample application and its dependencies:

```
# Build the Docker images from the root directory of Holohub
./dev_container build --docker_file applications/nvidia_nim/nvidia_nim_nvclip/Dockerfile
```

Then, run the Docker image:

```bash
./dev_container launch
```


### Run the Application

To use the NIMs on [build.nvidia.com/](https://build.nvidia.com/), configure your API key in the `nvidia_nim.yaml` configuration file and run the sample app as follows:

```bash
./run launch nvidia_nim_nvclip
```

## Using the Application

Once the application is ready, it will prompt you to input URLs to the images you want to perform inference.

```bash
Enter a URL to an image: https://domain.to/my/image-cat.jpg
Downloading image...

Enter a URL to another image or hit ENTER to continue: https://domain.to/my/image-rabbit.jpg
Downloading image...

Enter a URL to another image or hit ENTER to continue: https://domain.to/my/image-dog.jpg
Downloading image...

```

If there are no more images that you want to use, hit ENTER to continue and then enter a prompt:

```bash
Enter a URL to another image or hit ENTER to continue:

Enter a prompt: Which image contains a rabbit?
```

The application will connect to the NIM to generate an answer and then calculate the cosine similarity between the images and the prompt:

```bash
⠧ Generating...
Prompt: Which image contains a rabbit?
Output:
Image 1: 3.0%
Image 2: 52.0%
Image 3: 46.0%
```