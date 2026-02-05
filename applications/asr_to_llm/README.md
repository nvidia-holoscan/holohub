# Real-time Riva ASR to local-LLM

This application streams microphone input to [NVIDIA Riva](https://www.nvidia.com/en-us/ai-data-science/products/riva/) Automatic Speech Recognition (ASR), which once the user specifies they are done speaking, passes the transcribed text to an LLM running locally that then summarizes this text.

While this workflow in principle could be used for a number of domains, the app is currently configured to be healthcare specific. The current LLM prompt is created for radiology interpretation, but this can be easily changed in the YAML file to tailor the LLM's output to a wide array of potential use cases.

## Example output
Example output can be found at [example_output.md](./example_output.md)  
* Description of output fields:  
**Final Transcription:** Riva's transcription of the provided mic input  
**LLM Summary:** The LLM's output summarization


## YAML Configuration

The directions for the LLM are determined by the `stt_to_nlp.yaml` file. As you see from our example, the directions for the LLM are made via natural language, and can result in very different applications.

With the current YAML configuration, the resulting prompt to the LLM is:

```
<|system|>
You are a veteran radiologist, who can answer any medical related question.

<|user|>
Transcript from Radiologist:
{**transcribed text inserted here**}
Request(s):
Make a summary of the transcript (and correct any transcription errors in CAPS).
Create a Patient Summary with no medical jargon.
Create a full radiological report write-up.
Give likely ICD-10 Codes.
Suggest follow-up steps.

<|assistant|>
```

## Setup Instructions

### Install Riva:
First, you must follow the [Riva local deployment quickstart guide](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html#local-deployment-using-quick-start-scripts). For x86 and ARM64 devices with dGPU follow the "Data Center" instructions, for ARM64 devices with iGPU follow the "Embedded" instructions.

* Note: to minimize the Riva install size you can change the `config.sh` file in the `riva_quickstart_vX.XX.X` directory such that it specifies to only install the ASR models (Riva has more features but only ASR is needed for this app). To do this, find the `sevice_enabled_*` variables and set them as shown below:
```bash
service_enabled_asr=true
service_enabled_nlp=false
service_enabled_tts=false
service_enabled_nmt=false
```

 ⚠️ Note: If you are using ARM64 w/ iGPU or an x86 platform the quick-start scripts *should* work as intended. However, if you are using ARM64 w/ dGPU, you will need to make the following modifications to the Riva Quick-start scripts:

In `riva_init.sh` make the following changes to ensure the ARM64 version of [NGC-CLI](https://docs.ngc.nvidia.com/cli/index.html) is downloaded and your dGPU is used to run the container:
```diff
# download required models
-if [[ $riva_target_gpu_family == "tegra" ]]; then
-    docker run -it -d --rm -v $riva_model_loc:/data \
+if [[ $riva_target_gpu_family == "non-tegra" ]]; then
+            docker run -it -d --rm --gpus '"'$gpus_to_use'"' -v $riva_model_loc:/data \
                -e "NGC_CLI_API_KEY=$NGC_CLI_API_KEY" \
```
Then in `riva_start.sh` make the changes below to ensure your Riva server has access to your sound devices:
```diff
docker rm $riva_daemon_speech &> /dev/null
-if [[ $riva_target_gpu_family == "tegra" ]]; then
+if [[ $riva_target_gpu_family == "non-tegra" ]]; then
    docker_run_args="-p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8888:8888 --device /dev/bus/usb --device /dev/snd"
```

### Setup Instructions:
Download the [quantized Mistral 7B LLM](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF) from HugginFace.co:
```bash
wget -nc -P <your_model_dir> https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q8_0.gguf
```

## Run instructions

Build and launch the `holohub:asr_to_llm` container:
```bash
./holohub run-container asr_to_llm --add-volume <your_model_dir>
```
Run the application and use the `--list-devices` arg to determine which microphone to use:
```bash
python <streaming_asr_to_llm_dir>/asr_to_llm.py --list-devices
```
Then run the application with the `--input-device` arg to specify the correct microphone:
```bash
python <streaming_asr_to_llm_dir>/asr_to_llm.py --input-device <device-index>
```

Once `asr_to_llm.py` is running, you will see output from ALSA for loading the selected audio device and also from llama_cpp for loading the LLM onto GPU memory. Once this is complete it will immediately begin printing out the transcribed text. **To signal that the audio you wish to transcribe is complete, enter `x` on the keyboard**. This will terminate the ASR and microphone instance, and feed the complete transcribed text into the LLM for summarization.

## Stopping Instructions
Note: The `python asr_to_llm.py` command will complete on its own once the LLM is finished summarizing the transcription
* Stopping Riva services:
```bash
bash <Riva_install_dir>riva_stop.sh
```

## ASR_To_LLM Application arguments
The `asr_to_llm.py` can receive several cli arguments:

`--input-device`: The index of the input audio device to use.
`--list-devices`: List input audio device indices.
`--sample-rate-hz`: The number of frames per second in audio streamed from the selected microphone.
`--file-streaming-chunk`: A maximum number of frames in a audio chunk sent to server.

## Implementation Details
This application adapted the [speech_to_text_llm](../speech_to_text_llm/) Holohub application to transcribe audio in real-time using Riva ASR, as well as ensure that the complete app runs 100% locally.

The LLM currently used in this application is [Mistral-7B-OpenOrca-GGUF](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF), which is a quantized [Mistal 7B model](https://mistral.ai/news/announcing-mistral-7b/) that is finetuned on the [OpenOrca dataset](https://huggingface.co/datasets/Open-Orca/OpenOrca). However, any model in the GGUF file format will work as long as it can fit within your device's VRAM constraints.

The inference engine used to run the LLM is [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), which is a Python binding for [llama.cpp](https://github.com/ggerganov/llama.cpp). The reason for this is that the underlying llama.cpp library is hardware agnostic, dependency free, and it runs **quantized** LLMs with very high throughput.

The RivaStreamingOp is a Holoscan SDK adaptation of the [transcribe_mic.py](https://github.com/nvidia-riva/python-clients/blob/main/scripts/asr/transcribe_mic.py) script that is part of the [Riva python-clients](https://github.com/nvidia-riva/python-clients/tree/main) repository.


## Dev Container

To start the the Dev Container, run the following command from the root directory of Holohub:

```bash
./holohub vscode asr_to_llm
```

### VS Code Launch Profiles

There are two launch profiles configured for this application:

1. **(debugpy) asr_to_llm/python**: Launch asr_to_llm using a launch profile that enables debugging of Python code.
2. **(pythoncpp) asr_to_llm/python**: Launch asr_to_llm using a launch profile that enables debugging of Python and C++ code.