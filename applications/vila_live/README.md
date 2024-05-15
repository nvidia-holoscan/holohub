# Holoscan VILA Live

This applications demonstrates how to run [VILA 1.5](https://github.com/Efficient-Large-Model/VILA) models on live video feed with the possibility
of changing the prompt in real time.

![Holoscan VILA Live](./screenshot.png)
Note: This demo currently uses [Llama-3-VILA1.5-8b-AWQ](https://huggingface.co/Efficient-Large-Model/Llama-3-VILA1.5-8b-AWQ), but any of the following AWQ-quantized models from the VILA 1.5 familty should work as long as the file names are changed in the [Dockerfile](./Dockerfile) and [run_vila_live.sh](./run_vila_live.sh):
- [VILA1.5-3b-AWQ](https://huggingface.co/Efficient-Large-Model/VILA1.5-3b-AWQ)
- [VILA1.5-3b-s2-AWQ](https://huggingface.co/Efficient-Large-Model/VILA1.5-3b-s2-AWQ)
- [Llama-3-VILA1.5-8b-AWQ](https://huggingface.co/Efficient-Large-Model/Llama-3-VILA1.5-8b-AWQ)
- [VILA1.5-13b-AWQ](https://huggingface.co/Efficient-Large-Model/VILA1.5-13b-AWQ)
- [VILA1.5-40b-AWQ](https://huggingface.co/Efficient-Large-Model/VILA1.5-40b-AWQ)
## Setup Instructions
The app defaults to using the video device at `/dev/video0`

To debug if this is the correct device download `v4l2-ctl`:
```bash
sudo apt-get install v4l-utils
```
To check for your devices run:
```bash
v4l2-ctl --list-devices
```
Then map your desired video device to the source device in [vila_live.yaml](vila_live.yaml)

## Build and Run Instructions
From the Holohub main directory run the following command:
```bash
./dev_container build_and_run vila_live
```
Note: The first build will take **~1.5 hours** if you're on ARM64. This is largely due to building [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) since ARM64 wheels are not distributed for this platform.

Once the main LMM-based app is running, you will see a link for the app at `http://127.0.0.1:8050`


## Models used
The docker file downloads the [AWQ Llama-3-VILA1.5-8B VLM](https://huggingface.co/Efficient-Large-Model/Llama-3-VILA1.5-8b-AWQ).
