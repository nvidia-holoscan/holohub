# Segment Everything App


Segment anything allows computing masks or pre-thresholded outputs for objects in natural images. An output mask is created for each query point. In this app the sam encoder and decoder parts are applied to a videostream.

## Model

This application uses a segment anything model (SAM) from https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints

## Setup

The setup requires three steps. 
1.  create a pytorch docker container, with a tensorRT version that matches the holoscan tensorRT version. 
2. Then this container creates the tensorRT engine files.
3. We create a holoscan container, that mounts the created engine files, and the holohub repository, and runs the application. 

## 1. pytorch container

We need to use a pytorch container to create the TRT engine files for SAM models
First get the docker: 
```sh
docker pull nvcr.io/nvidia/pytorch:23.03-py3
``` 

#### tensorRT version
To use the trt engine files in holoscan later, we need to have the matching tensorrt version. 
In this case we are looking for the version that matches 
- os="ubuntu2004"
- tag="8.6.1-cuda-12.0"
- arm based system (using a CAGX devkit)

following the instructions on https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/install-guide/index.html#downloading for the debian installation procedure.

The tensorRT version needs to be downloaded from https://developer.nvidia.com/nvidia-tensorrt-8x-download

In this case I download it to a folder on the m2 drive. e.g. 
```sh
mkdir /media/m2/software/tensorrt
cd /media/m2/software/tensorrt

wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-12.0_1.0-1_arm64.deb
```

#### run docker container
Now the pytorch container can be run. Mount <HOLOHUB_SOURCE_DIR>, which contains the holohub repo, and mount the software folder, which contains the downloaded trt .deb installation file.

```sh
docker run --rm -it -v $PWD:/workspace -v /media/m2/software:/workspace/software nvcr.io/nvidia/pytorch:23.04-py3 /bin/bash
```
##### installation of tensorrt 
follow the commandline steps below to install tensorRT and additional packages.
```sh
cd /workspace/software/tensorrt
dpkg -i nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-12.0_1.0-1_arm64.deb

cp /var/nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-12.0/nv-tensorrt-local-7148CA18-keyring.gpg /usr/share/keyrings/

apt-get update

apt-get install tensorrt
apt-get install python3-libnvinfer-dev
python3 -m pip install numpy onnx onnx-graphsurgeon
```

Check the installed version 
```sh
dpkg-query -W tensorrt
```

## 2. create trt engine files

In the same docker container we can now proceed with building the trt engine files.
First, pull the sam_trt_light repo, which contains a script to create the engine files.
```
cd /workspace
git clone https://github.com/maximilianofir/sam_trt_light.git
```

Install the package sam_trt_light, and some additional requirements.
```
cd /workspace/sam_trt_light
pip install -e .
pip install onnxruntime onnx_graphsurgeon colored polygraphy --upgrade
```

Download the models, e,g, sam_vit_b:
```
mkdir downloads
cd downloads
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
I downloaded them to a downloads folder, e.g. "/workspace/sam_trt_light/downloads"

Create folders for the onnx and engine files 

```
cd ..
mkdir onnx engine
```

Run the trt_inference.py script and save the onnx and engine files to the created folders

```
python scripts/trt_inference.py --checkpoint=/workspace/sam_trt_light/downloads/sam_vit_b_01ec64.pth --input-image=images/apples.jpg --mode point --visualize --output-image=output.png --model-type vit_b --onnx-dir /workspace/sam_trt_light/onnx --engine-dir /workspace/sam_trt_light/engine
```

The <MODEL_DIR> is this case is ```/workspace/sam_trt_light/engine```.
You can use [netron.app](https://netron.app/) to visualize the onnx files. Make sure the datatypes are fp32 for all inputs and outputs.


## 3. launch application container

### Requirements
This application uses a v4l2 compatible device as input.  Please plug in your input device (e.g., webcam) and ensure that `applications/segment_everything/segment_everything.yaml` is set to the corresponding device.  The default input device is set to `/dev/video0`.

Build a holohub container in a new terminal to launch the segment_anything app.
```
cd <HOLOHUB_SOURCE_DIR>
./dev_container build --docker_file applications/segment_everything/Dockerfile --img holohub:sam2.1
```
launch the dev container and mount the folder that contains holohub and sam_trt_light
```
./dev_container launch --img holohub:sam2.1 --add-volume <MODEL_DIR>
```
Adjust the segment_one_thing.yaml file to point to the trt engine files, for example for the encoder inference block, adjust: 
```yaml
  model_path_map:
    "encoder": "/workspace/engine/encoder.engine"
```
```sh
cd /workspace/holohub
python applications/segment_everything/segment_one_thing.py
```

