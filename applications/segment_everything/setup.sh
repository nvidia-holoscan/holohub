#!/bin/bash

# Get the current working directory
CWD=$(pwd)

# create the data/segment_everything directory if it does not exist
if [ ! -d "data/segment_everything" ]; then
    mkdir -p data/segment_everything
fi
DATA_DIR="$CWD/data/segment_everything"

# save the current working directory
pushd $CWD
# Create the directory based on the current working directory
mkdir -p "$DATA_DIR/software/tensorrt"
cd "$DATA_DIR/software/tensorrt"

# Detect the platform
ARCH=$(uname -m)

# Set the URL and filename based on the architecture
if [ "$ARCH" == "aarch64" ]; then
    FILENAME="nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-12.0_1.0-1_arm64.deb"
else
    FILENAME="nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-12.0_1.0-1_amd64.deb"
fi

URL="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/$FILENAME"

# Check if the file has been downloaded already
if [ ! -f "$FILENAME" ]; then
    wget $URL
else
    echo "$FILENAME already exists, skipping download."
fi

popd

# sam_trt_light
# download the repository if it does not exist
cd $DATA_DIR
if [ ! -d "sam_trt_light" ]; then
    git clone https://github.com/maximilianofir/sam_trt_light.git
fi

cd sam_trt_light
# create folders
if [ ! -d "downloads" ]; then
    mkdir downloads  && \
    cd downloads && \
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth 
    cd ..
fi 
if [ ! -d "onnx" ]; then
    mkdir onnx 
fi
if [ ! -d "engine" ]; then
    mkdir engine 
fi

cd $CWD
# Run the Docker container with the commands to be executed inside it
docker run --rm -it -v "$PWD:/workspace" nvcr.io/nvidia/pytorch:23.04-py3 /bin/bash -c "
cd /workspace/data/segment_everything/software/tensorrt && \
dpkg -i $FILENAME && \
KEYRING_PATH=/var/nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-12.0/nv-tensorrt-local-9A1EDFBA-keyring.gpg && \
cp \$KEYRING_PATH /usr/share/keyrings/
dpkg -i $FILENAME && \
apt-get update && \
apt-get install -y tensorrt python3-libnvinfer-dev && \
python3 -m pip install numpy onnx onnx-graphsurgeon && \
dpkg-query -W tensorrt && \
cd /workspace/data/segment_everything/sam_trt_light && \
pip install -e . && \
pip install onnxruntime onnx_graphsurgeon colored polygraphy --upgrade && \
cd data/segment_everything/sam_trt_light/
python scripts/trt_inference.py --checkpoint=downloads/sam_vit_b_01ec64.pth --input-image=images/apples.jpg --mode point --visualize --output-image=output.png --model-type vit_b --onnx-dir onnx --engine-dir engine
cd /workspace && \
exec /bin/bash
"