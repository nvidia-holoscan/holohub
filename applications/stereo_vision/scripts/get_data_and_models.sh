#SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#SPDX-License-Identifier: Apache-2.0

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

if [ "$#" -ne 1 ]; then
    echo "Error: expecting path/to/data as only input argument"
    exit 1
fi

SCRIPTDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CURDIR=$(pwd)
mkdir -p "$1"
cd "$1"

#get the yolo model from ultralytics
mkdir -p source/yolo
cd source/yolo
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
#add the non-maximum suspression and export an onnx file usable by the inference operator
git clone https://github.com/triple-Mu/YOLOv8-TensorRT.git
pip install onnx torch ultralytics onnx_graphsurgeon
python3 YOLOv8-TensorRT/export-det.py --weights yolov8s.pt --iou-thres 0.65 --conf-thres 0.25 --topk 100 --opset 11 --sim --input-shape 1 3 640 640 --device cuda:0
cd "$CURDIR"

# Download graph_surgeon script
wget --content-disposition 'https://raw.githubusercontent.com/nvidia-holoscan/holoscan-sdk/refs/tags/v2.8.1/scripts/graph_surgeon.py' -O $SCRIPTDIR/graph_surgeon.py
python3 "$SCRIPTDIR/graph_surgeon.py" "$1/source/yolo/yolov8s.onnx" "$1/yolov8-nms-update.onnx"
rm $SCRIPTDIR/graph_surgeon.py

#get the ess and tao converter from ngc to produce engine file for ess stereo matching
mkdir -p "$1/source/ess"
cd "$1/source/ess"

# Check the architecture
arch=$(uname -m)
if [ "$arch" == "x86_64" ]; then
    wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/tao/tao-converter/v5.1.0_8.6.3.1_x86/files?redirect=true&path=tao-converter' -O tao-converter
elif [ "$arch" == "aarch64" ]; then
    wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/tao/tao-converter/v5.1.0_jp6.0_aarch64/files?redirect=true&path=tao-converter' -O tao-converter
else
    echo "Error: Unknown architecture: $arch"
fi
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/isaac/dnn_stereo_disparity/3.0.0/files?redirect=true&path=ess.etlt' -O ess.etlt
chmod +x tao-converter
./tao-converter ess.etlt -k ess -t fp32 -o output_left,output_conf -e "../../ess.engine"
cd "$CURDIR"
