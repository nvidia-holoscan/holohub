#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is the entrypoint file for the docker release container. Executed in container at runtime.

if [ $# == 0 ] && [ "$STARTED_FROM_SCRIPT" != "1" ]; then
    # the image has been run directly help the user to get started by outputting the render_volume_xr script.
    cat /scripts/docker-run-release.sh
    echo ""
    echo "##############################################################################"
    echo "# Please redirect this output to a script and use the script to execute the app"
    echo "#   $ docker run --rm holoscan-openxr > ./render-volume-xr"
    echo "#   $ chmod +x ./render-volume-xr"
    echo "# Run app with default dataset"
    echo "#   $ ./render-volume-xr"
    echo "# List datasets"
    echo "#   $ ./render-volume-xr --list"
    echo "# Run app with dataset"
    echo "#   $ ./render-volume-xr --dataset DATASET_NAME"
    echo "##############################################################################"
    exit 0
fi

# arguments for each dataset
declare -A datasets
datasets[highResCT]="-c configs/ctnv_bb_er.json -d data/volume_rendering_xr/highResCT.mhd -m data/volume_rendering_xr/smoothmasks.seg.mhd"
# developers may manually add more datasets here for local packaging:
# datasets[name]="..."

# Parse arguments
ARGS=("$@")
dataset="highResCT"
SKIP_NEXT=0
for i in "${!ARGS[@]}"; do
    arg="${ARGS[i]}"
    if [ "${SKIP_NEXT}" = "1" ]; then
        continue
    elif [ "$arg" = "--help" ]; then
        echo "Holoscan OpenXR demo release container entrypoint"
        echo ""
        echo "Arguments:"
        echo "  --help : print this help message"
        echo "  --list : list available datasets"
        echo "  --dataset <dataset> : use the specified dataset"
        echo ""
        echo "Remaining arguments are passed to the app. Additional options:"
        echo ""
        /workspace/holohub/install/bin/volume_rendering_xr --help
        exit 0
    elif [ "$arg" = "--list" ]; then
        echo "Available datasets"
        echo ${!datasets[@]}
        exit 0
    elif [ "$arg" = "--dataset" ]; then
        dataset="${ARGS[i+1]}"
        SKIP_NEXT="1"
    else
        arguments+=" $arg"
    fi
done

if [[ -z $dataset ]]; then
    echo "No dataset selected"
    exit 1
else 
    if [[ ${datasets[@]} =~ $dataset ]]; then
        arguments+="${datasets[$dataset]}"
    else
        echo "Dataset $dataset not found"
        exit -1
    fi
fi

# run the windrunner service in the background
# Make sure sys_nice setting is accurate for the current host
dpkg-reconfigure windrunner-service

# Workaround: need to manually start pulseaudio before windrunner-service
# in the cuda-runtime container
pulseaudio --check
pulseaudio -D

# Workaround: need to set the HOME directory, windrunner service is expecting this to exist to write the log files
HOME=/tmp
XRT_SKIP_STDIN='true' windrunner-service &

# display the pairing QR code
setup_viewer --mode qr --qr_output terminal

# now run the app
echo "Running from $(pwd): /workspace/holohub/install/bin/volume_rendering_xr $arguments"
/workspace/holohub/install/bin/volume_rendering_xr $arguments
