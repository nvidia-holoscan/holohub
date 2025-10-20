#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Read arguments.
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-sdk)
      SKIP_SDK=1
      shift # past argument
      ;;
    --load-driver)
      LOAD_DRIVER=1
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      echo "Usage: $(basename $0) [--skip-sdk] [--load-driver]"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters
basedir=$(pwd)

# Set the appropriate build flags.
echo "=========================================================="
echo -n "  Building AJA driver "
if [ -z "$SKIP_SDK" ]; then
    echo -n "and SDK "
fi
echo -n "with RDMA support for "
export AJA_RDMA=1
if lsmod | grep -q nvgpu ; then
    echo "iGPU"
    export AJA_IGPU=1
else
    echo "dGPU"
    unset AJA_IGPU
fi
echo "==========================================================" && echo

# Ensure the open source dGPU driver is being used.
if [ -z "$AJA_IGPU" ]; then
    LICENSE=$(modinfo -l nvidia)
    if [ "$LICENSE" == "NVIDIA" ]; then
        echo "ERROR: The open source NVIDIA drivers are required for RDMA support"
        echo "       but the closed source drivers are currently installed. Please"
        echo "       install the open source drivers then run this script again."
        exit 1
    fi
fi

# Ensure CMake is installed.
if [ -z "$SKIP_SDK" ]; then
    if ! command -v cmake &> /dev/null; then
        echo "ERROR: CMake is not installed. Install it with the following then"
        echo "       run this script again:"
        echo "         sudo apt install -y cmake"
        exit 1
    fi
fi

# Checkout the libajantv2 repo.
if [ ! -d libajantv2 ]; then
    git clone https://github.com/nvidia-holoscan/libajantv2.git
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to checkout libajantv2 repo."
        exit 1
    fi
    cd libajantv2/
else
    cd libajantv2/ && git pull
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to checkout libajantv2 repo."
        exit 1
    fi
fi

# Build the driver.
make -j --directory driver/linux/
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build libajantv2 driver."
    exit 1
fi

# Build the SDK.
if [ -z "$SKIP_SDK" ]; then
    mkdir -p build && cd build
    cmake .. -Wno-dev && make -j
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build libajantv2 SDK."
        exit 1
    fi
    if ! [ -f tools/rdmawhacker/rdmawhacker ]; then
        echo && echo "WARNING: rdmawhacker build was skipped. Is CUDA installed?"
    fi
fi

# Load the driver.
if [ -n "$LOAD_DRIVER" ]; then
    echo && echo "=========================================================="
    echo "Loading AJA driver..."
    cd $basedir
    sudo ./libajantv2/driver/bin/load_ajantv2
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to load AJA driver."
        exit 1
    fi
    if [ -z "$SKIP_SDK" ]; then
        echo && echo "Enumerating AJA Devices:"
        ./libajantv2/build/demos/ntv2enumerateboards/ntv2enumerateboards
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to enumerate AJA devices."
            exit 1
        fi
    fi
fi

# Finish up.
echo && echo "============================================================"
echo "SUCCESS!"
if [ -z "$LOAD_DRIVER" ]; then
    echo "Load driver using 'sudo ./libajantv2/driver/bin/load_ajantv2'"
    if [ -f ${basedir}/libajantv2/build/demos/ntv2enumerateboards/ntv2enumerateboards ]; then
        echo "Use ntv2enumerateboards tool to list available AJA devices:"
        echo "  ./libajantv2/build/demos/ntv2enumerateboards/ntv2enumerateboards"
    fi
fi
if [ -f ${basedir}/libajantv2/build/tools/rdmawhacker/rdmawhacker ]; then
    echo "Use rdmawhacker tool to check RDMA is functional (CTRL-C to exit):"
    echo "  ./libajantv2/build/tools/rdmawhacker/rdmawhacker"
fi
exit 0
