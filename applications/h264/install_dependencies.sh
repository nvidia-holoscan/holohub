#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This scripts installs gxf multimedia libraies and corresponding dependecies
# required for H264 Encode / Decode applications.

# URL of the gxf-mm release
GXF_MM_URL="https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/nightly/extensions/multimedia/release-3.1/multimedia_release_5.tar"

# URL of the DeepStream dependencies required for gxf-mm extensions above
DS_DEPS_URL="https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/internal/multimedia/nvv4l2_x86_ds-6.4.deb"

ARCH=$(arch)
HOLOSCAN_LIBS_DIR=/opt/nvidia/holoscan/lib/

# Download and install gxf-mm libs depending on the architecture
if [[ $ARCH == "x86_64" ]]; then
  SUBFOLDER="gxf_x86_64_cuda_12_2"
elif [[ $ARCH == "aarch64" ]]; then
  SUBFOLDER="gxf_hp21ea_sbsa"
else
  echo "Unsupported architecture"
  exit 1
fi
wget "$GXF_MM_URL" -O gxf-mm.tar
mkdir -p gxf-mm
tar -xf gxf-mm.tar -C gxf-mm
rm gxf-mm.tar
mkdir -p /opt/nvidia/holoscan/lib
find ./gxf-mm/$SUBFOLDER/ -iname *.so -execdir cp "{}" $HOLOSCAN_LIBS_DIR ";"
rm -rf gxf-mm

# Download and install DeepStream dependencies, required only on x86_64.
echo "Downloading and installing DeepStream dependencies"
if [[ $ARCH == "x86_64" ]]; then
  wget "$DS_DEPS_URL" -O ds_deps.deb
  dpkg -i ./ds_deps.deb
  rm ds_deps.deb
fi

echo "Installation completed successfully."


