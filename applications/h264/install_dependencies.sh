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

# This scripts installs gxf multimedia libraies and corresponding dependecies
# required for H264 Encode / Decode applications.

# URL of the DeepStream dependencies required for gxf-mm extensions above
DS_DEPS_URL="https://api.ngc.nvidia.com/v2/resources/org/nvidia/gxf_and_gc/4.0.0/files?redirect=true&path=nvv4l2_x86_ds-7.0.deb"

ARCH=$(arch)
HOLOSCAN_LIBS_DIR=/opt/nvidia/holoscan/lib/

# Download and install gxf-mm libs depending on the architecture
if [[ $ARCH == "x86_64" ]] || [[ $ARCH == "aarch64" ]]; then
  echo "Downloading and installing GXF H.264 Encode / Decode libraries for ${ARCH}..."
else
  echo "Unsupported architecture"
  exit 1
fi
declare -A gxf_mm_extensions
gxf_mm_extensions=(["decoderio"]="45081ccb-982e-4946-96f9-0d684f2cfbd0" \
                   ["encoderio"]="1a0a562d-378c-4618-bb5e-d3f70825aed2" \
                   ["decoder"]="edc99001-73bd-435c-af0c-e013dcda3439" \
                   ["encoder"]="ea5c44e4-15db-4448-a3a6-f32004303338")
mkdir -p gxf-mm
mkdir -p ${HOLOSCAN_LIBS_DIR}
CUDA_VERSION=12.2
for extension in "${!gxf_mm_extensions[@]}"; do
  if [[ $extension == "decoderio" ]] || [[ $extension == "encoderio" ]]; then
    extension_url="https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/graph-composer/video${extension}extension/1.2.0-linux-${ARCH}-ubuntu_22.04/files?redirect=true&path=${gxf_mm_extensions[${extension}]}.tar.gz"
  else
    extension_url="https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/graph-composer/video${extension}extension/1.2.0-linux-${ARCH}-ubuntu_22.04-cuda-${CUDA_VERSION}/files?redirect=true&path=${gxf_mm_extensions[${extension}]}.tar.gz"
  fi
  extension_tar=${gxf_mm_extensions[$extension]}.tar.gz
  wget --content-disposition ${extension_url} -O ${extension_tar}
  tar -xvf ${extension_tar} -C gxf-mm/
  rm ${extension_tar}
  extensions_lib="libgxf_video${extension}.so"
  cp ./gxf-mm/${extensions_lib} ${HOLOSCAN_LIBS_DIR}/
done
rm -rf gxf-mm

# Download and install DeepStream dependencies, required only on x86_64.
echo "Downloading and installing DeepStream dependencies..."
if [[ $ARCH == "x86_64" ]]; then
  wget --content-disposition "$DS_DEPS_URL" -O ds_deps.deb
  dpkg -i ./ds_deps.deb
  rm ds_deps.deb
fi

echo "Installation completed successfully."
