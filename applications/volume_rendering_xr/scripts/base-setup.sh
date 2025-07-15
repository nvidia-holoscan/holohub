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

WINDRUNNER_VERSION=${WINDRUNNER_VERSION:-1.11.74-5b8fe25-1~20240426ubuntu2204}

if [ ! -e /usr/local/share/keyrings/magicleap/MagicLeapRemoteRendering.gpg ]; then
    echo "Could not find MagicLeapRemoteRendering.gpg"
    ls /usr/local/share/keyrings/magicleap
    exit 1
fi

apt update

# Install dependencies
#  libvulkan1 - Vulkan loader
#  libegl1 - to run headless Vulkan apps
apt install --no-install-recommends -y \
    libopenxr-loader1 \
    libopenxr-dev \
    libvulkan1 \
    libegl1 \
    net-tools
rm -rf /var/lib/apt/lists/*

# Install Magic Leap Windrunner OpenXR backend
LIST_FILE=/etc/apt/sources.list.d/MagicLeapRemoteRendering.list
chmod -R 755 "/usr/local/share/keyrings/magicleap/"
CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
echo "deb [signed-by=/usr/local/share/keyrings/magicleap/MagicLeapRemoteRendering.gpg] https://apt.magicleap.cloud/Stable/ $CODENAME main" \
        | tee "$LIST_FILE"
chmod a+r "$LIST_FILE"
apt update
echo "debconf-set-selections windrunner/accept_eula boolean true" | debconf-set-selections
apt install --no-install-recommends -y \
        windrunner-service=${WINDRUNNER_VERSION} \
        libopenxr1-windrunner=${WINDRUNNER_VERSION}
rm -rf /var/lib/apt/lists/*

printf 'Package: windrunner-service
Pin: version %s
Pin-Priority: 1337

Package: libopenxr1-windrunner
Pin: version %s
Pin-Priority: 1337
' ${WINDRUNNER_VERSION} ${WINDRUNNER_VERSION} > /etc/apt/preferences.d/pin-windrunner
update-alternatives --set openxr1-active-runtime /usr/share/openxr/1/openxr_windrunner.json

#  https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/
mkdir -p "/workspace/holohub/.cache/ComputeCache"
#  https://raytracing-docs.nvidia.com/optix7/api/optix__host_8h.html#a59a60f5f600df0f9321b0a0b1090d76b
mkdir -p "/workspace/holohub/.cache/OptixCache"
# https://download.nvidia.com/XFree86/Linux-x86_64/460.67/README/openglenvvariables.html
mkdir -p "/workspace/holohub/.cache/GLCache"
