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

#get the ess and tao converter from ngc to produce engine file for ess stereo matching
mkdir -p $1/source/ess
cd $1/source/ess

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
