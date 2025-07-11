#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

printf "#################### .devcontainer/post-start-command.sh ####################\n"
printf "#                                                                           #\n"
printf "# Edit this file to add commands that should be run after the container is  #\n"
printf "# started.                                                                  #\n"
printf "#                                                                           #\n"
printf "#############################################################################\n"

# Create directories and set permissions for the holoscan user
sudo mkdir -p /home/holoscan/

echo Copying Holoscan SDK example applications to ~/examples...
sudo cp -r /opt/nvidia/holoscan/examples /home/holoscan/examples

echo Configuring /home/holoscan for $USER...
sudo chown -R $USER:$USER /home/holoscan

# Set shell welcome message
if ! grep -q "Welcome" ~/.zshrc 2>/dev/null ; then
   echo "echo \"Welcome to Holoscan SDK NGC container!\n\n\
Here's a list of useful locations:\n\n\
       - Workspace: /workspace \n\
       - Holoscan SDK Installation: /opt/nvidia/holoscan \n\
       - Holoscan SDK Source Code: /workspace/holoscan-sdk \n\
       - Sample applications: /workspace/holoscan-sdk/examples\n\n\
       - Sample data: /opt/nvidia/holoscan/data\n\n\
Checkout the /workspace/my/README.md file for more information on the Holoscan SDK DevContainer.\n\n\"" >> ~/.zshrc
fi
