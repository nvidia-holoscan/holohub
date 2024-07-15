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

printf "#################### .devcontainer/post-start-command.sh ####################\n"
printf "#                                                                           #\n"
printf "# Edit this file to add commands that should be run after the container is  #\n"
printf "# started.                                                                  #\n"
printf "#                                                                           #\n"
printf "#############################################################################\n"

# Create directories and set permissions for the holoscan user
sudo mkdir -p /home/holoscan/.config/fish

echo Copying Holoscan SDK example applications to ~/examples...
sudo cp -r /opt/nvidia/holoscan/examples /home/holoscan/examples

echo Configuring /home/holoscan for $USER...
sudo chown -R $USER:$USER /home/holoscan

# Set shell welcome message
fish -c "set -U fish_greeting Welcome to Holoscan SDK NGC container!\n\n \
       Here\'s a list of useful locations:\n\n \
        \t- Workspace: (set_color yellow)/workspace(set_color normal) \n \
        \t- Holoscan SDK: (set_color yellow)/opt/nvidia/holoscan(set_color normal) \n \
        \t- Sample data (set_color yellow)HOLOSCAN_INPUT_PATH(set_color normal): (set_color yellow)/opt/nvidia/holoscan/data(set_color normal).\n \
        \t- Sample applications: (set_color yellow)~/examples(set_color normal).\n\n \
        You are using (set_color yellow)fish(set_color normal) shell, for more information, visit https://fishshell.com/.\n\n  \
        Checkout the (set_color yellow)/workspace/README.md(set_color normal) file for more information on the Holoscan SDK DevContainer.\n\n"

