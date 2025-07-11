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

# Set shell welcome message
if ! grep -q "Welcome" ~/.zshrc 2>/dev/null ; then
   echo "echo \"Welcome to Holohub!\n\n\
Here's a list of useful locations:\n\n\
       - Workspace: ${WORKSPACE_DIR} \n\
       - Holoscan SDK Installation: /opt/nvidia/holoscan \n\
       - Holoscan SDK Source Code: /workspace/holoscan-sdk \n\"" >> ~/.zshrc
fi


echo Updating permissions...
sudo chown -R $USER ~/

# Hide pkexec as pythoncpp debugger does not work correctly with it
# WHen the pythoncpp debugger finds pkexec, it tries to execute the gdb debugger as another user
# using pkexec instead of sudo. This causes the debugger to fail to attach to the process.
if [ -f /usr/bin/pkexec ]; then
    echo "Hiding pkexec..."
   sudo mv /usr/bin/pkexec /usr/bin/pkexec.old
fi

echo Done!