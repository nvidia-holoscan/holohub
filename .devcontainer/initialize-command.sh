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

printf "#################### .devcontainer/initialize-command.sh ####################\n"
printf "#                                                                           #\n"
printf "# Edit this file to add commands that should be run on the host machine     #\n"
printf "# during initialization.                                                    #\n"
printf "#                                                                           #\n"
printf "#############################################################################\n"

# Use default docker buildx builder
docker buildx use default