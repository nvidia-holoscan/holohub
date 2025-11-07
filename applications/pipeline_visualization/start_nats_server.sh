#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Error out if a command fails or a variable is not defined
set -eu

tmp_dir="$(mktemp -d)"

# use the default nats-server.conf without cluster configuration file but extend the max_payload to 8MB
cat <<EOF > "${tmp_dir}/nats-server.conf"
# Client port of 4222 on all interfaces
port: 4222

# HTTP monitoring port
monitor_port: 8222

max_payload: 8MB
EOF

# Start the NATS server in a docker container
docker run --network host -ti -v "$tmp_dir/nats-server.conf:/nats-server.conf" nats:latest "$@"

trap 'rm -rf $tmp_dir' EXIT