#!/bin/bash

#
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
#

if [ -z "${NGC_PERSONAL_API_KEY:+x}" ]; then
    echo "NGC_PERSONAL_API_KEY must be set"
    exit 1
fi

if [ -z "${STREAMING_FUNCTION_ID:+x}" ]; then
    echo "STREAMING_FUNCTION_ID must be set"
    exit 1
fi

if [ -z "${NVCF_SERVER:+x}" ]; then
    NVCF_SERVER=grpc.nvcf.nvidia.com
    echo "NVCF_SERVER not set, using default: "$NVCF_SERVER
fi

echo Launching test intermediate proxy for function id [$STREAMING_FUNCTION_ID] on [$NVCF_SERVER]

mkdir -p _test_proxy
CONF_DIR=./_test_proxy
cat << EOF > $CONF_DIR/haproxy.cfg
global
        log /dev/log    local0
        log /dev/log    local1 notice
        stats timeout 30s
        user haproxy
        daemon

        # Default SSL material locations
        ca-base /etc/ssl/certs
        crt-base /etc/ssl/private

        # SSL server verification enabled for security
        ssl-server-verify required

        # See: https://ssl-config.mozilla.org/#server=haproxy&server-version=2.0.3&config=intermediate
        ssl-default-bind-ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384
        ssl-default-bind-ciphersuites TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256
        ssl-default-bind-options ssl-min-ver TLSv1.2 no-tls-tickets

defaults
        log     global
        option  httplog
        option  dontlognull
        timeout connect 5000
        timeout client  50000
        timeout server  50000

frontend test_frontend
        log  global
        bind *:49100
        mode http
        option  httplog
        timeout client  7s
        timeout http-request 30m
        use_backend webrtc_backend

backend webrtc_backend
        log  global
        mode http
        timeout connect 4s
        timeout server 7s
        http-request set-header Host $NVCF_SERVER
        http-request set-header Authorization "Bearer $NGC_PERSONAL_API_KEY"
        http-request set-header Function-ID $STREAMING_FUNCTION_ID
        server s1 $NVCF_SERVER:443 ssl verify required
EOF
docker run --rm --net=host --name test-intermediate-haproxy -v $CONF_DIR:/usr/local/etc/haproxy:ro haproxy:2.4

