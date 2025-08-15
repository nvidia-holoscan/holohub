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

if [ -z "${STREAMING_CONTAINER_IMAGE:+x}" ]; then
    echo "STREAMING_CONTAINER_IMAGE must be set"
    exit 1
fi

if [ -z "${STREAMING_FUNCTION_ID:+x}" ]; then
    echo "STREAMING_FUNCTION_ID must be set"
    exit 1
fi

if [ -z "${STREAMING_FUNCTION_NAME:+x}" ]; then
    echo "STREAMING_FUNCTION_NAME must be set"
    exit 1
fi

if [ -z "${STREAMING_SERVER_PORT:+x}" ]; then
    STREAMING_SERVER_PORT=49100
    echo "STREAMING_SERVER_PORT not set, using default: "$STREAMING_SERVER_PORT
fi

if [ -z "${HTTP_SERVER_PORT:+x}" ]; then
    HTTP_SERVER_PORT=8011
    echo "HTTP_SERVER_PORT not set, using default: "$HTTP_SERVER_PORT
fi

if [ -z "${NGC_DOMAIN:+x}" ]; then
    NGC_DOMAIN=api.ngc.nvidia.com
    echo "NGC_DOMAIN not set, using default: "$NGC_DOMAIN
fi

response=$(curl -s --location --request POST 'https://'$NGC_DOMAIN'/v2/nvcf/functions/'$STREAMING_FUNCTION_ID'/versions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer '$NGC_PERSONAL_API_KEY'' \
--data '{
  "name": "'$STREAMING_FUNCTION_NAME'",
  "inferenceUrl": "/sign_in",
  "inferencePort": '$STREAMING_SERVER_PORT',
  "health": {
    "protocol": "HTTP",
    "uri": "/v1/streaming/ready",
    "port": '$HTTP_SERVER_PORT',
    "timeout": "PT10S",
    "expectedStatusCode": 200
  },
  "containerImage": "'$STREAMING_CONTAINER_IMAGE'",
  "apiBodyFormat": "CUSTOM",
  "description": "'$STREAMING_FUNCTION_NAME'",
  "functionType": "STREAMING"
}
')

function_id=$(echo $response | jq -r '.function.id')
function_version_id=$(echo $response | jq -r '.function.versionId')

echo "============================="
echo "Function Updated Successfully"
echo "Function ID: "$function_id
echo "Function Version ID: "$function_version_id
echo "============================="
