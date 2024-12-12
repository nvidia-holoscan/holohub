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

holohub_configure_deb(
  NAME "holoscan-networking"
  DESCRIPTION "Holoscan Networking"
  VERSION "0.1.0"
  VENDOR "NVIDIA"
  CONTACT "Alexis Girault <agirault@nvidia.com>"
  DEPENDS "holoscan (>= 2.6), mlnx-dpdk, doca-sdk-gpunetio, doca-sdk-flow, doca-sdk-eth"
)

# NOTE: could be broken down in
#   holoscan-networking-common
#   holoscan-networking-dpdk
#   holoscan-networking-gpunetio
#   holoscan-networking-rivermax
#   holoscan-networking (umbrella)
