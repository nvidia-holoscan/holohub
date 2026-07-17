# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  COMPONENTS
    daqiri_raw_ethernet_bench-cpp
    daqiri_raw_ethernet_bench-configs
    daqiri_raw_ethernet_bench-cppsrc
    daqiri_socket_ping-cpp
    daqiri_socket_ping-configs
    daqiri_socket_ping-cppsrc
    # daqiri_raw_ethernet_bench-py
    # daqiri_socket_ping-py
  EXPORT_NAME holoscan-networking-targets
  # Must stay in sync with pkg/holoscan-networking/Dockerfile: CUDA_VER 13.x guard,
  # hsdk-remote HOLOSCAN_DEB_REMOTE_VERSION bounds, and hsdk-local deb validation.
  DEPENDS "holoscan-cuda-13 (>= 4.0), holoscan-cuda-13 (<< 5.0), daqiri"
  RECOMMENDS "infiniband-diags, ibverbs-utils, mlnx-ofed-kernel-utils, mft, libhugetlbfs-bin"
)

# NOTE: could be broken down in
#   holoscan-networking-common
#   holoscan-networking-dpdk
#   holoscan-networking (umbrella)
