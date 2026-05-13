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
    basic_networking_ping-cpp
    basic_networking_ping-configs
    basic_networking_ping-cppsrc
    adv_networking_bench-cpp
    adv_networking_bench-configs
    adv_networking_bench-cppsrc
    # basic_networking_ping-py
    # adv_networking_bench-py
  EXPORT_NAME holoscan-networking-targets
  DEPENDS "holoscan (>= 3.0), holoscan (<< 4.0), daqiri"
  RECOMMENDS "infiniband-diags, ibverbs-utils, mlnx-ofed-kernel-utils, mft, libhugetlbfs-bin"
)

# NOTE: could be broken down in
#   holoscan-networking-common
#   holoscan-networking-dpdk
#   holoscan-networking (umbrella)
