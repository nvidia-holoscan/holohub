#!/bin/bashSPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
# reserved. SPDX-License-Identifier: Apache-2.0
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

# Set the maximum send and receive socket buffer sizes to match what
# is used in the UDPv4 section of the QoS profile (qos_profiles.xml).
# For more details, see the RTI Connext guide to Improve DDS Network
# Performance on Linux Systems:
#   https://community.rti.com/howto/improve-rti-connext-dds-network-performance-linux-systems
sudo sysctl -w net.core.rmem_max="4194304"
sudo sysctl -w net.core.wmem_max="4194304"
