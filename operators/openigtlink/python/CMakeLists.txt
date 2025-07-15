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

include(pybind11_add_holohub_module)

pybind11_add_holohub_module(
    CPP_CMAKE_TARGET openigtlink_rx
    CLASS_NAME OpenIGTLinkRxOp
    SOURCES openigtlink_rx.cpp
)
target_link_libraries(openigtlink_rx_python PRIVATE openigtlink_rx)

pybind11_add_holohub_module(
    CPP_CMAKE_TARGET openigtlink_tx
    CLASS_NAME OpenIGTLinkTxOp
    SOURCES openigtlink_tx.cpp
)
target_link_libraries(openigtlink_tx_python PRIVATE openigtlink_tx)