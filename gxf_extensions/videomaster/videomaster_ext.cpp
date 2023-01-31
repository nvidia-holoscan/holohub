/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "gxf/std/extension_factory_helper.hpp"
#include "videomaster_source.hpp"
#include "videomaster_transmitter.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0x46841212e268490f, 0xa2f9a144bcbe34d0, "VideoMaster", "VideoMaster Extension", "NVIDIA",
                         "1.0.0", "LICENSE");
GXF_EXT_FACTORY_ADD(0x3c86dd3b82cf4a30, 0x99e14676542c5d70, nvidia::holoscan::videomaster::VideoMasterSource,
                    nvidia::gxf::Codelet, "VideoMaster Source Codelet");
GXF_EXT_FACTORY_ADD(0x1178a1b2fea542b8, 0x947ed0d4e03ca86a, nvidia::holoscan::videomaster::VideoMasterTransmitter,
                    nvidia::gxf::Codelet, "VideoMaster Transmitter Codelet");
GXF_EXT_FACTORY_END()
