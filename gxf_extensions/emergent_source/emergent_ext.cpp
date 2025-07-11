/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "emergent_source.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0xa874c830207a47e7, 0x9270c6997e5ff8dc, "Emergent", "Emergent Extension",
                        "NVIDIA", "1.0.0", "LICENSE");
GXF_EXT_FACTORY_ADD(0xcaf7eb335da24a28, 0xbdcd3f3319a356db, nvidia::holoscan::EmergentSource,
                    nvidia::gxf::Codelet, "Emergent Source Codelet");
GXF_EXT_FACTORY_END()
