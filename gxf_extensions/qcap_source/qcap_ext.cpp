/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 YUAN High-Tech Development Co., Ltd. All rights reserved.
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

#include "qcap_source.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0xc2579b7f68df4303, 0x915cb5af54592e41, "QCAP", "QCAP Extension", "YUAN",
                         "1.0.0", "LICENSE");
GXF_EXT_FACTORY_ADD(0x32c1951e1bcb4f42, 0x91275278040afa5a, yuan::holoscan::QCAPSource,
                    nvidia::gxf::Codelet, "QCAP Source Codelet");
GXF_EXT_FACTORY_END()
