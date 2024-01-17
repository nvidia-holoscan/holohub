/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "adv_network_mgr.h"
#include "adv_network_dpdk_mgr.h"
#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

ANOMgr *g_ano_mgr = nullptr;

/* This function decides what ANO backend or "manager" is used for packet processing. The choice of
   manager is based on what we believe is the best selection based on the user's configuration. */
void set_ano_mgr(const AdvNetConfigYaml &cfg) {
  if (g_ano_mgr == nullptr) {
    if (1) {
      HOLOSCAN_LOG_INFO("Selecting DPDK as ANO manager");
      g_ano_mgr = new DpdkMgr{};
    }
    else {
      HOLOSCAN_LOG_CRITICAL("Failed to set ANO manager");
    }
  }
}

};