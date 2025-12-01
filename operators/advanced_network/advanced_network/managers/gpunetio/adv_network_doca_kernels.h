/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include "adv_network_doca_mgr.h"

#if __cplusplus
extern "C" {
#endif

doca_error_t doca_receiver_packet_kernel(cudaStream_t stream, int rxqn, uintptr_t* eth_rxq_gpu,
                                         uintptr_t* pkt_gpu_list, uint32_t* pkt_idx_list,
                                         uint32_t* batch_list, uint32_t* gpu_exit_condition,
                                         bool persistent);

doca_error_t doca_sender_packet_kernel(cudaStream_t stream, struct doca_gpu_eth_txq* txq,
                                       uint64_t pkt_buff_addr, const uint32_t pkt_buff_mkey,
                                       uint32_t gpu_pkt0_idx,
                                       const size_t num_pkts, uint32_t max_pkts,
                                       const uint64_t max_pkt_size,
                                       uint32_t* gpu_pkts_len, bool set_completion);
#if __cplusplus
}
#endif
