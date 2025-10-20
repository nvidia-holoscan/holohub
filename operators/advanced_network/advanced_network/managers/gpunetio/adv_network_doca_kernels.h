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

#pragma once
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>
#include <doca_gpunetio_dev_sem.cuh>
#include <doca_gpunetio_dev_buf.cuh>
#include "adv_network_doca_mgr.h"

#if __cplusplus
extern "C" {
#endif

doca_error_t doca_receiver_packet_kernel(cudaStream_t stream, int rxqn, uintptr_t* eth_rxq_gpu,
                                         uintptr_t* sem_gpu, uint32_t* sem_idx_list,
                                         uint32_t* batch_list, uint32_t* gpu_exit_condition,
                                         bool persistent);
doca_error_t doca_sender_packet_kernel(cudaStream_t stream, struct doca_gpu_eth_txq* txq,
                                       struct doca_gpu_buf_arr* buf_arr, uint32_t gpu_pkt0_idx,
                                       const size_t num_pkts, uint32_t max_pkts,
                                       uint32_t* gpu_pkts_len, bool set_completion);
#if __cplusplus
}
#endif
