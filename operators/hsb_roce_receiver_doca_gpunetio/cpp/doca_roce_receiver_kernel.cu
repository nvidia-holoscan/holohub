/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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
 *
 * Normal (non-persistent) CUDA kernels for DOCA GPUNetIO RoCE receive.
 * These run inside the CUDA Graph WHILE node managed by the GPU-resident
 * execution engine.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <doca_error.h>
#include <doca_gpunetio_dev_verbs_twosided.cuh>
#include <holoscan/utils/cuda_macros.hpp>

#include "doca_helpers.hpp"

#define RWQE_TERMINATE_KEY 0x100

static constexpr unsigned int CTRL_DATA_NOT_READY = 1;
static constexpr unsigned int CTRL_DATA_READY = 2;

// ---------------------------------------------------------------------------
// Pre-post all receive WQEs (called once during setup).
// ---------------------------------------------------------------------------
__global__ void prepare_receive(struct doca_gpu_dev_verbs_qp* qp,
                                const size_t frame_size) {
    doca_gpu_dev_verbs_ticket_t out_ticket;
    doca_gpu_dev_verbs_recv<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                            DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(
        qp,
        doca_gpu_dev_verbs_addr{
            .addr = 0,
            .key = doca_gpu_dev_verbs_bswap32(RWQE_TERMINATE_KEY)},
        0, &out_ticket);
}

// ---------------------------------------------------------------------------
// Non-blocking CQ check + data-ready kernel.
//
// Runs inside the CUDA Graph WHILE node.  Performs a non-blocking check of
// the Receive CQ for the next RDMA Write + Immediate Data completion.
// If no completion is available, sets *data_ready_ptr = CTRL_DATA_NOT_READY
// and returns immediately so the WHILE node retries.
// When a completion arrives, extracts the frame address from the ring buffer
// and writes it to *chosen_frame_memory, then marks CTRL_DATA_READY.
//
// rx_state layout: [out_ticket, wqe_idx]
// ---------------------------------------------------------------------------
__global__ void doca_receive_data_ready_kernel(
    struct doca_gpu_dev_verbs_qp* qp,
    uint8_t* ring_buf, const size_t ring_buf_stride_sz,
    const uint32_t ring_buf_stride_num,
    uint64_t* rx_state,
    unsigned char** chosen_frame_memory,
    unsigned int* data_ready_ptr) {
  if (qp == nullptr) {
    printf("ERROR: qp is nullptr in doca_receive_data_ready_kernel\n");
    *data_ready_ptr = CTRL_DATA_NOT_READY;
    return;
  }

    doca_gpu_dev_verbs_ticket_t out_ticket =
        (doca_gpu_dev_verbs_ticket_t)rx_state[0];  // which cqe to look
    uint64_t wqe_idx = rx_state[1];                // which wqe to submit next

    struct doca_gpu_dev_verbs_cq* cq_rq = doca_gpu_dev_verbs_qp_get_cq_rq(qp);

    // Non-blocking poll: returns 0 if CQE ready, EBUSY if not yet.
    int poll_status =
        doca_gpu_dev_verbs_poll_one_cq_at<
            DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
            DOCA_GPUNETIO_VERBS_QP_RQ>(cq_rq, out_ticket);

    if (poll_status != 0) {
        *data_ready_ptr = CTRL_DATA_NOT_READY;
        return;
    }

    // CQE available - extract page index from immediate data (low 8 bits).
    uint8_t* cqe = (uint8_t*)__ldg((uintptr_t*)&cq_rq->cqe_daddr);  // device address of the CQE
    const uint32_t cqe_mask = (__ldg(&cq_rq->cqe_num) - 1);
    // out_ticket % cqe_num is the index of the CQE in the CQ
    // then go to the exact address of the CQE in the CQ
    struct mlx5_cqe64* cqe64 =
        (struct mlx5_cqe64*)(cqe + ((out_ticket & cqe_mask) *
                                     DOCA_GPUNETIO_VERBS_CQE_SIZE));
    // convert from network byte order to gpu device (host) byte order
    uint32_t stride = doca_gpu_dev_verbs_bswap32(cqe64->imm_inval_pkey) & 0xFF;

    if (stride < ring_buf_stride_num) {
        *((volatile unsigned char**)chosen_frame_memory) =
            (unsigned char*)(ring_buf + ((uint64_t)ring_buf_stride_sz * stride));
        *data_ready_ptr = CTRL_DATA_READY;
    } else {
        *data_ready_ptr = CTRL_DATA_NOT_READY;
    }

    // Advance both counters independently:
    //  - out_ticket: we consumed one CQE, move to the next CQ slot.
    //  - wqe_idx:    we used one receive WQE, post the next one.
    out_ticket += 1;
    wqe_idx += 1;
    doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA,
                              DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA,
                              DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB,
                              DOCA_GPUNETIO_VERBS_QP_RQ>(qp, wqe_idx + 1);

    rx_state[0] = out_ticket;
    rx_state[1] = wqe_idx;
}

// ============================== Launchers ==================================

extern "C" {

doca_error_t DocaRoceReceiverPrepareKernel(
    cudaStream_t stream, struct doca_gpu_dev_verbs_qp* qp,
    size_t frame_size, uint32_t cuda_blocks, uint32_t cuda_threads) {
    void* args[] = {&qp, &frame_size};
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaLaunchKernel((const void*)prepare_receive,
                         dim3(cuda_blocks), dim3(cuda_threads),
                         args, 0, stream),
        "Failed to launch prepare_receive kernel");
    return DOCA_SUCCESS;
}

doca_error_t DocaRoceReceiverDataReadyKernel(
    cudaStream_t stream,
    struct doca_gpu_dev_verbs_qp* qp,
    uint8_t* ring_buf, size_t ring_buf_stride_sz,
    uint32_t ring_buf_stride_num,
    uint64_t* rx_state,
    unsigned char** chosen_frame_memory,
    unsigned int* data_ready_ptr,
    uint32_t cuda_blocks, uint32_t cuda_threads) {
    void* args[] = {&qp, &ring_buf, &ring_buf_stride_sz,
                    &ring_buf_stride_num, &rx_state,
                    &chosen_frame_memory, &data_ready_ptr};
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaLaunchKernel((const void*)doca_receive_data_ready_kernel,
                         dim3(cuda_blocks), dim3(cuda_threads),
                         args, 0, stream),
        "Failed to launch doca_receive_data_ready_kernel");
    return DOCA_SUCCESS;
}

}  // extern C
