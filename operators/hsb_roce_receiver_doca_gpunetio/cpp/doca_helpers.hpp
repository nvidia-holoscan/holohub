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
 * Adapted from hololink gpu_roce_transceiver for use in holohub.
 */

#ifndef HOLOHUB_DOCA_GPUNETIO_HELPERS_HPP
#define HOLOHUB_DOCA_GPUNETIO_HELPERS_HPP

#include <arpa/inet.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#include <stddef.h>
#include <stdint.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <doca_dev.h>
#include <doca_error.h>
#include <doca_gpunetio.h>
#include <doca_rdma_bridge.h>
#include <doca_uar.h>
#include <doca_umem.h>
#include <doca_verbs.h>
#include <doca_verbs_bridge.h>

#define WQE_NUM 64
#define DOCA_UC_QP_RST2INIT_REQ_ATTR_MASK \
    (DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE | \
     DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM)
#define DOCA_UC_QP_INIT2RTR_REQ_ATTR_MASK \
    (DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN | \
     DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU | \
     DOCA_VERBS_QP_ATTR_AH_ATTR)
#define DOCA_UC_QP_RTR2RTS_REQ_ATTR_MASK \
    (DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN)

#define VERBS_TEST_DBR_SIZE (8)
#define ROUND_UP(unaligned_mapping_size, align_val) \
    (((unaligned_mapping_size) + (align_val)-1) & (~((align_val)-1)))

struct gpu_roce_ring_buffer {
    uint8_t* addr = nullptr;
    size_t stride_sz = 0;
    uint32_t stride_num = 0;
    struct ibv_mr* addr_mr = nullptr;
    int dmabuf_fd = 0;
    uint64_t* flag = nullptr;
};

inline size_t get_page_size(void) {
    long ret = sysconf(_SC_PAGESIZE);
    if (ret == -1) return 4096;
    return (size_t)ret;
}

namespace hololink::operators {

struct doca_verbs_context* open_doca_ib_device(char* name);

class DocaCq {
 public:
    DocaCq(uint32_t cqe_num, struct doca_gpu* gdev, struct doca_dev* ndev,
           struct doca_uar* uar, struct doca_verbs_context* vctx, bool umem_cpu);
    ~DocaCq();

    doca_error_t create();
    struct doca_verbs_cq* get() const { return cq; }

 private:
    struct doca_gpu* gdev;
    struct doca_dev* ndev;
    struct doca_uar* uar;
    struct doca_verbs_context* vctx;
    uint32_t cqe_num;
    struct doca_verbs_cq* cq = nullptr;
    void* umem_dev_ptr = nullptr;
    struct doca_umem* umem = nullptr;
    bool umem_cpu;
};

class DocaQp {
 public:
    DocaQp(uint32_t wqe_num, struct doca_gpu* gdev, struct doca_dev* ndev,
           struct doca_uar* uar, struct doca_verbs_context* vctx,
           struct doca_verbs_pd* vpd, struct doca_verbs_cq* cq_rq,
           struct doca_verbs_cq* cq_sq, bool umem_cpu);
    ~DocaQp();

    doca_error_t create(struct doca_verbs_context* verbs_ctx, size_t frame_size);
    doca_error_t create_ring(size_t stride_sz, unsigned stride_num, struct ibv_pd* ibv_pd);
    doca_error_t connect(struct doca_verbs_gid& doca_rgid, uint32_t gid_index,
                         uint32_t dest_qp_num);

    struct doca_verbs_qp* get() const { return qp; }
    struct doca_gpu_verbs_qp* get_gpu() const { return gpu_qp; }
    struct doca_gpu_dev_verbs_qp* get_gpu_dev() const { return gpu_dev_qp; }

    struct gpu_roce_ring_buffer gpu_rx_ring;

 private:
    struct doca_gpu* gdev;
    struct doca_dev* ndev;
    struct doca_uar* uar;
    struct doca_verbs_context* vctx;
    struct doca_verbs_pd* vpd;
    uint32_t wqe_num;
    struct doca_verbs_qp* qp = nullptr;
    void* umem_dev_ptr = nullptr;
    struct doca_umem* umem = nullptr;
    struct doca_umem* umem_dbr = nullptr;
    void* umem_dbr_dev_ptr = nullptr;
    struct doca_gpu_verbs_qp* gpu_qp = nullptr;
    struct doca_gpu_dev_verbs_qp* gpu_dev_qp = nullptr;
    struct doca_verbs_cq* cq_rq;
    struct doca_verbs_cq* cq_sq;
    bool umem_cpu;
};

// GPU kernel launchers
extern "C" {

doca_error_t DocaRoceReceiverPrepareKernel(
    cudaStream_t stream, struct doca_gpu_dev_verbs_qp* qp,
    size_t frame_size, uint32_t cuda_blocks, uint32_t cuda_threads);

doca_error_t DocaRoceReceiverDataReadyKernel(
    cudaStream_t stream, struct doca_gpu_dev_verbs_qp* qp,
    uint8_t* ring_buf, size_t ring_buf_stride_sz,
    uint32_t ring_buf_stride_num, uint64_t* rx_state,
    unsigned char** chosen_frame_memory,
    unsigned int* data_ready_ptr,
    uint32_t cuda_blocks, uint32_t cuda_threads);

}  // extern C

}  // namespace hololink::operators

#endif /* HOLOHUB_DOCA_GPUNETIO_HELPERS_HPP */
