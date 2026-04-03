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
 */
#ifndef HOLOHUB_HSB_ROCE_RECEIVER_DOCA_GPUNETIO_OP
#define HOLOHUB_HSB_ROCE_RECEIVER_DOCA_GPUNETIO_OP

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>

#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include <hololink/core/data_channel.hpp>
#include <holoscan/core/gpu_resident_operator.hpp>

struct doca_dev;
struct doca_gpu;
struct doca_verbs_context;
struct doca_verbs_pd;
struct doca_uar;

namespace hololink::operators {

class DocaCq;
class DocaQp;

/**
 * GPU-resident operator for DOCA GPUNetIO RoCE reception.
 *
 * Used as the data-ready handler: its kernel runs inside the CUDA Graph
 * WHILE node, performs a non-blocking CQ check for an RDMA Write
 * completion, and returns CTRL_DATA_NOT_READY if none is available yet.
 * When a completion arrives it selects the received frame (writing its
 * address to chosen_frame_memory_) and marks CTRL_DATA_READY so the rest
 * of the pipeline proceeds.
 *
 * All DOCA resource creation (verbs context, CQ, QP, ring buffer, GPU
 * export) and HSB authentication/configuration happen in initialize().
 */
class DocaRoceReceiverOp : public holoscan::GPUResidentOperator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(DocaRoceReceiverOp,
                                         holoscan::GPUResidentOperator);

    DocaRoceReceiverOp() = default;
    ~DocaRoceReceiverOp() override;

    void setup(holoscan::OperatorSpec& spec) override;
    void initialize() override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext&, holoscan::OutputContext&,
                 holoscan::ExecutionContext&) override;

    /// Register a callback invoked during initialize() once chosen_frame_memory_
    /// has been allocated.  The callback receives the device pointer-to-pointer
    /// so the caller can publish it to the pipeline's frame source operator.
    void set_on_chosen_frame_memory_ready(std::function<void(unsigned char**)> callback);

 private:
    holoscan::Parameter<std::string> ibv_name_;
    holoscan::Parameter<uint32_t> ibv_port_;
    holoscan::Parameter<size_t> frame_size_;
    holoscan::Parameter<uint32_t> pages_;
    holoscan::Parameter<uint32_t> gpu_id_;
    holoscan::Parameter<DataChannel*> hololink_channel_;
    holoscan::Parameter<std::function<void()>> device_start_;
    holoscan::Parameter<std::function<void()>> device_stop_;

    // Called during initialize() after chosen_frame_memory_ is allocated
    std::function<void(unsigned char**)> on_chosen_frame_memory_ready_;

    // GPU pointer written by the data-ready kernel with the chosen frame addr
    unsigned char** chosen_frame_memory_ = nullptr;

    // DOCA resources
    ::ibv_pd* ibv_pd_ = nullptr;
    struct doca_dev* doca_device_ = nullptr;
    struct doca_gpu* doca_gpu_device_ = nullptr;
    struct doca_verbs_context* doca_verbs_ctx_ = nullptr;
    struct doca_verbs_pd* doca_pd_ = nullptr;
    struct doca_uar* uar_ = nullptr;
    DocaCq* doca_cq_rq_ = nullptr;
    DocaCq* doca_cq_sq_ = nullptr;
    DocaQp* doca_qp_ = nullptr;

    uint32_t qp_number_ = 0;
    uint32_t rkey_ = 0;

    // Per-thread persistent kernel state [out_ticket, wqe_idx] per thread
    uint64_t* rx_state_ = nullptr;

    bool umem_cpu_ = false;
    CUdevice cu_device_;
    CUcontext cu_context_;

    std::tuple<std::string, uint32_t> local_ip_and_port();
    static std::mutex& get_lock();
};

}  // namespace hololink::operators

#endif /* HOLOHUB_HSB_ROCE_RECEIVER_DOCA_GPUNETIO_OP */
