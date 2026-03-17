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

#include <hsb_roce_receiver_doca_gpunetio/doca_roce_receiver_op.hpp>

#include <arpa/inet.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include <doca_gpunetio.h>
#include <doca_log.h>
#include <doca_rdma_bridge.h>
#include <doca_verbs.h>
#include <doca_verbs_bridge.h>
#include <infiniband/verbs.h>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/networking.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/utils/cuda_macros.hpp>

#include "doca_helpers.hpp"

static inline CUresult create_cuda_context(CUcontext* ctx, unsigned int flags, CUdevice dev) {
#if CUDA_VERSION >= 13000
    return cuCtxCreate(ctx, nullptr, flags, dev);
#else
    return cuCtxCreate(ctx, flags, dev);
#endif
}

#define YAML_CONVERTER(TYPE)                                                                \
    template <>                                                                             \
    struct YAML::convert<TYPE> {                                                            \
        static Node encode(TYPE&) { throw std::runtime_error("Unsupported"); }              \
                                                                                            \
        static bool decode(const Node&, TYPE&) { throw std::runtime_error("Unsupported"); } \
    };

YAML_CONVERTER(hololink::DataChannel*);
YAML_CONVERTER(std::function<void()>);

namespace hololink::operators {

// ============================= lifecycle ===================================

DocaRoceReceiverOp::~DocaRoceReceiverOp() {
    if (chosen_frame_memory_) cudaFree(chosen_frame_memory_);
    if (rx_state_ && doca_gpu_device_)
        doca_gpu_mem_free(doca_gpu_device_, rx_state_);

    delete doca_cq_rq_;
    delete doca_cq_sq_;
    delete doca_qp_;

    if (doca_gpu_device_) doca_gpu_destroy(doca_gpu_device_);
    if (uar_) doca_uar_destroy(uar_);
    if (doca_pd_) doca_verbs_pd_destroy(doca_pd_);
    if (doca_verbs_ctx_) doca_verbs_context_destroy(doca_verbs_ctx_);
    if (doca_device_) doca_dev_close(doca_device_);
    if (cu_context_)
      cuCtxDestroy(cu_context_);
}

void DocaRoceReceiverOp::setup(holoscan::OperatorSpec& spec) {
    spec.param(ibv_name_, "ibv_name", "IBVName", "IBV device name",
               std::string("roceP5p3s0f0"));
    spec.param(ibv_port_, "ibv_port", "IBVPort", "IBV port number", 1u);
    spec.param(frame_size_, "frame_size", "FrameSize", "Frame size in bytes",
               size_t(0));
    spec.param(pages_, "pages", "Pages", "Number of ring buffer pages", 2u);
    spec.param(gpu_id_, "gpu_id", "GPUID", "CUDA GPU device ordinal", 0u);

    register_converter<hololink::DataChannel*>();
    register_converter<std::function<void()>>();

    spec.param(hololink_channel_, "hololink_channel", "HololinkChannel",
               "DataChannel for HSB communication");
    spec.param(device_start_, "device_start", "DeviceStart",
               "Callback to start the sensor device");
    spec.param(device_stop_, "device_stop", "DeviceStop",
               "Callback to stop the sensor device");
}

void DocaRoceReceiverOp::set_on_chosen_frame_memory_ready(
    std::function<void(unsigned char**)> callback) {
    on_chosen_frame_memory_ready_ = std::move(callback);
}

// ======================== initialize (DOCA setup) ==========================

void DocaRoceReceiverOp::initialize() {
    holoscan::GPUResidentOperator::initialize();

    doca_error_t result;
    struct doca_log_backend* sdk_backend;

    std::lock_guard lock(get_lock());

    // chosen_frame_memory on GPU
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMalloc(&chosen_frame_memory_, sizeof(unsigned char*)),
        "Failed to allocate chosen_frame_memory");

    if (on_chosen_frame_memory_ready_) {
        on_chosen_frame_memory_ready_(chosen_frame_memory_);
    }

    // DOCA log
    doca_log_backend_create_with_file_sdk(stderr, &sdk_backend);
    doca_log_level_set_global_sdk_limit(DOCA_LOG_LEVEL_ERROR);

    // Open DOCA verbs device
    char* ibv_name_c = strdup(ibv_name_.get().c_str());
    doca_verbs_ctx_ = open_doca_ib_device(ibv_name_c);
    free(ibv_name_c);
    if (!doca_verbs_ctx_)
        throw std::runtime_error("Failed to open DOCA verbs device");
    HOLOSCAN_LOG_INFO("DOCA device opened: {}", ibv_name_.get());

    // GPU context
    cudaFree(0);
    cudaSetDevice(gpu_id_.get());
    CUresult cu_result;
    cu_result = cuDeviceGet(&cu_device_, gpu_id_.get());
    if (cu_result != CUDA_SUCCESS)
        throw std::runtime_error("Failed to get CUDA device " + std::to_string(gpu_id_.get()));
    cu_result = create_cuda_context(&cu_context_, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, cu_device_);
    if (cu_result != CUDA_SUCCESS)
        throw std::runtime_error("Failed to create CUDA context");
    cu_result = cuCtxPushCurrent(cu_context_);
    if (cu_result != CUDA_SUCCESS)
        throw std::runtime_error("Failed to push CUDA context");

    try {
      char gpu_bus_id[256];
      cudaDeviceGetPCIBusId(gpu_bus_id, 256, gpu_id_.get());
      struct cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, gpu_id_.get());
      umem_cpu_ = prop.integrated;
      HOLOSCAN_LOG_INFO("GPU {}: {} ({})", gpu_id_.get(), prop.name, umem_cpu_ ? "iGPU" : "dGPU");

      result = doca_gpu_create(gpu_bus_id, &doca_gpu_device_);
      if (result != DOCA_SUCCESS)
        throw std::runtime_error("Failed to create DOCA GPU device");

      // PD + DOCA device
      result = doca_verbs_pd_create(doca_verbs_ctx_, &doca_pd_);
      if (result != DOCA_SUCCESS)
        throw std::runtime_error("Failed to create DOCA PD");

      ibv_pd_ = doca_verbs_bridge_verbs_pd_get_ibv_pd(doca_pd_);
      if (!ibv_pd_)
        throw std::runtime_error("Failed to get ibv_pd");

      result = doca_rdma_bridge_open_dev_from_pd(ibv_pd_, &doca_device_);
      if (result != DOCA_SUCCESS)
        throw std::runtime_error("Failed to open DOCA device from PD");

      // Find RoCE v2 GID
      uint32_t gid_index = 0;
      bool gid_found = false;
      struct ibv_gid_entry ib_gid_entry = {};
      for (gid_index = 0;; gid_index++) {
        int ret = ibv_query_gid_ex(ibv_pd_->context, ibv_port_.get(), gid_index, &ib_gid_entry, 0);
        if (ret != 0 && errno != ENODATA)
          break;
        if (ib_gid_entry.gid_type == IBV_GID_TYPE_ROCE_V2 &&
            ib_gid_entry.gid.global.subnet_prefix == 0 &&
            (ib_gid_entry.gid.global.interface_id & 0xFFFFFFFF) == 0xFFFF0000) {
          gid_found = true;
          break;
        }
      }
      if (!gid_found)
        throw std::runtime_error("Cannot find RoCE v2 GID");
      HOLOSCAN_LOG_INFO("Found RoCE v2 GID at index {}", gid_index);

      // UAR
      result = doca_uar_create(doca_device_, DOCA_UAR_ALLOCATION_TYPE_NONCACHE, &uar_);
      if (result != DOCA_SUCCESS)
        throw std::runtime_error("Failed to create UAR");

      // CQs
      doca_cq_rq_ =
          new DocaCq(WQE_NUM, doca_gpu_device_, doca_device_, uar_, doca_verbs_ctx_, umem_cpu_);
      if (doca_cq_rq_->create() != DOCA_SUCCESS)
        throw std::runtime_error("Failed to create RQ CQ");

      doca_cq_sq_ =
          new DocaCq(WQE_NUM, doca_gpu_device_, doca_device_, uar_, doca_verbs_ctx_, umem_cpu_);
      if (doca_cq_sq_->create() != DOCA_SUCCESS)
        throw std::runtime_error("Failed to create SQ CQ");

      // QP + ring buffer
      doca_qp_ = new DocaQp(WQE_NUM,
                            doca_gpu_device_,
                            doca_device_,
                            uar_,
                            doca_verbs_ctx_,
                            doca_pd_,
                            doca_cq_rq_->get(),
                            doca_cq_sq_->get(),
                            umem_cpu_);

      const size_t host_page_size = get_page_size();
      size_t page_size = hololink::core::round_up(frame_size_.get(), host_page_size);

      if (doca_qp_->create(doca_verbs_ctx_, frame_size_.get()) != DOCA_SUCCESS)
        throw std::runtime_error("Failed to create DOCA QP");

      if (doca_qp_->create_ring(page_size, pages_.get(), ibv_pd_) != DOCA_SUCCESS)
        throw std::runtime_error("Failed to create DOCA ring buffer");

      rkey_ = doca_qp_->gpu_rx_ring.addr_mr->rkey;
      qp_number_ = doca_verbs_qp_get_qpn(doca_qp_->get());
      HOLOSCAN_LOG_INFO("DOCA QP number: {:#x}, rkey: {}", qp_number_, rkey_);

      // Connect QP to peer (HSB)
      const std::string& peer_ip = hololink_channel_.get()->peer_ip();
      unsigned long client_ip = 0;
      if (inet_pton(AF_INET, peer_ip.c_str(), &client_ip) != 1)
        throw std::runtime_error("Invalid peer IP: " + peer_ip);

      uint64_t client_iid = ((uint64_t)client_ip << 32) | 0xFFFF0000ULL;
      union ibv_gid rgid = {.global = {.subnet_prefix = 0, .interface_id = client_iid}};
      struct doca_verbs_gid doca_rgid;
      memcpy(doca_rgid.raw, rgid.raw, 16);

      if (doca_qp_->connect(doca_rgid, gid_index, 0x2) != DOCA_SUCCESS)
        throw std::runtime_error("Failed to connect DOCA QP");

      // Pre-post receive WQEs
      DocaRoceReceiverPrepareKernel(0, doca_qp_->get_gpu_dev(), frame_size_.get(), 1, WQE_NUM);
      cudaStreamSynchronize(0);

      // Persistent state for single-thread CQ poller: [out_ticket, wqe_idx]
      size_t state_bytes = 2 * sizeof(uint64_t);
      result = doca_gpu_mem_alloc(doca_gpu_device_,
                                  state_bytes,
                                  get_page_size(),
                                  DOCA_GPU_MEM_TYPE_GPU,
                                  (void**)&rx_state_,
                                  nullptr);
      if (result != DOCA_SUCCESS)
        throw std::runtime_error("Failed to allocate rx_state");

      std::vector<uint64_t> init_state = {0, 0};
    cudaMemcpy(rx_state_, init_state.data(), state_bytes,
               cudaMemcpyHostToDevice);

      // HSB authentication & RoCE configuration
      hololink_channel_.get()->authenticate(qp_number_, rkey_);

      auto [local_ip, local_port] = local_ip_and_port();
      HOLOSCAN_LOG_INFO("local_ip={} local_port={}", local_ip, local_port);

      // The third argument to configure_roce is DP_PAGE_INC — the stride
      // the HSB uses between successive pages.  It must match the DOCA ring
      // buffer's stride_sz (= page_size = round_up(frame_size, PAGE_SIZE)).
      hololink_channel_.get()->configure_roce(
          0, frame_size_.get(), page_size, pages_.get(), local_port);

    } catch (...) {
      cuCtxPopCurrent(&cu_context_);
      throw;
    }
    cuCtxPopCurrent(&cu_context_);
    HOLOSCAN_LOG_INFO("DOCA RoCE receiver initialised");
}

// ============================== start / stop ===============================

void DocaRoceReceiverOp::start() {
    holoscan::GPUResidentOperator::start();
    if (device_start_.get()) device_start_.get()();
    HOLOSCAN_LOG_INFO("DOCA RoCE receiver started");
}

void DocaRoceReceiverOp::stop() {
    if (device_stop_.get()) device_stop_.get()();
    hololink_channel_.get()->unconfigure();
    holoscan::GPUResidentOperator::stop();
    HOLOSCAN_LOG_INFO("DOCA RoCE receiver stopped");
}

// ============================== compute ====================================

void DocaRoceReceiverOp::compute(holoscan::InputContext&,
                                 holoscan::OutputContext&,
                                 holoscan::ExecutionContext&) {
    auto* data_ready_addr =
        static_cast<unsigned int*>(data_ready_device_address());
    if (!data_ready_addr || !doca_qp_)
        throw std::runtime_error(
            "DocaRoceReceiverOp::compute() preconditions not met");

    auto stream_ptr = data_ready_handler_cuda_stream();
    if (!stream_ptr)
        throw std::runtime_error("Data ready handler CUDA stream is null");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(*stream_ptr);

    // CQ poll kernel: blocks until one CQE, writes chosen frame addr
    // to chosen_frame_memory_ and sets data_ready flag.
    doca_error_t result = DocaRoceReceiverDataReadyKernel(
        stream,
        doca_qp_->get_gpu_dev(),
        doca_qp_->gpu_rx_ring.addr,
        doca_qp_->gpu_rx_ring.stride_sz,
        doca_qp_->gpu_rx_ring.stride_num,
        rx_state_,
        chosen_frame_memory_,
        data_ready_addr,
        1, 1);
    if (result != DOCA_SUCCESS) {
        throw std::runtime_error(
            "Failed to launch DocaRoceReceiverDataReadyKernel: " +
            std::string(doca_error_get_descr(result)) + " (DOCA error code " +
            std::to_string(static_cast<int>(result)) + ")");
    }

    cudaError_t cuda_result = cudaPeekAtLastError();
    if (cuda_result != cudaSuccess) {
        throw std::runtime_error(
            std::string("DocaRoceReceiverDataReadyKernel launch failed: ") +
            cudaGetErrorString(cuda_result));
    }
}

// ===========================================================================

std::tuple<std::string, uint32_t> DocaRoceReceiverOp::local_ip_and_port() {
    struct ibv_gid_entry ib_gid_entry = {};
    for (uint32_t idx = 0;; idx++) {
        int ret = ibv_query_gid_ex(ibv_pd_->context, ibv_port_.get(),
                                   idx, &ib_gid_entry, 0);
        if (ret != 0 && errno != ENODATA) break;
        if (ib_gid_entry.gid_type == IBV_GID_TYPE_ROCE_V2 &&
            ib_gid_entry.gid.global.subnet_prefix == 0 &&
            (ib_gid_entry.gid.global.interface_id & 0xFFFFFFFF) == 0xFFFF0000) {
            uint32_t ip_nbo = static_cast<uint32_t>(
                ib_gid_entry.gid.global.interface_id >> 32);
            struct in_addr addr;
            addr.s_addr = ip_nbo;
            constexpr uint32_t roce_port = 4791;
            return {inet_ntoa(addr), roce_port};
        }
    }
    throw std::runtime_error("Cannot find RoCE v2 GID for local_ip_and_port");
}

std::mutex& DocaRoceReceiverOp::get_lock() {
    static std::mutex instance;
    return instance;
}

}  // namespace hololink::operators
