/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <stdio.h>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>

#include "adv_network_doca_kernels.h"

#define ETHER_ADDR_LEN 6
#define DOCA_DEBUG_KERNEL 0

#define BYTE_SWAP16(v) \
  ((((uint16_t)(v)&UINT16_C(0x00ff)) << 8) | (((uint16_t)(v)&UINT16_C(0xff00)) >> 8))

struct ether_hdr {
  uint8_t d_addr_bytes[ETHER_ADDR_LEN]; /* Destination addr bytes in tx order */
  uint8_t s_addr_bytes[ETHER_ADDR_LEN]; /* Source addr bytes in tx order */
  uint16_t ether_type;                  /* Frame type */
} __attribute__((__packed__));

struct ipv4_hdr {
  uint8_t version_ihl;      /* version and header length */
  uint8_t type_of_service;  /* type of service */
  uint16_t total_length;    /* length of packet */
  uint16_t packet_id;       /* packet ID */
  uint16_t fragment_offset; /* fragmentation offset */
  uint8_t time_to_live;     /* time to live */
  uint8_t next_proto_id;    /* protocol ID */
  uint16_t hdr_checksum;    /* header checksum */
  uint32_t src_addr;        /* source address */
  uint32_t dst_addr;        /* destination address */
} __attribute__((__packed__));

struct udp_hdr {
  uint16_t src_port;    /* UDP source port */
  uint16_t dst_port;    /* UDP destination port */
  uint16_t dgram_len;   /* UDP datagram length */
  uint16_t dgram_cksum; /* UDP datagram checksum */
} __attribute__((__packed__));

struct eth_ip_udp_hdr {
  struct ether_hdr l2_hdr; /* Ethernet header */
  struct ipv4_hdr l3_hdr;  /* IP header */
  struct udp_hdr l4_hdr;   /* UDP header */
} __attribute__((__packed__));

__device__ __inline__ int raw_to_udp(const uintptr_t buf_addr, struct eth_ip_udp_hdr** hdr,
                                     uint8_t** payload) {
  (*hdr) = (struct eth_ip_udp_hdr*)buf_addr;
  (*payload) = (uint8_t*)(buf_addr + sizeof(struct eth_ip_udp_hdr));

  return 0;
}

/**
 * @brief Receiver packet kernel to where each CUDA Block receives on a different queue.
 * Works in persistent mode.
 *
 * @param out Output buffer
 * @param in Pointer to list of input packet pointers
 * @param pkt_len Length of each packet. All packets must be same length for this example
 * @param num_pkts Number of packets
 */
__global__ void receive_packets_kernel_persistent(int rxqn, uintptr_t* eth_rxq_gpu,
                                                  uintptr_t* pkt_gpu_list, uint32_t* pkt_idx_list,
                                                  const uint32_t* batch_list, uint32_t* exit_cond) {
  doca_error_t ret;
  uintptr_t buf_addr;
  uint64_t buf_idx;
  struct doca_gpu_eth_rxq* rxq;
  int stats_idx = 0;
  struct eth_ip_udp_hdr* hdr;
  uint8_t* payload;

  __shared__ uint64_t out_first_pkt_idx;
  __shared__ uint32_t out_pkt_num;
  __shared__ uint32_t rx_pkt_bytes;
  uint32_t tot_pkts_batch = 0;
  struct adv_doca_rx_gpu_info* packets_stats =
      ((struct adv_doca_rx_gpu_info*)pkt_gpu_list[blockIdx.x]);

  // Warmup
  if (eth_rxq_gpu == NULL)
    return;

  rxq = (struct doca_gpu_eth_rxq*)eth_rxq_gpu[blockIdx.x];

  if (threadIdx.x == 0)
    DOCA_GPUNETIO_VOLATILE(rx_pkt_bytes) = 0;
  __syncthreads();

  do {
    ret =
        doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK, DOCA_GPUNETIO_ETH_MCST_AUTO>(
            rxq,
            batch_list[blockIdx.x],
            CUDA_MAX_RX_TIMEOUT_NS,
            &out_first_pkt_idx,
            &out_pkt_num,
            NULL);
    /* If any thread returns receive error, the whole execution stops */
    if (ret != DOCA_SUCCESS) {
      if (threadIdx.x == 0) {
/*
 * printf in CUDA kernel may be a good idea only to report critical errors or debugging.
 * If application prints this message on the console, something bad happened and
 * applications needs to exit
 */
#if DOCA_DEBUG_KERNEL == 1
        printf("Receive UDP kernel error %d rxpkts %d error %d\n", ret, out_pkt_num, ret);
#endif
        DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
      }
      break;
    }

    if (out_pkt_num == 0)
      continue;

    buf_idx = threadIdx.x;
    while (buf_idx < out_pkt_num) {
      buf_addr = doca_gpu_dev_eth_rxq_get_pkt_addr(rxq, out_first_pkt_idx + buf_idx);
      raw_to_udp(buf_addr, &hdr, &payload);
#if DOCA_DEBUG_KERNEL == 1
      printf(
          "Queue %d Thread %d received UDP packet with "
          "Eth src %02x:%02x:%02x:%02x:%02x:%02x - "
          "Eth dst %02x:%02x:%02x:%02x:%02x:%02x - "
          "Src port %d - Dst port %d\n",
          blockIdx.x,
          threadIdx.x,
          ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[0],
          ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[1],
          ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[2],
          ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[3],
          ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[4],
          ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[5],
          ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[0],
          ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[1],
          ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[2],
          ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[3],
          ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[4],
          ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[5],
          BYTE_SWAP16(hdr->l4_hdr.src_port),
          BYTE_SWAP16(hdr->l4_hdr.dst_port));
#endif

      /* Add packet processing/filtering function here. */

      if (threadIdx.x == 0 && buf_idx == 0) {
        DOCA_GPUNETIO_VOLATILE(packets_stats[stats_idx].gpu_pkt0_addr) = buf_addr;
        DOCA_GPUNETIO_VOLATILE(packets_stats[stats_idx].gpu_pkt0_idx) = out_first_pkt_idx;
      }

      // Eth + IP + (UDP + Payload)
      atomicAdd_block(
          &rx_pkt_bytes,
          DOCA_GPUNETIO_ETH_BSWAP16(hdr->l3_hdr.total_length) + sizeof(struct ether_hdr));
      buf_idx += blockDim.x;
    }
    __syncthreads();

    if (threadIdx.x == 0 && out_pkt_num > 0) {
      tot_pkts_batch += out_pkt_num;
#if DOCA_DEBUG_KERNEL == 1
      printf("Queue %d tot pkts %d/%d stats_idx %d\n",
             blockIdx.x,
             tot_pkts_batch,
             batch_list[blockIdx.x],
             stats_idx);
#endif

      if (tot_pkts_batch >= batch_list[blockIdx.x]) {
        DOCA_GPUNETIO_VOLATILE(packets_stats[stats_idx].num_pkts) =
            DOCA_GPUNETIO_VOLATILE(tot_pkts_batch);
        DOCA_GPUNETIO_VOLATILE(packets_stats[stats_idx].nbytes) =
            DOCA_GPUNETIO_VOLATILE(rx_pkt_bytes);
        // Replace the __threadfence_system();
        doca_gpu_dev_eth_fence_release<DOCA_GPUNETIO_ETH_SYNC_SCOPE_SYS>();
        DOCA_GPUNETIO_VOLATILE(packets_stats[stats_idx].status) = DOCA_GPU_SEMAPHORE_STATUS_READY;
        stats_idx = (stats_idx + 1) % MAX_DEFAULT_SEM_X_QUEUE;
        DOCA_GPUNETIO_VOLATILE(rx_pkt_bytes) = 0;
        tot_pkts_batch = 0;
      }
    }
    __syncthreads();
  } while (DOCA_GPUNETIO_VOLATILE(*exit_cond) == 0);

  __syncthreads();

  /* Flush remaining packets in the last partial batch */
  if (threadIdx.x == 0) {
    DOCA_GPUNETIO_VOLATILE(packets_stats[stats_idx].num_pkts) =
        DOCA_GPUNETIO_VOLATILE(tot_pkts_batch);
    DOCA_GPUNETIO_VOLATILE(packets_stats[stats_idx].nbytes) = DOCA_GPUNETIO_VOLATILE(rx_pkt_bytes);
    // Replace the __threadfence_system();
    doca_gpu_dev_eth_fence_release<DOCA_GPUNETIO_ETH_SYNC_SCOPE_SYS>();
    DOCA_GPUNETIO_VOLATILE(packets_stats[stats_idx].status) = DOCA_GPU_SEMAPHORE_STATUS_READY;
  }
}

/**
 * @brief Receiver packet kernel to where each CUDA Block receives on a different queue.
 * Works in non-persistent mode.
 *
 * @param out Output buffer
 * @param in Pointer to list of input packet pointers
 * @param pkt_len Length of each packet. All packets must be same length for this example
 * @param num_pkts Number of packets
 */
__global__ void receive_packets_kernel_non_persistent(int rxqn, uintptr_t* eth_rxq_gpu,
                                                      uintptr_t* pkt_gpu_list,
                                                      uint32_t* pkt_idx_list,
                                                      const uint32_t* batch_list) {
  doca_error_t ret;
  uintptr_t buf_addr;
  uint64_t buf_idx;
  struct doca_gpu_eth_rxq* rxq;
  struct eth_ip_udp_hdr* hdr;
  uint8_t* payload;

  __shared__ uint64_t out_first_pkt_idx;
  __shared__ uint32_t out_pkt_num;
  __shared__ uint32_t rx_pkt_bytes;
  struct adv_doca_rx_gpu_info* packets_stats =
      ((struct adv_doca_rx_gpu_info*)pkt_gpu_list[blockIdx.x]) + pkt_idx_list[blockIdx.x];

  // Warmup
  if (eth_rxq_gpu == NULL)
    return;

  rxq = (struct doca_gpu_eth_rxq*)eth_rxq_gpu[blockIdx.x];

  if (threadIdx.x == 0)
    DOCA_GPUNETIO_VOLATILE(rx_pkt_bytes) = 0;
  __syncthreads();

  ret = doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK, DOCA_GPUNETIO_ETH_MCST_AUTO>(
      rxq, batch_list[blockIdx.x], CUDA_MAX_RX_TIMEOUT_NS, &out_first_pkt_idx, &out_pkt_num, NULL);
  /* If any thread returns receive error, the whole execution stops */
  if (ret != DOCA_SUCCESS) {
    out_pkt_num = 0;
    if (threadIdx.x == 0) {
/*
 * printf in CUDA kernel may be a good idea only to report critical errors or debugging.
 * If application prints this message on the console, something bad happened and
 * applications needs to exit
 */
#if DOCA_DEBUG_KERNEL == 1
      printf("Receive UDP kernel error %d rxpkts %d error %d\n", ret, out_pkt_num, ret);
#endif
    }
  }

  buf_idx = threadIdx.x;
  while (buf_idx < out_pkt_num) {
    buf_addr = doca_gpu_dev_eth_rxq_get_pkt_addr(rxq, out_first_pkt_idx + buf_idx);
    raw_to_udp(buf_addr, &hdr, &payload);

#if DOCA_DEBUG_KERNEL == 1
    printf(
        "Queue %d Thread %d received UDP packet with "
        "Eth src %02x:%02x:%02x:%02x:%02x:%02x - "
        "Eth dst %02x:%02x:%02x:%02x:%02x:%02x - "
        "Src port %d - Dst port %d\n",
        blockIdx.x,
        threadIdx.x,
        ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[0],
        ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[1],
        ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[2],
        ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[3],
        ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[4],
        ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[5],
        ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[0],
        ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[1],
        ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[2],
        ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[3],
        ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[4],
        ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[5],
        BYTE_SWAP16(hdr->l4_hdr.src_port),
        BYTE_SWAP16(hdr->l4_hdr.dst_port));
#endif

    /* Add custom packet processing/filtering function here. */

    if (threadIdx.x == 0 && buf_idx == 0) {
      DOCA_GPUNETIO_VOLATILE(packets_stats->gpu_pkt0_addr) = buf_addr;
      DOCA_GPUNETIO_VOLATILE(packets_stats->gpu_pkt0_idx) = out_first_pkt_idx;
    }

    // Eth + IP + (UDP + Payload)
    atomicAdd_block(&rx_pkt_bytes,
                    DOCA_GPUNETIO_ETH_BSWAP16(hdr->l3_hdr.total_length) + sizeof(struct ether_hdr));

    buf_idx += blockDim.x;
  }
  __syncthreads();

  if (threadIdx.x == 0 && out_pkt_num > 0) {
#if DOCA_DEBUG_KERNEL == 1
    printf("Queue %d tot pkts %d/%d pkt_idx_list[blockIdx.x] %d\n",
           blockIdx.x,
           out_pkt_num,
           batch_list[blockIdx.x],
           pkt_idx_list[blockIdx.x]);
#endif
    DOCA_GPUNETIO_VOLATILE(packets_stats->num_pkts) = DOCA_GPUNETIO_VOLATILE(out_pkt_num);
    DOCA_GPUNETIO_VOLATILE(packets_stats->nbytes) = DOCA_GPUNETIO_VOLATILE(rx_pkt_bytes);
    // Replace the __threadfence_system();
    doca_gpu_dev_eth_fence_release<DOCA_GPUNETIO_ETH_SYNC_SCOPE_SYS>();
    DOCA_GPUNETIO_VOLATILE(packets_stats->status) = DOCA_GPU_SEMAPHORE_STATUS_READY;
  }
}

/**
 * @brief Receiver packet kernel to where each CUDA Block receives on a different queue.
 *
 * @param out Output buffer
 * @param in Pointer to list of input packet pointers
 * @param pkt_len Length of each packet. All packets must be same length for this example
 * @param num_pkts Number of packets
 */
__global__ void send_packets_kernel(struct doca_gpu_eth_txq* txq, uint64_t pkt_buff_addr,
                                    const uint32_t pkt_buff_mkey, uint32_t gpu_pkt0_idx,
                                    const size_t num_pkts, uint32_t max_pkts,
                                    const uint64_t max_pkt_size, uint32_t* gpu_pkts_len,
                                    const bool set_completion) {
  uint32_t pkt_idx = threadIdx.x;
  struct doca_gpu_dev_eth_txq_wqe* wqe_ptr;
  __shared__ uint64_t base_wqe_idx;

#if DOCA_DEBUG_KERNEL == 2
  struct eth_ip_udp_hdr* hdr;
  uint8_t* payload;
#endif

  // Warmup
  if (num_pkts == 0)
    return;

  if (threadIdx.x == 0)
    base_wqe_idx = doca_gpu_dev_eth_txq_reserve_wq_slots(txq, num_pkts);
  __syncthreads();

  while (pkt_idx < num_pkts) {
    uint64_t addr = ((uint64_t)pkt_buff_addr) +
                    (max_pkt_size * (uint64_t)(((pkt_idx + gpu_pkt0_idx) % max_pkts)));
    wqe_ptr = doca_gpu_dev_eth_txq_get_wqe_ptr(txq, base_wqe_idx + pkt_idx);

    if (set_completion && pkt_idx == (num_pkts - 1))
      doca_gpu_dev_eth_txq_wqe_prepare_send(txq,
                                            wqe_ptr,
                                            base_wqe_idx + pkt_idx,
                                            addr,
                                            pkt_buff_mkey,
                                            gpu_pkts_len[pkt_idx],
                                            DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY);
    else
      doca_gpu_dev_eth_txq_wqe_prepare_send(txq,
                                            wqe_ptr,
                                            base_wqe_idx + pkt_idx,
                                            addr,
                                            pkt_buff_mkey,
                                            gpu_pkts_len[pkt_idx],
                                            DOCA_GPUNETIO_ETH_SEND_FLAG_NONE);

#if DOCA_DEBUG_KERNEL == 2
    raw_to_udp(addr, &hdr, &payload);
    printf(
        "Queue %d Thread %d num_pkts %d base_wqe_idx %ld wqeidx %ld received UDP packet %lx len %d "
        "with "
        "Eth src %02x:%02x:%02x:%02x:%02x:%02x - "
        "Eth dst %02x:%02x:%02x:%02x:%02x:%02x - "
        "Src port %d - Dst port %d\n",
        blockIdx.x,
        threadIdx.x,
        (int)num_pkts,
        base_wqe_idx,
        base_wqe_idx + pkt_idx,
        addr,
        gpu_pkts_len[pkt_idx],
        ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[0],
        ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[1],
        ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[2],
        ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[3],
        ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[4],
        ((uint8_t*)hdr->l2_hdr.s_addr_bytes)[5],
        ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[0],
        ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[1],
        ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[2],
        ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[3],
        ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[4],
        ((uint8_t*)hdr->l2_hdr.d_addr_bytes)[5],
        hdr->l4_hdr.src_port,
        hdr->l4_hdr.dst_port);
#endif

    pkt_idx = (pkt_idx + blockDim.x);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
#if DOCA_DEBUG_KERNEL == 2
    printf(
        "Submit base_wqe_idx %ld num_pkts %d sq %x from gpu_pkt0_idx %d max_pkts %d set_completion "
        "%d\n",
        base_wqe_idx,
        (int)num_pkts,
        txq->sq_num_shift8,
        gpu_pkt0_idx,
        max_pkts,
        set_completion);
#endif

    doca_gpu_dev_eth_txq_submit(txq, base_wqe_idx + num_pkts);
  }
  __syncthreads();
}

extern "C" {

doca_error_t doca_receiver_packet_kernel(cudaStream_t stream, int rxqn, uintptr_t* eth_rxq_gpu,
                                         uintptr_t* pkt_gpu_list, uint32_t* pkt_idx_list,
                                         uint32_t* batch_list, uint32_t* gpu_exit_condition,
                                         bool persistent) {
  cudaError_t result = cudaSuccess;

  if (rxqn == 0 || gpu_exit_condition == NULL) {
    HOLOSCAN_LOG_ERROR("kernel_receive_packets invalid input values");
    return DOCA_ERROR_INVALID_VALUE;
  }

  /* Check no previous CUDA errors */
  result = cudaGetLastError();
  if (cudaSuccess != result) {
    HOLOSCAN_LOG_ERROR(
        "[{}:{}] cuda failed with {} \n", __FILE__, __LINE__, cudaGetErrorString(result));
    return DOCA_ERROR_BAD_STATE;
  }

  /* For simplicity launch 1 CUDA block with 32 CUDA threads */
  if (persistent)
    receive_packets_kernel_persistent<<<rxqn, CUDA_BLOCK_THREADS, 0, stream>>>(
        rxqn, eth_rxq_gpu, pkt_gpu_list, pkt_idx_list, batch_list, gpu_exit_condition);
  else
    receive_packets_kernel_non_persistent<<<rxqn, CUDA_BLOCK_THREADS, 0, stream>>>(
        rxqn, eth_rxq_gpu, pkt_gpu_list, pkt_idx_list, batch_list);

  result = cudaGetLastError();
  if (cudaSuccess != result) {
    HOLOSCAN_LOG_ERROR(
        "[{}:{}] cuda failed with {} \n", __FILE__, __LINE__, cudaGetErrorString(result));
    return DOCA_ERROR_BAD_STATE;
  }

  return DOCA_SUCCESS;
}

doca_error_t doca_sender_packet_kernel(cudaStream_t stream, struct doca_gpu_eth_txq* txq,
                                       uint64_t pkt_buff_addr, const uint32_t pkt_buff_mkey,
                                       uint32_t gpu_pkt0_idx, const size_t num_pkts,
                                       uint32_t max_pkts, const uint64_t max_pkt_size,
                                       uint32_t* gpu_pkts_len, bool set_completion) {
  cudaError_t result = cudaSuccess;

  if (txq == NULL) {
    HOLOSCAN_LOG_ERROR("kernel_receive_packets invalid input values");
    return DOCA_ERROR_INVALID_VALUE;
  }

  /* Check no previous CUDA errors */
  result = cudaGetLastError();
  if (cudaSuccess != result) {
    HOLOSCAN_LOG_ERROR(
        "[{}:{}] cuda failed with {} \n", __FILE__, __LINE__, cudaGetErrorString(result));
    return DOCA_ERROR_BAD_STATE;
  }

  /* For simplicity launch 1 CUDA block with 32 CUDA threads */
  send_packets_kernel<<<1, CUDA_BLOCK_THREADS, 0, stream>>>(txq,
                                                            pkt_buff_addr,
                                                            pkt_buff_mkey,
                                                            gpu_pkt0_idx,
                                                            num_pkts,
                                                            max_pkts,
                                                            max_pkt_size,
                                                            gpu_pkts_len,
                                                            set_completion);
  result = cudaGetLastError();
  if (cudaSuccess != result) {
    HOLOSCAN_LOG_ERROR(
        "[{}:{}] cuda failed with {} \n", __FILE__, __LINE__, cudaGetErrorString(result));
    return DOCA_ERROR_BAD_STATE;
  }

  return DOCA_SUCCESS;
}

} /* extern C */
