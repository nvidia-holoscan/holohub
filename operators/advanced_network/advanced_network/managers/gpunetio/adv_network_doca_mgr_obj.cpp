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

#include <atomic>
#include <cmath>
#include <complex>
#include <chrono>
#include <iostream>
#include <map>
#include <set>
#include <unistd.h>
#include <sys/time.h>
#include "adv_network_doca_mgr.h"
#include "holoscan/holoscan.hpp"

using namespace std::chrono;

/*
 * Retrieve host page size
 *
 * @return: host page size
 */
static size_t get_host_page_size(void)
{
	long ret = sysconf(_SC_PAGESIZE);
	if (ret == -1)
		return 4096; // 4KB, default Linux page size

	return (size_t)ret;
}

namespace holoscan::advanced_network {

DocaRxQueue::DocaRxQueue(struct doca_dev* dev_, struct doca_gpu* gdev_,
                         struct doca_flow_port* df_port_, uint16_t qid_, int max_pkt_num_,
                         int max_pkt_size_, enum doca_gpu_mem_type mtype_)
    : ddev(dev_),
      gdev(gdev_),
      df_port(df_port_),
      qid(qid_),
      max_pkt_num(max_pkt_num_),
      max_pkt_size(max_pkt_size_),
      mtype(mtype_) {
  HOLOSCAN_LOG_INFO(
      "Creating UDP Eth Rxq {} max_pkt_size {} max_pkt_num {}", qid, max_pkt_size, max_pkt_num);

  doca_error_t result;
  uint32_t cyclic_buffer_size = 0;

  result = doca_eth_rxq_create(ddev, max_pkt_num, max_pkt_size, &(eth_rxq_cpu));
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed doca_eth_rxq_create: {}", doca_error_get_descr(result));
  }

  result = doca_eth_rxq_set_type(eth_rxq_cpu, DOCA_ETH_RXQ_TYPE_CYCLIC);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed doca_eth_rxq_set_type: {}", doca_error_get_descr(result));
  }

  result = doca_eth_rxq_estimate_packet_buf_size(DOCA_ETH_RXQ_TYPE_CYCLIC,
						       0, 0,
						       max_pkt_size,
						       max_pkt_num,
						       0, 0, 0,
						       &cyclic_buffer_size);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to get eth_rxq cyclic buffer size: {}",
                          doca_error_get_descr(result));
  }

  result = doca_mmap_create(&pkt_buff_mmap);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to create mmap: {}", doca_error_get_descr(result));
  }

  result = doca_mmap_add_dev(pkt_buff_mmap, ddev);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to add dev to mmap: {}", doca_error_get_descr(result));
  }

  ALIGN_SIZE(cyclic_buffer_size, get_host_page_size());

  result = doca_gpu_mem_alloc(
      gdev, cyclic_buffer_size, GPU_PAGE_SIZE, mtype, &gpu_pkt_addr, &cpu_pkt_addr);
  if (result != DOCA_SUCCESS || gpu_pkt_addr == nullptr) {
    HOLOSCAN_LOG_CRITICAL("Failed to allocate gpu memory {}", doca_error_get_descr(result));
  }

  dmabuf_fd = -1;
  if (mtype == DOCA_GPU_MEM_TYPE_GPU) {
    /* Map GPU memory buffer used to recv packets with DMABuf */
	  result = doca_gpu_dmabuf_fd(gdev, gpu_pkt_addr, cyclic_buffer_size, &(dmabuf_fd));
    if (result == DOCA_SUCCESS) {
      		HOLOSCAN_LOG_INFO("Mapping recv queue buffer ({} size {}B dmabuf fd {}) with dmabuf mode",
			      gpu_pkt_addr,
			      cyclic_buffer_size,
			      dmabuf_fd);

      result = doca_mmap_set_dmabuf_memrange(pkt_buff_mmap,
                    dmabuf_fd,
                    gpu_pkt_addr,
                    0,
                    cyclic_buffer_size);
      if (result != DOCA_SUCCESS) {
        HOLOSCAN_LOG_ERROR("Failed to set dmabuf memrange for mmap {}", doca_error_get_descr(result));
        dmabuf_fd = -1;
      }
    } else
      dmabuf_fd = -1;
  }

	if (dmabuf_fd == -1) {
		HOLOSCAN_LOG_INFO("Mapping recv queue buffer ({} size {}B) with nvidia-peermem mode",
			      gpu_pkt_addr,
			      cyclic_buffer_size);

		/* If failed, use nvidia-peermem legacy method */
		result = doca_mmap_set_memrange(pkt_buff_mmap, gpu_pkt_addr, cyclic_buffer_size);
		if (result != DOCA_SUCCESS) {
			HOLOSCAN_LOG_ERROR("Failed to set memrange for mmap {}", doca_error_get_descr(result));
		}
	}

  result = doca_mmap_set_permissions(
      pkt_buff_mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_PCI_RELAXED_ORDERING);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to set permissions for mmap {}", doca_error_get_descr(result));
  }

  result = doca_mmap_start(pkt_buff_mmap);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to start mmap {}", doca_error_get_descr(result));
  }

  result = doca_eth_rxq_set_pkt_buf(eth_rxq_cpu, pkt_buff_mmap, 0, cyclic_buffer_size);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to set cyclic buffer  {}", doca_error_get_descr(result));
  }

  eth_rxq_ctx = doca_eth_rxq_as_doca_ctx(eth_rxq_cpu);
  if (eth_rxq_ctx == nullptr) {
    HOLOSCAN_LOG_CRITICAL("Failed doca_eth_rxq_as_doca_ctx: {}", doca_error_get_descr(result));
  }

  result = doca_ctx_set_datapath_on_gpu(eth_rxq_ctx, gdev);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed doca_ctx_set_datapath_on_gpu: {}", doca_error_get_descr(result));
  }

  result = doca_ctx_start(eth_rxq_ctx);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed doca_ctx_start: {}", doca_error_get_descr(result));
  }

  result = doca_eth_rxq_get_gpu_handle(eth_rxq_cpu, &(eth_rxq_gpu));
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed doca_eth_rxq_get_gpu_handle: {}", doca_error_get_descr(result));
  }

  rxq_pipe = nullptr;
  root_udp_entry = nullptr;
}

DocaRxQueue::~DocaRxQueue() {
  doca_error_t result;

  result = doca_ctx_stop(eth_rxq_ctx);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed doca_ctx_stop: {}", doca_error_get_descr(result));
  }

  result = doca_eth_rxq_destroy(eth_rxq_cpu);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed doca_eth_rxq_destroy: {}", doca_error_get_descr(result));
  }

  result = doca_mmap_destroy(pkt_buff_mmap);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to destroy mmap: {}", doca_error_get_descr(result));
  }

  result = doca_gpu_mem_free(gdev, gpu_pkt_addr);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to free gpu memory: {}", doca_error_get_descr(result));
  }

  result = destroy_rx_packet_list();
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to destroy_rx_packet_list: {}", doca_error_get_descr(result));
  }
}

doca_error_t DocaRxQueue::create_udp_pipe(const FlowConfig& cfg,
                                          struct doca_flow_pipe* rxq_pipe_default,
                                          uint16_t &flow_queue_id) {
  doca_error_t result;
  struct doca_flow_match match = {0};
  struct doca_flow_fwd fwd = {};
  struct doca_flow_fwd miss_fwd = {};
  struct doca_flow_pipe_cfg* pipe_cfg;
  struct doca_flow_pipe_entry* entry;
  uint16_t rss_queues[1]; /* stick to 1 queue per flow now, no RSS */
  struct doca_flow_monitor monitor = {
      .counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
  };
  const char* pipe_name = "GPU_RXQ_UDP_PIPE";

  match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;
	match.parser_meta.outer_l4_type = DOCA_FLOW_L4_META_UDP;
  match.outer.udp.l4_port.src_port = htons(cfg.match_.udp_src_);
  match.outer.udp.l4_port.dst_port = htons(cfg.match_.udp_dst_);

  HOLOSCAN_LOG_INFO(
      "UDP pipe with src port {} dst port {}", cfg.match_.udp_src_, cfg.match_.udp_dst_);

  result = doca_flow_pipe_cfg_create(&pipe_cfg, df_port);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to create doca_flow_pipe_cfg: {}", doca_error_get_descr(result));
    return result;
  }

  result = doca_flow_pipe_cfg_set_name(pipe_cfg, pipe_name);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg name: {}", doca_error_get_descr(result));
    return result;
  }
  result = doca_flow_pipe_cfg_set_type(pipe_cfg, DOCA_FLOW_PIPE_BASIC);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg type: {}", doca_error_get_descr(result));
    return result;
  }
  result = doca_flow_pipe_cfg_set_is_root(pipe_cfg, false);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg is_root: %s",
                       doca_error_get_descr(result));
    return result;
  }
  result = doca_flow_pipe_cfg_set_match(pipe_cfg, &match, nullptr);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg match: {}", doca_error_get_descr(result));
    return result;
  }
  result = doca_flow_pipe_cfg_set_monitor(pipe_cfg, &monitor);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg monitor: %s",
                       doca_error_get_descr(result));
    return result;
  }

  rss_queues[0] = flow_queue_id;
  doca_eth_rxq_apply_queue_id(eth_rxq_cpu, rss_queues[0]);
  flow_queue_id++;

  fwd.type = DOCA_FLOW_FWD_RSS;
	fwd.rss_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
	fwd.rss.queues_array = rss_queues;
	fwd.rss.outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_UDP;
	fwd.rss.nr_queues = 1;

  if (rxq_pipe_default != nullptr) {
    miss_fwd.type = DOCA_FLOW_FWD_PIPE;
    miss_fwd.next_pipe = rxq_pipe_default;
  } else {
    miss_fwd.type = DOCA_FLOW_FWD_DROP;
  }

  result = doca_flow_pipe_create(pipe_cfg, &fwd, &miss_fwd, &(rxq_pipe));
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("RxQ pipe creation failed with: {}", doca_error_get_descr(result));
    return result;
  }

  /* Add HW offload */
  result = doca_flow_pipe_add_entry(
      0, rxq_pipe, &match, 0, nullptr, nullptr, nullptr, DOCA_FLOW_NO_WAIT, nullptr, &entry);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("RxQ pipe entry creation failed with: {}", doca_error_get_descr(result));
    return result;
  }

  // default_flow_timeout_usec = 0;
  result = doca_flow_entries_process(df_port, 0, default_flow_timeout_usec, 1);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("RxQ pipe entry process failed with: {}", doca_error_get_descr(result));
    return result;
  }

  HOLOSCAN_LOG_INFO("Created UDP Pipe {}", pipe_name);

  return DOCA_SUCCESS;
}

doca_error_t DocaRxQueue::create_rx_packet_list() {
  doca_error_t result;

  result = doca_gpu_mem_alloc(gdev, MAX_DEFAULT_SEM_X_QUEUE * sizeof(struct adv_doca_rx_gpu_info), GPU_PAGE_SIZE, DOCA_GPU_MEM_TYPE_CPU_GPU, (void**)&pkt_list_gpu, (void**)&pkt_list_cpu);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed allocate packet list memory: {}", doca_error_get_descr(result));
    return DOCA_ERROR_BAD_STATE;
  }

  memset(pkt_list_cpu, 0, MAX_DEFAULT_SEM_X_QUEUE * sizeof(struct adv_doca_rx_gpu_info));
  
  return result;
}

doca_error_t DocaRxQueue::destroy_rx_packet_list() {
  doca_error_t result = DOCA_SUCCESS;

  if (pkt_list_gpu) {
    result = doca_gpu_mem_free(gdev, pkt_list_gpu);
    if (result != DOCA_SUCCESS) {
      HOLOSCAN_LOG_ERROR("Failed destroy packet list memory: {}", doca_error_get_descr(result));
      return DOCA_ERROR_BAD_STATE;
    }
  }

  return result;
}

DocaTxQueue::DocaTxQueue(struct doca_dev* ddev_, struct doca_gpu* gdev_, uint16_t qid_,
                         int max_pkt_num_, int max_pkt_size_, enum doca_gpu_mem_type mtype,
                         doca_eth_txq_gpu_event_notify_send_packet_cb_t event_notify_send_packet_cb)
    : ddev(ddev_), gdev(gdev_), qid(qid_), max_pkt_num(max_pkt_num_), max_pkt_size(max_pkt_size_) {
  doca_error_t result;
  uint32_t tx_buffer_size = max_pkt_size * max_pkt_num;
  union doca_data event_user_data[1] = {0};

  result = doca_eth_txq_create(ddev, MAX_SQ_DESCR_NUM, &(eth_txq_cpu));
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed doca_eth_txq_create: {}", doca_error_get_descr(result));
  }

  result = doca_eth_txq_set_l3_chksum_offload(eth_txq_cpu, 1);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set eth_txq l3 offloads: {}", doca_error_get_descr(result));
  }

  result = doca_eth_txq_set_l4_chksum_offload(eth_txq_cpu, 1);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set eth_txq l3 offloads: {}", doca_error_get_descr(result));
  }

  eth_txq_ctx = doca_eth_txq_as_doca_ctx(eth_txq_cpu);
  if (eth_txq_ctx == nullptr) {
    HOLOSCAN_LOG_ERROR("Failed doca_eth_txq_as_doca_ctx: {}", doca_error_get_descr(result));
  }

  result = doca_ctx_set_datapath_on_gpu(eth_txq_ctx, gdev);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed doca_ctx_set_datapath_on_gpu: {}", doca_error_get_descr(result));
  }

  result = doca_pe_create(&pe);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Unable to create pe queue: {}", doca_error_get_descr(result));
  }

  event_user_data[0].u64 = (uint64_t)&tx_cmp_posted;
  result = doca_eth_txq_gpu_event_notify_send_packet_register(
      eth_txq_cpu, event_notify_send_packet_cb, event_user_data[0]);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Unable to set DOCA progress engine callback: {}",
                       doca_error_get_descr(result));
  }

  result = doca_pe_connect_ctx(pe, eth_txq_ctx);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Unable to set DOCA progress engine to DOCA Eth Txq: {}",
                       doca_error_get_descr(result));
  }

  result = doca_ctx_start(eth_txq_ctx);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed doca_ctx_start: {}", doca_error_get_descr(result));
  }

  result = doca_eth_txq_get_gpu_handle(eth_txq_cpu, &(eth_txq_gpu));
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed doca_eth_txq_get_gpu_handle: {}", doca_error_get_descr(result));
  }

  // Send buffer
  result = doca_mmap_create(&pkt_buff_mmap);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to create mmap: {}", doca_error_get_descr(result));
  }

  result = doca_mmap_add_dev(pkt_buff_mmap, ddev);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to add dev to mmap: {}", doca_error_get_descr(result));
  }

  ALIGN_SIZE(tx_buffer_size, get_host_page_size());

  result =
      doca_gpu_mem_alloc(gdev, tx_buffer_size, get_host_page_size(), mtype, &gpu_pkt_addr, &cpu_pkt_addr);
  if (result != DOCA_SUCCESS || gpu_pkt_addr == nullptr) {
    HOLOSCAN_LOG_ERROR("Failed to allocate gpu memory {}", doca_error_get_descr(result));
  }

  dmabuf_fd = -1;
  if (mtype == DOCA_GPU_MEM_TYPE_GPU) {
    /* Map GPU memory buffer used to send packets with DMABuf */
	  result = doca_gpu_dmabuf_fd(gdev, gpu_pkt_addr, tx_buffer_size, &(dmabuf_fd));
    if (result == DOCA_SUCCESS) {
      		HOLOSCAN_LOG_INFO("Mapping send queue buffer ({} size {}B dmabuf fd {}) with dmabuf mode",
			      gpu_pkt_addr,
			      tx_buffer_size,
			      dmabuf_fd);

      result = doca_mmap_set_dmabuf_memrange(pkt_buff_mmap,
                    dmabuf_fd,
                    gpu_pkt_addr,
                    0,
                    tx_buffer_size);
      if (result != DOCA_SUCCESS) {
        HOLOSCAN_LOG_ERROR("Failed to set dmabuf memrange for mmap {}", doca_error_get_descr(result));
      }
    }
  }

	if (dmabuf_fd == -1) {
		HOLOSCAN_LOG_INFO("Mapping send queue buffer ({} size {}B) with nvidia-peermem mode",
			      gpu_pkt_addr,
			      tx_buffer_size);

		/* If failed, use nvidia-peermem legacy method */
		result = doca_mmap_set_memrange(pkt_buff_mmap, gpu_pkt_addr, tx_buffer_size);
		if (result != DOCA_SUCCESS) {
			HOLOSCAN_LOG_ERROR("Failed to set memrange for mmap {}", doca_error_get_descr(result));
		}
	}

  result = doca_mmap_set_permissions(
      pkt_buff_mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_PCI_RELAXED_ORDERING);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set permissions for mmap {}", doca_error_get_descr(result));
  }

  result = doca_mmap_start(pkt_buff_mmap);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to start mmap {}", doca_error_get_descr(result));
  }
  
  result = doca_mmap_get_mkey(pkt_buff_mmap, ddev, &pkt_mkey);
	if (result != DOCA_SUCCESS) {
		HOLOSCAN_LOG_ERROR("Failed to get mmap mkey {}", doca_error_get_descr(result));
	}
	// N.B. mkey must be in network byte order
	pkt_mkey = htobe32(pkt_mkey);

  buff_arr_idx = 0;
  tx_cmp_posted = 0;
}

DocaTxQueue::~DocaTxQueue() {
  doca_error_t result;

  result = doca_ctx_stop(eth_txq_ctx);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed doca_ctx_stop: {}", doca_error_get_descr(result));
  }

  result = doca_eth_txq_destroy(eth_txq_cpu);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed doca_eth_rxq_destroy: {}", doca_error_get_descr(result));
  }
  result = doca_mmap_destroy(pkt_buff_mmap);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to destroy mmap: {}", doca_error_get_descr(result));
  }

  result = doca_gpu_mem_free(gdev, gpu_pkt_addr);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to free gpu memory: {}", doca_error_get_descr(result));
  }

  result = doca_pe_destroy(pe);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Function doca_pe_destroy returned {}", doca_error_get_descr(result));
  }

  HOLOSCAN_LOG_INFO("DocaTxQueue destroyed\n");
}

};  // namespace holoscan::advanced_network
