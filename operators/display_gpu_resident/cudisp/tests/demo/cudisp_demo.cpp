/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//
// cuDisp Demo Application
//
// Minimal standalone application demonstrating the cuDisp API with GPU
// present. Renders animated color bars using a CUDA kernel and presents
// via cuDispLaunchPresentKernel (GPU-driven present thread).
//
// This uses only currently-supported cuDisp features:
//   - ARGB8888 / XRGB8888 / ABGR16161616 formats
//   - Single layer (primary plane)
//   - Double-buffered swapchain
//   - GPU present path
//
// Usage:
//   cudisp_demo [options]
//
// Options:
//   --card <N>            DRI card number (default: auto-detect)
//   --mode <WxH>          Display resolution (default: preferred mode)
//   --duration <seconds>  How long to run (default: 10)
//   --help                Show this help
//

#include <cuda.h>
#include <cuda_runtime.h>

#include <xf86drm.h>
#include <xf86drmMode.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <getopt.h>
#include <thread>
#include <unistd.h>

#include "cuDisp.h"
#include "cuDispDevice.h"
#include "cudisp_demo_kernels.h"

#define TAG "[cudisp_demo] "

static bool query_preferred_mode(int card, uint32_t* out_w, uint32_t* out_h,
                                 uint32_t* out_refresh_hz = nullptr) {
  char path[64];
  snprintf(path, sizeof(path), "/dev/dri/card%d", card);
  int fd = open(path, O_RDWR | O_CLOEXEC);
  if (fd < 0) return false;

  drmModeRes* res = drmModeGetResources(fd);
  if (!res) { close(fd); return false; }

  bool found = false;
  for (int c = 0; c < res->count_connectors && !found; c++) {
    drmModeConnector* conn = drmModeGetConnector(fd, res->connectors[c]);
    if (!conn) continue;
    if (conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
      for (int m = 0; m < conn->count_modes; m++) {
        if (conn->modes[m].type & DRM_MODE_TYPE_PREFERRED) {
          *out_w = conn->modes[m].hdisplay;
          *out_h = conn->modes[m].vdisplay;
          if (out_refresh_hz) *out_refresh_hz = conn->modes[m].vrefresh;
          found = true;
          break;
        }
      }
      if (!found) {
        *out_w = conn->modes[0].hdisplay;
        *out_h = conn->modes[0].vdisplay;
        if (out_refresh_hz) *out_refresh_hz = conn->modes[0].vrefresh;
        found = true;
      }
    }
    drmModeFreeConnector(conn);
  }
  drmModeFreeResources(res);
  close(fd);
  return found;
}

static int detect_card() {
  DIR* dir = opendir("/dev/dri");
  if (!dir) return -1;
  int card = -1;
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (strncmp(entry->d_name, "card", 4) == 0) {
      card = atoi(entry->d_name + 4);
      break;
    }
  }
  closedir(dir);
  return card;
}

int main(int argc, char** argv) {
  int card = -1;
  uint32_t req_w = 0, req_h = 0;
  double duration_s = 10.0;

  static struct option long_opts[] = {
    {"card",     required_argument, nullptr, 'c'},
    {"mode",     required_argument, nullptr, 'm'},
    {"duration", required_argument, nullptr, 'd'},
    {"help",     no_argument,       nullptr, 'h'},
    {nullptr, 0, nullptr, 0}
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "c:m:d:h", long_opts, nullptr)) != -1) {
    switch (opt) {
      case 'c': card = atoi(optarg); break;
      case 'm': sscanf(optarg, "%ux%u", &req_w, &req_h); break;
      case 'd': duration_s = atof(optarg); break;
      case 'h':
        printf("Usage: cudisp_demo [--card N] [--mode WxH] [--duration secs]\n");
        return 0;
      default: return 1;
    }
  }

  if (card < 0) card = detect_card();
  if (card < 0) {
    printf(TAG "ERROR: no DRM device found\n");
    return 1;
  }

  uint32_t mode_w = req_w, mode_h = req_h;
  uint32_t refresh_hz = 60;
  if (mode_w == 0 || mode_h == 0) {
    if (!query_preferred_mode(card, &mode_w, &mode_h, &refresh_hz)) {
      printf(TAG "ERROR: could not query preferred mode\n");
      return 1;
    }
  }
  if (refresh_hz == 0) refresh_hz = 60;
  auto frame_interval = std::chrono::microseconds(1050000 / refresh_hz);

  if (cuInit(0) != CUDA_SUCCESS) {
    printf(TAG "ERROR: cuInit failed\n");
    return 1;
  }
  CUdevice dev;
  CUcontext ctx;
  cuDeviceGet(&dev, 0);
  if (cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) {
    printf(TAG "ERROR: cuCtxCreate failed\n");
    return 1;
  }

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  const uint32_t NUM_BUFFERS = 2;

  printf(TAG "Creating swapchain: %ux%u, ARGB8888, %u buffers, GPU present\n",
         mode_w, mode_h, NUM_BUFFERS);

  cuDispCreateAttribute attrs[3];
  memset(attrs, 0, sizeof(attrs));

  attrs[0].id = CUDISP_CREATE_ATTRIBUTE_MODE_INFO;
  attrs[0].value.modeInfo.modeWidth = mode_w;
  attrs[0].value.modeInfo.modeHeight = mode_h;

  attrs[1].id = CUDISP_CREATE_ATTRIBUTE_BUFFER_INFO;
  attrs[1].value.bufferInfo.layerIndex = 0;
  attrs[1].value.bufferInfo.numBuffers = NUM_BUFFERS;
  attrs[1].value.bufferInfo.format = CUDISP_SURFACE_FORMAT_ARGB8888;
  attrs[1].value.bufferInfo.width = mode_w;
  attrs[1].value.bufferInfo.height = mode_h;
  attrs[1].value.bufferInfo.alpha = 0xFFFF;

  void* gpu_handle = nullptr;
  attrs[2].id = CUDISP_CREATE_ATTRIBUTE_GPU_PRESENT;
  attrs[2].value.gpuPresent.handleGPUPresent = &gpu_handle;

  cuDispSwapchain sc = nullptr;
  cuDispStatus st = cuDispCreateSwapchain(&sc, attrs, 3, 0);
  if (st != cuDispSuccess) {
    printf(TAG "ERROR: cuDispCreateSwapchain failed (%d)\n", st);
    cuCtxDestroy(ctx);
    return 1;
  }

  CUdeviceptr dptrs[NUM_BUFFERS];
  uint64_t buf_size = 0;
  uint32_t stride = 0;

  for (uint32_t i = 0; i < NUM_BUFFERS; i++) {
    cuDispBufferMemory mem = {};
    mem.devicePtr = &dptrs[i];
    mem.size = (i == 0) ? &buf_size : nullptr;
    mem.stride = (i == 0) ? &stride : nullptr;
    st = cuDispGetBuffer(sc, 0, i, &mem, 0);
    if (st != cuDispSuccess) {
      printf(TAG "ERROR: cuDispGetBuffer(%u) failed (%d)\n", i, st);
      cuDispDestroySwapchain(sc);
      cuCtxDestroy(ctx);
      return 1;
    }
  }

  uint32_t stride_px = stride / 4;
  printf(TAG "Buffers acquired: stride=%u bytes (%u px), size=%llu\n",
         stride, stride_px, (unsigned long long)buf_size);

  for (uint32_t i = 0; i < NUM_BUFFERS; i++)
    cudaMemset(reinterpret_cast<void*>(dptrs[i]), 0, buf_size);

  // GPU present data structures
  void** display_ptr_location = nullptr;
  cudaMalloc(&display_ptr_location, sizeof(void*));
  void* initial_render_target = reinterpret_cast<void*>(dptrs[1]);
  cudaMemcpy(display_ptr_location, &initial_render_target,
             sizeof(void*), cudaMemcpyHostToDevice);

  void** display_ptr_locations_device = nullptr;
  cudaMalloc(&display_ptr_locations_device, sizeof(void*));
  cudaMemcpy(display_ptr_locations_device, &display_ptr_location,
             sizeof(void*), cudaMemcpyHostToDevice);

  void** display_ptrs_device = nullptr;
  cudaMalloc(&display_ptrs_device, NUM_BUFFERS * sizeof(void*));
  void* host_ptrs[NUM_BUFFERS];
  for (uint32_t i = 0; i < NUM_BUFFERS; i++)
    host_ptrs[i] = reinterpret_cast<void*>(dptrs[i]);
  cudaMemcpy(display_ptrs_device, host_ptrs,
             NUM_BUFFERS * sizeof(void*), cudaMemcpyHostToDevice);

  unsigned int* num_bufs_device = nullptr;
  cudaMalloc(&num_bufs_device, sizeof(unsigned int));
  unsigned int host_num = NUM_BUFFERS;
  cudaMemcpy(num_bufs_device, &host_num, sizeof(unsigned int), cudaMemcpyHostToDevice);

  // Initial host present (starts the present thread)
  CUdeviceptr init_dptr = dptrs[0];
  cuDispBufferMemory init_mem = {};
  init_mem.devicePtr = &init_dptr;
  st = cuDispPresent(sc, nullptr, &init_mem, 1, 0);
  if (st != cuDispSuccess) {
    printf(TAG "ERROR: initial present failed (%d)\n", st);
    cuDispDestroySwapchain(sc);
    cuCtxDestroy(ctx);
    return 1;
  }
  cudaStreamSynchronize(stream);

  printf(TAG "Running for %.0f seconds (GPU present, animated color bars)...\n", duration_s);
  printf(TAG ">>> Scrolling color bars should be visible on the display. <<<\n");

  auto t_start = std::chrono::steady_clock::now();
  auto next_frame = t_start + frame_interval;
  uint32_t frame = 0;

  while (true) {
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - t_start).count() >= duration_s) break;

    void* render_target = nullptr;
    cudaMemcpy(&render_target, display_ptr_location, sizeof(void*), cudaMemcpyDeviceToHost);

    launch_render_bars(reinterpret_cast<uint32_t*>(render_target),
                       stride_px, mode_w, mode_h, frame, stream);

    std::this_thread::sleep_until(next_frame);
    next_frame += frame_interval;

    cuDispLaunchPresentKernel(stream,
                              display_ptr_locations_device,
                              gpu_handle,
                              display_ptrs_device,
                              num_bufs_device,
                              1u);

    cudaStreamSynchronize(stream);
    frame++;
  }

  auto t_end = std::chrono::steady_clock::now();
  double total_s = std::chrono::duration<double>(t_end - t_start).count();

  printf(TAG "\n");
  printf(TAG "==========================================\n");
  printf(TAG "  cuDisp Demo Results\n");
  printf(TAG "==========================================\n");
  printf(TAG "  Mode:       %ux%u\n", mode_w, mode_h);
  printf(TAG "  Format:     ARGB8888\n");
  printf(TAG "  Buffers:    %u (GPU present)\n", NUM_BUFFERS);
  printf(TAG "  Duration:   %.1f s\n", total_s);
  printf(TAG "  Frames:     %u\n", frame);
  printf(TAG "  Avg FPS:    %.1f\n", frame / total_s);
  printf(TAG "==========================================\n");

  cudaFree(display_ptrs_device);
  cudaFree(display_ptr_location);
  cudaFree(display_ptr_locations_device);
  cudaFree(num_bufs_device);
  cudaStreamDestroy(stream);
  cuDispDestroySwapchain(sc);
  cuCtxDestroy(ctx);

  printf(TAG "Done.\n");
  return 0;
}
