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
// cuDisp Performance / Stress Test
//
// Measures sustained present latency and throughput for host present.
// Optionally measures GPU present via cuDispLaunchPresentKernel.
//
// Usage:
//   perf_test [options]
//
// Options:
//   --card <N>              DRI card number (default: auto-detect)
//   --mode <WxH>            Display resolution (default: preferred mode)
//   --duration <seconds>    Test duration (default: 10)
//   --present-mode <host|gpu|both>  Which present path to test (default: host)
//   --vrr <on|off>          Skip frame pacing (default: off)
//   --help                  Show this help
//

#include <cuda.h>
#include <cuda_runtime.h>

#include <xf86drm.h>
#include <xf86drmMode.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <getopt.h>
#include <numeric>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "cuDisp.h"
#include "cuDispDevice.h"
#include "perf_test_kernels.h"

static inline CUresult create_cuda_context(CUcontext* ctx, unsigned int flags, CUdevice dev) {
#if CUDA_VERSION >= 13000
  return cuCtxCreate(ctx, nullptr, flags, dev);
#else
  return cuCtxCreate(ctx, flags, dev);
#endif
}

#define TAG "[perf_test] "

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

static void print_stats(const char* label, std::vector<double>& latencies, double total_s) {
  if (latencies.empty()) {
    printf(TAG "  %s: no successful presents\n", label);
    return;
  }
  std::sort(latencies.begin(), latencies.end());
  double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
  double avg = sum / latencies.size();
  size_t p99_idx = static_cast<size_t>(latencies.size() * 0.99);

  printf(TAG "  ------------------------------------------\n");
  printf(TAG "  %s\n", label);
  printf(TAG "  ------------------------------------------\n");
  printf(TAG "  Frames:     %zu\n", latencies.size());
  printf(TAG "  Duration:   %.1f s\n", total_s);
  printf(TAG "  Avg FPS:    %.1f\n", latencies.size() / total_s);
  printf(TAG "  Latency:    avg=%.2f ms  min=%.2f ms  max=%.2f ms  P99=%.2f ms\n",
         avg, latencies.front(), latencies.back(), latencies[p99_idx]);
}

static int run_host_test(cuDispSwapchain sc, CUdeviceptr* dptrs, uint64_t buf_size,
                         uint32_t num_buffers, double duration_s, cudaStream_t stream,
                         std::chrono::microseconds frame_interval) {
  printf(TAG "Running host present benchmark for %.0f seconds...\n", duration_s);

  uint32_t pixel_count = static_cast<uint32_t>(buf_size / 4);
  std::vector<double> latencies;
  auto t_start = std::chrono::steady_clock::now();
  auto next_frame = t_start + frame_interval;
  uint32_t frame = 0;
  uint32_t ebusy_count = 0;

  while (true) {
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - t_start).count() >= duration_s) break;

    uint32_t buf_idx = frame % num_buffers;
    uint32_t color = (frame % 2 == 0) ? 0xFF204060 : 0xFF602040;
    launch_fill_flat(reinterpret_cast<uint32_t*>(dptrs[buf_idx]), pixel_count, color, stream);
    cudaStreamSynchronize(stream);

    CUdeviceptr present_dptr = dptrs[buf_idx];
    cuDispBufferMemory mem = {};
    mem.devicePtr = &present_dptr;

    if (frame_interval.count() > 0) {
      std::this_thread::sleep_until(next_frame);
      next_frame += frame_interval;
    }

    auto t0 = std::chrono::steady_clock::now();
    cuDispStatus st = cuDispPresent(sc, nullptr, &mem, 1, 0);
    auto t1 = std::chrono::steady_clock::now();

    if (st == cuDispSuccess) {
      latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    } else {
      ebusy_count++;
    }
    frame++;
  }

  auto t_end = std::chrono::steady_clock::now();
  print_stats("Host Present", latencies,
              std::chrono::duration<double>(t_end - t_start).count());
  if (ebusy_count > 0)
    printf(TAG "  EBUSY:      %u / %u frames\n", ebusy_count, frame);
  return 0;
}

static int run_gpu_test(cuDispSwapchain sc, CUdeviceptr* dptrs, uint64_t buf_size,
                        uint32_t num_buffers, void* gpu_present_handle,
                        double duration_s, cudaStream_t stream,
                        std::chrono::microseconds frame_interval) {
  printf(TAG "Running GPU present benchmark for %.0f seconds...\n", duration_s);

  void** display_ptrs_device = nullptr;
  cudaMalloc(&display_ptrs_device, num_buffers * sizeof(void*));
  std::vector<void*> host_ptrs(num_buffers);
  for (uint32_t i = 0; i < num_buffers; i++)
    host_ptrs[i] = reinterpret_cast<void*>(dptrs[i]);
  cudaMemcpy(display_ptrs_device, host_ptrs.data(),
             num_buffers * sizeof(void*), cudaMemcpyHostToDevice);

  void** display_ptr_location = nullptr;
  cudaMalloc(&display_ptr_location, sizeof(void*));
  void* initial = host_ptrs[1 % num_buffers];
  cudaMemcpy(display_ptr_location, &initial, sizeof(void*), cudaMemcpyHostToDevice);

  void** display_ptr_locations_device = nullptr;
  cudaMalloc(&display_ptr_locations_device, sizeof(void*));
  cudaMemcpy(display_ptr_locations_device, &display_ptr_location,
             sizeof(void*), cudaMemcpyHostToDevice);

  unsigned int* num_bufs_device = nullptr;
  cudaMalloc(&num_bufs_device, sizeof(unsigned int));
  unsigned int host_num = num_buffers;
  cudaMemcpy(num_bufs_device, &host_num, sizeof(unsigned int), cudaMemcpyHostToDevice);

  CUdeviceptr init_dptr = dptrs[0];
  cuDispBufferMemory init_mem = {};
  init_mem.devicePtr = &init_dptr;
  cuDispStatus st = cuDispPresent(sc, nullptr, &init_mem, 1, 0);
  if (st != cuDispSuccess) {
    printf(TAG "ERROR: initial host present for GPU test failed (%d)\n", st);
    return 1;
  }

  uint32_t pixel_count = static_cast<uint32_t>(buf_size / 4);
  std::vector<double> latencies;
  auto t_start = std::chrono::steady_clock::now();
  auto next_frame = t_start + frame_interval;
  uint32_t frame = 0;

  while (true) {
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - t_start).count() >= duration_s) break;

    uint32_t color = (frame % 2 == 0) ? 0xFF106030 : 0xFF301060;

    void* render_target = nullptr;
    cudaMemcpy(&render_target, display_ptr_location, sizeof(void*), cudaMemcpyDeviceToHost);

    launch_fill_flat(reinterpret_cast<uint32_t*>(render_target), pixel_count, color, stream);

    if (frame_interval.count() > 0) {
      std::this_thread::sleep_until(next_frame);
      next_frame += frame_interval;
    }

    auto t0 = std::chrono::steady_clock::now();
    cuDispLaunchPresentKernel(stream,
                              display_ptr_locations_device,
                              gpu_present_handle,
                              display_ptrs_device,
                              num_bufs_device,
                              1u);
    cudaStreamSynchronize(stream);
    auto t1 = std::chrono::steady_clock::now();

    latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    frame++;
  }

  auto t_end = std::chrono::steady_clock::now();
  print_stats("GPU Present", latencies,
              std::chrono::duration<double>(t_end - t_start).count());

  cudaFree(display_ptrs_device);
  cudaFree(display_ptr_location);
  cudaFree(display_ptr_locations_device);
  cudaFree(num_bufs_device);
  return 0;
}

int main(int argc, char** argv) {
  int card = -1;
  uint32_t req_w = 0, req_h = 0;
  double duration_s = 10.0;
  std::string present_mode = "host";
  bool vrr = false;

  static struct option long_opts[] = {
    {"card",         required_argument, nullptr, 'c'},
    {"mode",         required_argument, nullptr, 'm'},
    {"duration",     required_argument, nullptr, 'd'},
    {"present-mode", required_argument, nullptr, 'p'},
    {"vrr",          required_argument, nullptr, 'v'},
    {"help",         no_argument,       nullptr, 'h'},
    {nullptr, 0, nullptr, 0}
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "c:m:d:p:v:h", long_opts, nullptr)) != -1) {
    switch (opt) {
      case 'c': card = atoi(optarg); break;
      case 'm': sscanf(optarg, "%ux%u", &req_w, &req_h); break;
      case 'd': duration_s = atof(optarg); break;
      case 'p': present_mode = optarg; break;
      case 'v': vrr = (strcmp(optarg, "on") == 0); break;
      case 'h':
        printf("Usage: perf_test [--card N] [--mode WxH] [--duration secs]\n"
               "       [--present-mode host|gpu|both] [--vrr on|off]\n");
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
  auto frame_interval = vrr ? std::chrono::microseconds(0)
                            : std::chrono::microseconds(1050000 / refresh_hz);

  bool do_host = (present_mode == "host" || present_mode == "both");
  bool do_gpu = (present_mode == "gpu" || present_mode == "both");

  if (cuInit(0) != CUDA_SUCCESS) {
    printf(TAG "ERROR: cuInit failed\n");
    return 1;
  }
  CUdevice dev;
  CUcontext ctx;
  cuDeviceGet(&dev, 0);
  if (create_cuda_context(&ctx, 0, dev) != CUDA_SUCCESS) {
    printf(TAG "ERROR: cuCtxCreate failed\n");
    return 1;
  }

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  printf(TAG "==========================================\n");
  printf(TAG "  cuDisp Performance Test\n");
  printf(TAG "==========================================\n");
  printf(TAG "  Mode:          %ux%u @ %u Hz\n", mode_w, mode_h, refresh_hz);
  printf(TAG "  Duration:      %.0f s per test\n", duration_s);
  printf(TAG "  Present mode:  %s\n", present_mode.c_str());
  printf(TAG "  VRR:           %s\n", vrr ? "on (no pacing)" : "off");
  printf(TAG "==========================================\n\n");

  const uint32_t NUM_BUFFERS = 2;
  int ret = 0;

  if (do_host) {
    cuDispCreateAttribute attrs[2];
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

    cuDispSwapchain sc = nullptr;
    cuDispStatus st = cuDispCreateSwapchain(&sc, attrs, 2, 0);
    if (st != cuDispSuccess) {
      printf(TAG "ERROR: cuDispCreateSwapchain (host) failed (%d)\n", st);
      ret = 1;
    } else {
      CUdeviceptr dptrs[NUM_BUFFERS];
      uint64_t buf_size = 0;
      for (uint32_t i = 0; i < NUM_BUFFERS; i++) {
        cuDispBufferMemory mem = {};
        mem.devicePtr = &dptrs[i];
        mem.size = (i == 0) ? &buf_size : nullptr;
        cuDispGetBuffer(sc, 0, i, &mem, 0);
      }
      ret = run_host_test(sc, dptrs, buf_size, NUM_BUFFERS, duration_s, stream,
                          frame_interval);
      cuDispDestroySwapchain(sc);
    }
  }

  if (do_gpu && ret == 0) {
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
      printf(TAG "ERROR: cuDispCreateSwapchain (gpu) failed (%d)\n", st);
      ret = 1;
    } else {
      CUdeviceptr dptrs[NUM_BUFFERS];
      uint64_t buf_size = 0;
      for (uint32_t i = 0; i < NUM_BUFFERS; i++) {
        cuDispBufferMemory mem = {};
        mem.devicePtr = &dptrs[i];
        mem.size = (i == 0) ? &buf_size : nullptr;
        cuDispGetBuffer(sc, 0, i, &mem, 0);
      }
      ret = run_gpu_test(sc, dptrs, buf_size, NUM_BUFFERS, gpu_handle, duration_s, stream,
                         frame_interval);
      cuDispDestroySwapchain(sc);
    }
  }

  cudaStreamDestroy(stream);
  cuCtxDestroy(ctx);

  printf(TAG "\nDone.\n");
  return ret;
}
