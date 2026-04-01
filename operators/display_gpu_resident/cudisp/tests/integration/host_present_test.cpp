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
// cuDisp Host Present Integration Test
//
// Exercises the full cuDisp public API via host (CPU) presents.
// Creates a swapchain, acquires buffers, fills with a CUDA kernel,
// and presents in a loop. No GPU present thread.
//
// Usage:
//   host_present_test [options]
//
// Options:
//   --card <N>            DRI card number (default: auto-detect)
//   --mode <WxH>          Display resolution (default: preferred mode)
//   --format <name>       Surface format: ARGB8888, XRGB8888, ABGR16161616
//   --num-buffers <1|2>   Buffer count (default: 2)
//   --frames <N>          Number of frames to present (default: 300)
//   --pattern <name>      Fill pattern: solid, gradient, bars (default: bars)
//   --help                Show this help
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
#include "host_present_kernels.h"

static inline CUresult create_cuda_context(CUcontext* ctx, unsigned int flags, CUdevice dev) {
#if CUDA_VERSION >= 13000
  return cuCtxCreate(ctx, nullptr, flags, dev);
#else
  return cuCtxCreate(ctx, flags, dev);
#endif
}

#define TAG "[host_present] "

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

static cuDispSurfaceFormat parse_format(const char* name) {
  if (strcasecmp(name, "XRGB8888") == 0) return CUDISP_SURFACE_FORMAT_XRGB8888;
  if (strcasecmp(name, "ABGR16161616") == 0) return CUDISP_SURFACE_FORMAT_ABGR16161616;
  return CUDISP_SURFACE_FORMAT_ARGB8888;
}

static const char* format_name(cuDispSurfaceFormat f) {
  switch (f) {
    case CUDISP_SURFACE_FORMAT_ARGB8888: return "ARGB8888";
    case CUDISP_SURFACE_FORMAT_XRGB8888: return "XRGB8888";
    case CUDISP_SURFACE_FORMAT_ABGR16161616: return "ABGR16161616";
    default: return "unknown";
  }
}

int main(int argc, char** argv) {
  int card = -1;
  uint32_t req_w = 0, req_h = 0;
  uint32_t num_buffers = 2;
  uint32_t num_frames = 300;
  cuDispSurfaceFormat format = CUDISP_SURFACE_FORMAT_ARGB8888;
  std::string pattern = "bars";

  static struct option long_opts[] = {
    {"card",        required_argument, nullptr, 'c'},
    {"mode",        required_argument, nullptr, 'm'},
    {"format",      required_argument, nullptr, 'f'},
    {"num-buffers", required_argument, nullptr, 'b'},
    {"frames",      required_argument, nullptr, 'n'},
    {"pattern",     required_argument, nullptr, 'p'},
    {"help",        no_argument,       nullptr, 'h'},
    {nullptr, 0, nullptr, 0}
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "c:m:f:b:n:p:h", long_opts, nullptr)) != -1) {
    switch (opt) {
      case 'c': card = atoi(optarg); break;
      case 'm': sscanf(optarg, "%ux%u", &req_w, &req_h); break;
      case 'f': format = parse_format(optarg); break;
      case 'b': num_buffers = static_cast<uint32_t>(atoi(optarg)); break;
      case 'n': num_frames = static_cast<uint32_t>(atoi(optarg)); break;
      case 'p': pattern = optarg; break;
      case 'h':
        printf("Usage: host_present_test [--card N] [--mode WxH] [--format name]\n"
               "       [--num-buffers 1|2] [--frames N] [--pattern solid|gradient|bars]\n");
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
      printf(TAG "ERROR: could not query preferred mode for card%d\n", card);
      return 1;
    }
    printf(TAG "Using preferred mode: %ux%u @ %u Hz\n", mode_w, mode_h, refresh_hz);
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
  if (create_cuda_context(&ctx, 0, dev) != CUDA_SUCCESS) {
    printf(TAG "ERROR: cuCtxCreate failed\n");
    return 1;
  }

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  printf(TAG "Creating swapchain: %ux%u, format=%s, buffers=%u\n",
         mode_w, mode_h, format_name(format), num_buffers);

  cuDispCreateAttribute attrs[2];
  memset(attrs, 0, sizeof(attrs));

  attrs[0].id = CUDISP_CREATE_ATTRIBUTE_MODE_INFO;
  attrs[0].value.modeInfo.modeWidth = mode_w;
  attrs[0].value.modeInfo.modeHeight = mode_h;

  attrs[1].id = CUDISP_CREATE_ATTRIBUTE_BUFFER_INFO;
  attrs[1].value.bufferInfo.layerIndex = 0;
  attrs[1].value.bufferInfo.numBuffers = num_buffers;
  attrs[1].value.bufferInfo.format = format;
  attrs[1].value.bufferInfo.width = mode_w;
  attrs[1].value.bufferInfo.height = mode_h;
  attrs[1].value.bufferInfo.alpha = 0xFFFF;

  cuDispSwapchain sc = nullptr;
  cuDispStatus st = cuDispCreateSwapchain(&sc, attrs, 2, 0);
  if (st != cuDispSuccess) {
    printf(TAG "ERROR: cuDispCreateSwapchain failed (%d)\n", st);
    cuCtxDestroy(ctx);
    return 1;
  }

  struct BufInfo { CUdeviceptr dptr; uint64_t size; uint32_t stride; };
  std::vector<BufInfo> bufs(num_buffers);
  for (uint32_t i = 0; i < num_buffers; i++) {
    bufs[i] = {};
    cuDispBufferMemory mem = {};
    mem.devicePtr = &bufs[i].dptr;
    mem.size = &bufs[i].size;
    mem.stride = &bufs[i].stride;
    st = cuDispGetBuffer(sc, 0, i, &mem, 0);
    if (st != cuDispSuccess) {
      printf(TAG "ERROR: cuDispGetBuffer(%u) failed (%d)\n", i, st);
      cuDispDestroySwapchain(sc);
      cuCtxDestroy(ctx);
      return 1;
    }
    printf(TAG "  Buffer %u: dptr=0x%llx, size=%llu, stride=%u\n",
           i, (unsigned long long)bufs[i].dptr,
           (unsigned long long)bufs[i].size, bufs[i].stride);
  }

  uint32_t stride_px = bufs[0].stride / 4;

  cudaMemsetAsync(reinterpret_cast<void*>(bufs[0].dptr), 0, bufs[0].size, stream);
  cudaStreamSynchronize(stream);

  CUdeviceptr present_dptr = bufs[0].dptr;
  cuDispBufferMemory present_mem = {};
  present_mem.devicePtr = &present_dptr;
  st = cuDispPresent(sc, nullptr, &present_mem, 1, 0);
  if (st != cuDispSuccess) {
    printf(TAG "ERROR: initial present failed (%d)\n", st);
    cuDispDestroySwapchain(sc);
    cuCtxDestroy(ctx);
    return 1;
  }

  printf(TAG "Presenting %u frames (%s pattern)...\n", num_frames, pattern.c_str());
  printf(TAG ">>> Color bars should be visible and scrolling on the display. <<<\n");

  std::vector<double> latencies;
  latencies.reserve(num_frames);
  uint32_t ebusy_count = 0;
  auto t_start = std::chrono::steady_clock::now();
  auto next_frame = t_start + frame_interval;

  for (uint32_t f = 0; f < num_frames; f++) {
    uint32_t buf_idx = f % num_buffers;
    uint32_t* dst = reinterpret_cast<uint32_t*>(bufs[buf_idx].dptr);

    if (pattern == "solid") {
      uint32_t color = (f % 3 == 0) ? 0xFFFF0000 : (f % 3 == 1) ? 0xFF00FF00 : 0xFF0000FF;
      launch_fill_solid(dst, stride_px, mode_w, mode_h, color, stream);
    } else if (pattern == "gradient") {
      launch_fill_gradient(dst, stride_px, mode_w, mode_h, f * 4, stream);
    } else {
      launch_fill_bars(dst, stride_px, mode_w, mode_h, f * 8, stream);
    }
    cudaStreamSynchronize(stream);

    present_dptr = bufs[buf_idx].dptr;
    present_mem.devicePtr = &present_dptr;

    std::this_thread::sleep_until(next_frame);
    next_frame += frame_interval;

    auto t0 = std::chrono::steady_clock::now();
    st = cuDispPresent(sc, nullptr, &present_mem, 1, 0);
    auto t1 = std::chrono::steady_clock::now();

    if (st != cuDispSuccess) {
      ebusy_count++;
    } else {
      latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  double total_s = std::chrono::duration<double>(t_end - t_start).count();

  printf("\n");
  printf(TAG "==========================================\n");
  printf(TAG "  Host Present Integration Test Results\n");
  printf(TAG "==========================================\n");
  printf(TAG "  Mode:       %ux%u\n", mode_w, mode_h);
  printf(TAG "  Format:     %s\n", format_name(format));
  printf(TAG "  Buffers:    %u\n", num_buffers);
  printf(TAG "  Pattern:    %s\n", pattern.c_str());
  printf(TAG "  Frames:     %u requested, %zu presented\n", num_frames, latencies.size());
  printf(TAG "  Duration:   %.1f s\n", total_s);
  printf(TAG "  Avg FPS:    %.1f\n", latencies.size() / total_s);
  if (ebusy_count > 0)
    printf(TAG "  EBUSY:      %u / %u frames\n", ebusy_count, num_frames);

  if (!latencies.empty()) {
    std::sort(latencies.begin(), latencies.end());
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / latencies.size();
    size_t p99_idx = static_cast<size_t>(latencies.size() * 0.99);
    printf(TAG "  Present latency:\n");
    printf(TAG "    avg=%.2f ms  min=%.2f ms  max=%.2f ms  P99=%.2f ms\n",
           avg, latencies.front(), latencies.back(), latencies[p99_idx]);
  }
  printf(TAG "==========================================\n");

  cudaStreamDestroy(stream);
  cuDispDestroySwapchain(sc);
  cuCtxDestroy(ctx);

  printf(TAG "Done.\n");
  return 0;
}
