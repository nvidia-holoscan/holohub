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
// GBM -> Vulkan -> CUDA Buffer Pipeline Validation Test
//
// Validates the full buffer interop pipeline used by cuDisp:
//   GBM alloc -> dmabuf export -> Vulkan import -> Vulkan re-export
//   -> CUDA import -> CUDA memset/kernel fill -> DRM scanout
//
// Supports both CPU fill (via cuMemcpy from host) and CUDA fill
// (via cuMemsetD32) to validate both paths.
//
// Usage:
//   buffer_pipeline_test [options]
//
// Options:
//   --card <N>           DRI card number (default: auto-detect)
//   --display <name>     Connector or monitor name
//   --mode <WxH>         Display mode (default: preferred)
//   --fill <cpu|cuda>    Fill method (default: cuda)
//   --duration <sec>     Hold duration (default: 3)
//   --help
//

#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>
#include <gbm.h>
#include <vulkan/vulkan.h>

#include <cuda.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <getopt.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

static inline CUresult create_cuda_context(CUcontext* ctx, unsigned int flags, CUdevice dev) {
#if CUDA_VERSION >= 13000
  return cuCtxCreate(ctx, nullptr, flags, dev);
#else
  return cuCtxCreate(ctx, flags, dev);
#endif
}

// ============================================================================
// Connector / DRM helpers
// ============================================================================
static const char* connector_type_name(uint32_t type) {
  static const char* names[] = {
    "Unknown", "VGA", "DVI-I", "DVI-D", "DVI-A", "Composite",
    "SVIDEO", "LVDS", "Component", "9PinDIN", "DP",
    "HDMI-A", "HDMI-B", "TV", "eDP", "Virtual", "DSI", "DPI",
    "Writeback", "SPI", "USB"
  };
  if (type < sizeof(names) / sizeof(names[0])) return names[type];
  return "Unknown";
}

static std::string connector_name_str(drmModeConnector* c) {
  return std::string(connector_type_name(c->connector_type)) +
         "-" + std::to_string(c->connector_type_id);
}

static std::string get_edid_monitor_name(int fd, uint32_t connector_id) {
  drmModeObjectPropertiesPtr props =
      drmModeObjectGetProperties(fd, connector_id, DRM_MODE_OBJECT_CONNECTOR);
  if (!props) return "";
  std::string result;
  for (uint32_t i = 0; i < props->count_props && result.empty(); i++) {
    drmModePropertyPtr prop = drmModeGetProperty(fd, props->props[i]);
    if (!prop) continue;
    if (strcmp(prop->name, "EDID") == 0 && (prop->flags & DRM_MODE_PROP_BLOB)) {
      drmModePropertyBlobPtr blob =
          drmModeGetPropertyBlob(fd, props->prop_values[i]);
      if (blob && blob->length >= 128) {
        auto* edid = static_cast<const uint8_t*>(blob->data);
        for (int d = 0; d < 4; d++) {
          int off = 54 + d * 18;
          if (off + 18 > static_cast<int>(blob->length)) break;
          if (edid[off] == 0 && edid[off+1] == 0 &&
              edid[off+2] == 0 && edid[off+3] == 0xFC) {
            char name[14] = {};
            memcpy(name, &edid[off + 5], 13);
            for (int k = 12; k >= 0; k--) {
              if (name[k] == '\n' || name[k] == ' ' || name[k] == '\0')
                name[k] = '\0';
              else break;
            }
            result = name;
          }
        }
      }
      if (blob) drmModeFreePropertyBlob(blob);
    }
    drmModeFreeProperty(prop);
  }
  drmModeFreeObjectProperties(props);
  return result;
}

static uint32_t mode_refresh_hz(const drmModeModeInfo* m) {
  if (m->htotal == 0 || m->vtotal == 0) return 0;
  return static_cast<uint32_t>(
      (uint64_t)m->clock * 1000ULL / ((uint64_t)m->htotal * m->vtotal));
}

static uint32_t find_plane_prop(int fd, uint32_t plane_id, const char* name) {
  drmModeObjectPropertiesPtr props =
      drmModeObjectGetProperties(fd, plane_id, DRM_MODE_OBJECT_PLANE);
  if (!props) return 0;
  uint32_t result = 0;
  for (uint32_t i = 0; i < props->count_props; i++) {
    drmModePropertyPtr prop = drmModeGetProperty(fd, props->props[i]);
    if (prop) {
      if (strcmp(prop->name, name) == 0) result = prop->prop_id;
      drmModeFreeProperty(prop);
    }
    if (result) break;
  }
  drmModeFreeObjectProperties(props);
  return result;
}

static int get_plane_type(int fd, uint32_t plane_id) {
  drmModeObjectPropertiesPtr props =
      drmModeObjectGetProperties(fd, plane_id, DRM_MODE_OBJECT_PLANE);
  if (!props) return -1;
  int type = -1;
  for (uint32_t i = 0; i < props->count_props; i++) {
    drmModePropertyPtr prop = drmModeGetProperty(fd, props->props[i]);
    if (prop && strcmp(prop->name, "type") == 0) {
      type = static_cast<int>(props->prop_values[i]);
      drmModeFreeProperty(prop);
      break;
    }
    if (prop) drmModeFreeProperty(prop);
  }
  drmModeFreeObjectProperties(props);
  return type;
}

static int open_drm_device(int card_num) {
  char path[64];
  if (card_num >= 0) {
    snprintf(path, sizeof(path), "/dev/dri/card%d", card_num);
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) printf("[ERROR] Cannot open %s: %s\n", path, strerror(errno));
    return fd;
  }
  DIR* dir = opendir("/dev/dri");
  if (!dir) return -1;
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (strncmp(entry->d_name, "card", 4) != 0) continue;
    snprintf(path, sizeof(path), "/dev/dri/%.50s", entry->d_name);
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) continue;
    drmModeRes* res = drmModeGetResources(fd);
    if (res && res->count_connectors > 0) {
      drmModeFreeResources(res);
      closedir(dir);
      printf("[INFO] Auto-detected DRM device: %s\n", path);
      return fd;
    }
    if (res) drmModeFreeResources(res);
    close(fd);
  }
  closedir(dir);
  return -1;
}

// ============================================================================
// Vulkan context
// ============================================================================
struct VulkanContext {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
};

static bool init_vulkan(VulkanContext* vk) {
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "buffer_pipeline_test";
  appInfo.apiVersion = VK_API_VERSION_1_0;

  const char* instExts[] = {VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME};
  VkInstanceCreateInfo instInfo = {};
  instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instInfo.pApplicationInfo = &appInfo;
  instInfo.enabledExtensionCount = 1;
  instInfo.ppEnabledExtensionNames = instExts;

  if (vkCreateInstance(&instInfo, nullptr, &vk->instance) != VK_SUCCESS) {
    printf("  [FAIL] vkCreateInstance\n");
    return false;
  }

  uint32_t devCount = 0;
  vkEnumeratePhysicalDevices(vk->instance, &devCount, nullptr);
  if (devCount == 0) {
    printf("  [FAIL] No Vulkan physical devices.\n");
    return false;
  }
  VkPhysicalDevice devs[16];
  devCount = std::min(devCount, 16u);
  vkEnumeratePhysicalDevices(vk->instance, &devCount, devs);
  vk->physicalDevice = devs[0];

  const char* devExts[] = {
    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME
  };

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qInfo = {};
  qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qInfo.queueCount = 1;
  qInfo.pQueuePriorities = &prio;

  VkDeviceCreateInfo devInfo = {};
  devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  devInfo.queueCreateInfoCount = 1;
  devInfo.pQueueCreateInfos = &qInfo;
  devInfo.enabledExtensionCount = 3;
  devInfo.ppEnabledExtensionNames = devExts;

  if (vkCreateDevice(vk->physicalDevice, &devInfo, nullptr,
                     &vk->device) != VK_SUCCESS) {
    printf("  [FAIL] vkCreateDevice\n");
    return false;
  }
  return true;
}

static void cleanup_vulkan(VulkanContext* vk) {
  if (vk->device) vkDestroyDevice(vk->device, nullptr);
  if (vk->instance) vkDestroyInstance(vk->instance, nullptr);
}

// ============================================================================
// Pipeline buffer: GBM -> Vulkan -> CUDA
// ============================================================================
struct PipelineBuffer {
  struct gbm_bo* bo = nullptr;
  int dmabuf_fd = -1;
  uint32_t drm_handle = 0;
  uint32_t fb_id = 0;
  uint32_t stride = 0;
  VkImage vk_image = VK_NULL_HANDLE;
  VkDeviceMemory vk_memory = VK_NULL_HANDLE;
  CUexternalMemory cu_ext_mem = nullptr;
  CUdeviceptr cu_ptr = 0;
  uint64_t alloc_size = 0;
};

static bool create_pipeline_buffer(int drm_fd, struct gbm_device* gbm,
                                   VulkanContext* vk, PipelineBuffer* buf,
                                   uint32_t w, uint32_t h) {
  // 1. GBM alloc
  buf->bo = gbm_bo_create(gbm, w, h, GBM_FORMAT_ARGB8888,
                          GBM_BO_USE_SCANOUT | GBM_BO_USE_LINEAR);
  if (!buf->bo) {
    printf("  [FAIL] gbm_bo_create: %s\n", strerror(errno));
    return false;
  }

  buf->dmabuf_fd = gbm_bo_get_fd(buf->bo);
  buf->stride = gbm_bo_get_stride(buf->bo);
  uint32_t buf_size = buf->stride * h;

  if (buf->dmabuf_fd < 0) {
    printf("  [FAIL] gbm_bo_get_fd: %s\n", strerror(errno));
    return false;
  }
  printf("  GBM: bo=%p, dmabuf_fd=%d, stride=%u, size=%u\n",
         static_cast<void*>(buf->bo), buf->dmabuf_fd, buf->stride, buf_size);

  // Destroy the GBM bo -- dmabuf keeps the allocation alive
  gbm_bo_destroy(buf->bo);
  buf->bo = nullptr;

  int vulkan_fd = dup(buf->dmabuf_fd);
  if (vulkan_fd < 0) {
    printf("  [FAIL] dup(dmabuf_fd): %s\n", strerror(errno));
    return false;
  }

  // 2. Vulkan import dmabuf -> image -> memory
  VkExternalMemoryImageCreateInfo extImgInfo = {};
  extImgInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
  extImgInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

  VkImageCreateInfo imgInfo = {};
  imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imgInfo.pNext = &extImgInfo;
  imgInfo.imageType = VK_IMAGE_TYPE_2D;
  imgInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
  imgInfo.extent = {w, h, 1};
  imgInfo.mipLevels = 1;
  imgInfo.arrayLayers = 1;
  imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imgInfo.tiling = VK_IMAGE_TILING_LINEAR;
  imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  if (vkCreateImage(vk->device, &imgInfo, nullptr,
                    &buf->vk_image) != VK_SUCCESS) {
    printf("  [FAIL] vkCreateImage\n");
    close(vulkan_fd);
    return false;
  }

  VkMemoryRequirements memReqs;
  vkGetImageMemoryRequirements(vk->device, buf->vk_image, &memReqs);
  buf->alloc_size = memReqs.size;
  printf("  Vulkan: memReqs.size=%llu (gbm_size=%u)\n",
         (unsigned long long)memReqs.size, buf_size);

  VkImportMemoryFdInfoKHR importFd = {};
  importFd.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
  importFd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
  importFd.fd = vulkan_fd;

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.pNext = &importFd;
  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex = 0;

  if (vkAllocateMemory(vk->device, &allocInfo, nullptr,
                       &buf->vk_memory) != VK_SUCCESS) {
    printf("  [FAIL] vkAllocateMemory (dmabuf import)\n");
    vkDestroyImage(vk->device, buf->vk_image, nullptr);
    buf->vk_image = VK_NULL_HANDLE;
    return false;
  }
  vkBindImageMemory(vk->device, buf->vk_image, buf->vk_memory, 0);

  // 3. Vulkan export as opaque FD for CUDA
  VkMemoryGetFdInfoKHR getFdInfo = {};
  getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  getFdInfo.memory = buf->vk_memory;
  getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  auto vkGetMemoryFdKHR = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
      vkGetDeviceProcAddr(vk->device, "vkGetMemoryFdKHR"));
  if (!vkGetMemoryFdKHR) {
    printf("  [FAIL] vkGetDeviceProcAddr(vkGetMemoryFdKHR)\n");
    return false;
  }

  int cuda_fd = -1;
  if (vkGetMemoryFdKHR(vk->device, &getFdInfo, &cuda_fd) != VK_SUCCESS) {
    printf("  [FAIL] vkGetMemoryFdKHR\n");
    return false;
  }
  printf("  Vulkan: exported opaque fd=%d for CUDA\n", cuda_fd);

  // 4. CUDA import
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC extDesc = {};
  extDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
  extDesc.handle.fd = cuda_fd;
  extDesc.size = memReqs.size;

  CUresult err = cuImportExternalMemory(&buf->cu_ext_mem, &extDesc);
  if (err != CUDA_SUCCESS) {
    const char* errStr = "unknown";
    cuGetErrorString(err, &errStr);
    printf("  [FAIL] cuImportExternalMemory: %s\n", errStr);
    return false;
  }

  CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufDesc = {};
  bufDesc.size = memReqs.size;

  err = cuExternalMemoryGetMappedBuffer(&buf->cu_ptr, buf->cu_ext_mem,
                                        &bufDesc);
  if (err != CUDA_SUCCESS) {
    const char* errStr = "unknown";
    cuGetErrorString(err, &errStr);
    printf("  [FAIL] cuExternalMemoryGetMappedBuffer: %s\n", errStr);
    return false;
  }
  printf("  CUDA: ptr=0x%llx, size=%llu\n",
         (unsigned long long)buf->cu_ptr,
         (unsigned long long)memReqs.size);

  // 5. DRM framebuffer from original dmabuf
  if (drmPrimeFDToHandle(drm_fd, buf->dmabuf_fd, &buf->drm_handle) != 0) {
    printf("  [FAIL] drmPrimeFDToHandle: %s\n", strerror(errno));
    return false;
  }

  uint32_t handles[4] = {buf->drm_handle, 0, 0, 0};
  uint32_t strides[4] = {buf->stride, 0, 0, 0};
  uint32_t offsets[4] = {0, 0, 0, 0};
  if (drmModeAddFB2(drm_fd, w, h, DRM_FORMAT_ARGB8888,
                    handles, strides, offsets, &buf->fb_id, 0) != 0) {
    printf("  [FAIL] drmModeAddFB2: %s\n", strerror(errno));
    return false;
  }
  printf("  DRM: fb=%u, handle=%u\n", buf->fb_id, buf->drm_handle);

  return true;
}

static void destroy_pipeline_buffer(int drm_fd, VulkanContext* vk,
                                    PipelineBuffer* buf) {
  if (buf->cu_ext_mem) cuDestroyExternalMemory(buf->cu_ext_mem);
  if (buf->fb_id) drmModeRmFB(drm_fd, buf->fb_id);
  if (buf->vk_memory) vkFreeMemory(vk->device, buf->vk_memory, nullptr);
  if (buf->vk_image) vkDestroyImage(vk->device, buf->vk_image, nullptr);
  if (buf->dmabuf_fd >= 0) close(buf->dmabuf_fd);
  if (buf->bo) gbm_bo_destroy(buf->bo);
}

// ============================================================================
// Config
// ============================================================================
struct Config {
  int card_num = -1;
  std::string display_name;
  uint32_t mode_w = 0;
  uint32_t mode_h = 0;
  std::string fill = "cuda";
  int duration = 3;
};

static void print_help() {
  printf(
      "GBM -> Vulkan -> CUDA Buffer Pipeline Validation Test\n"
      "\n"
      "Validates the full buffer interop pipeline on this platform.\n"
      "\n"
      "Options:\n"
      "  --card <N>         DRI card number (default: auto)\n"
      "  --display <name>   Connector or monitor name\n"
      "  --mode <WxH>       Resolution (default: preferred)\n"
      "  --fill <cpu|cuda>  Fill method (default: cuda)\n"
      "  --duration <sec>   Hold duration (default: 3)\n"
      "  --help\n");
}

static bool parse_args(int argc, char** argv, Config* cfg) {
  static struct option long_opts[] = {
    {"card",     required_argument, nullptr, 'c'},
    {"display",  required_argument, nullptr, 'D'},
    {"mode",     required_argument, nullptr, 'm'},
    {"fill",     required_argument, nullptr, 'f'},
    {"duration", required_argument, nullptr, 'd'},
    {"help",     no_argument,       nullptr, 'h'},
    {nullptr, 0, nullptr, 0}
  };
  int opt;
  while ((opt = getopt_long(argc, argv, "hc:D:m:f:d:",
                            long_opts, nullptr)) != -1) {
    switch (opt) {
      case 'c': cfg->card_num = atoi(optarg); break;
      case 'D': cfg->display_name = optarg; break;
      case 'm':
        if (sscanf(optarg, "%ux%u", &cfg->mode_w, &cfg->mode_h) != 2) {
          printf("[ERROR] Invalid --mode. Use WxH.\n");
          return false;
        }
        break;
      case 'f': cfg->fill = optarg; break;
      case 'd': cfg->duration = atoi(optarg); break;
      case 'h': print_help(); exit(0);
      default: return false;
    }
  }
  return true;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
  printf("=========================================="
         "==========================================\n");
  printf("  GBM -> Vulkan -> CUDA Pipeline Test\n");
  printf("  Validates buffer interop pipeline on"
         " this platform.\n");
  printf("=========================================="
         "==========================================\n\n");

  Config cfg;
  if (!parse_args(argc, argv, &cfg)) return 1;

  if (cfg.fill != "cpu" && cfg.fill != "cuda") {
    printf("[ERROR] --fill must be 'cpu' or 'cuda'.\n");
    return 1;
  }

  // --- CUDA init ---
  printf("[TEST] Initializing CUDA...\n");
  CUresult err = cuInit(0);
  if (err != CUDA_SUCCESS) {
    const char* errStr = "unknown";
    cuGetErrorString(err, &errStr);
    printf("  [FAIL] cuInit: %s\n", errStr);
    return 1;
  }
  CUdevice cuDev;
  CUcontext cuCtx;
  cuDeviceGet(&cuDev, 0);
  err = create_cuda_context(&cuCtx, 0, cuDev);
  if (err != CUDA_SUCCESS) {
    const char* errStr = "unknown";
    cuGetErrorString(err, &errStr);
    printf("  [FAIL] cuCtxCreate: %s\n", errStr);
    return 1;
  }

  char devName[256];
  cuDeviceGetName(devName, sizeof(devName), cuDev);
  printf("  [PASS] CUDA device: %s\n\n", devName);

  // --- DRM ---
  printf("[TEST] Opening DRM device...\n");
  int fd = open_drm_device(cfg.card_num);
  if (fd < 0) { printf("  [FAIL] No DRM device.\n"); return 1; }
  printf("  [PASS] DRM fd=%d\n\n", fd);

  drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
  drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1);

  // --- GBM ---
  printf("[TEST] Creating GBM device...\n");
  struct gbm_device* gbm = gbm_create_device(fd);
  if (!gbm) { printf("  [FAIL] gbm_create_device\n"); close(fd); return 1; }
  printf("  [PASS] GBM device created.\n\n");

  // --- Vulkan ---
  printf("[TEST] Initializing Vulkan...\n");
  VulkanContext vk;
  if (!init_vulkan(&vk)) {
    gbm_device_destroy(gbm); close(fd); return 1;
  }
  VkPhysicalDeviceProperties vkProps;
  vkGetPhysicalDeviceProperties(vk.physicalDevice, &vkProps);
  printf("  [PASS] Vulkan device: %s\n\n", vkProps.deviceName);

  // --- Find connector + mode ---
  printf("[TEST] Finding display...\n");
  drmModeRes* res = drmModeGetResources(fd);
  drmModeConnector* conn = nullptr;
  std::string selected_monitor;
  if (res) {
    for (int i = 0; i < res->count_connectors; i++) {
      drmModeConnector* c = drmModeGetConnector(fd, res->connectors[i]);
      if (!c) continue;
      std::string cname = connector_name_str(c);
      std::string mname = get_edid_monitor_name(fd, c->connector_id);
      if (c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) {
        bool match = cfg.display_name.empty() ||
                     cfg.display_name == cname ||
                     (!mname.empty() && cfg.display_name == mname);
        if (match && !conn) {
          conn = c;
          selected_monitor = mname;
          continue;
        }
      }
      drmModeFreeConnector(c);
    }
  }
  if (!conn) {
    printf("  [FAIL] No display.\n");
    cleanup_vulkan(&vk); gbm_device_destroy(gbm);
    if (res) drmModeFreeResources(res);
    close(fd);
    return 1;
  }
  printf("  [PASS] %s", connector_name_str(conn).c_str());
  if (!selected_monitor.empty()) printf(" (%s)", selected_monitor.c_str());
  printf("\n");

  drmModeModeInfo* mode = nullptr;
  for (int i = 0; i < conn->count_modes; i++) {
    drmModeModeInfo* m = &conn->modes[i];
    if (cfg.mode_w > 0 && cfg.mode_h > 0) {
      if (m->hdisplay == cfg.mode_w && m->vdisplay == cfg.mode_h) {
        if (!mode || mode_refresh_hz(m) > mode_refresh_hz(mode)) mode = m;
      }
    } else if (m->type & DRM_MODE_TYPE_PREFERRED) {
      mode = m;
    }
  }
  if (!mode && cfg.mode_w == 0) mode = &conn->modes[0];
  if (!mode) {
    printf("  [FAIL] No matching mode.\n");
    printf("  Available modes:\n");
    for (int i = 0; i < conn->count_modes; i++) {
      drmModeModeInfo* m = &conn->modes[i];
      printf("    %ux%u @ %u Hz%s\n", m->hdisplay, m->vdisplay,
             mode_refresh_hz(m),
             (m->type & DRM_MODE_TYPE_PREFERRED) ? " (preferred)" : "");
    }
    drmModeFreeConnector(conn); drmModeFreeResources(res);
    cleanup_vulkan(&vk); gbm_device_destroy(gbm); close(fd);
    return 1;
  }
  printf("  Mode: %ux%u @ %u Hz\n\n",
         mode->hdisplay, mode->vdisplay, mode_refresh_hz(mode));

  // CRTC + primary plane
  uint32_t crtc_id = 0;
  if (conn->encoder_id) {
    drmModeEncoder* enc = drmModeGetEncoder(fd, conn->encoder_id);
    if (enc) { crtc_id = enc->crtc_id; drmModeFreeEncoder(enc); }
  }
  if (!crtc_id) {
    for (int i = 0; i < res->count_crtcs; i++) {
      if (conn->count_encoders > 0) {
        drmModeEncoder* enc = drmModeGetEncoder(fd, conn->encoders[0]);
        if (enc && (enc->possible_crtcs & (1u << i))) {
          crtc_id = res->crtcs[i];
          drmModeFreeEncoder(enc);
          break;
        }
        if (enc) drmModeFreeEncoder(enc);
      }
    }
  }

  int crtc_index = -1;
  for (int i = 0; i < res->count_crtcs; i++)
    if (res->crtcs[i] == crtc_id) { crtc_index = i; break; }

  uint32_t primary_plane = 0;
  drmModePlaneRes* pr = drmModeGetPlaneResources(fd);
  if (pr) {
    for (uint32_t i = 0; i < pr->count_planes; i++) {
      drmModePlane* p = drmModeGetPlane(fd, pr->planes[i]);
      if (!p) continue;
      if (crtc_index >= 0 && !(p->possible_crtcs & (1u << crtc_index))) {
        drmModeFreePlane(p); continue;
      }
      if (get_plane_type(fd, p->plane_id) == 1) {
        primary_plane = p->plane_id;
        drmModeFreePlane(p); break;
      }
      drmModeFreePlane(p);
    }
    drmModeFreePlaneResources(pr);
  }

  if (!crtc_id || !primary_plane) {
    printf("[FAIL] No CRTC/primary plane.\n");
    drmModeFreeConnector(conn); drmModeFreeResources(res);
    cleanup_vulkan(&vk); gbm_device_destroy(gbm); close(fd);
    return 1;
  }

  // --- Create pipeline buffer ---
  uint32_t w = mode->hdisplay;
  uint32_t h = mode->vdisplay;
  printf("[TEST] Creating pipeline buffer (GBM -> Vulkan -> CUDA)...\n");
  PipelineBuffer pbuf;
  if (!create_pipeline_buffer(fd, gbm, &vk, &pbuf, w, h)) {
    printf("  [FAIL] Pipeline buffer creation failed.\n");
    drmModeFreeConnector(conn); drmModeFreeResources(res);
    cleanup_vulkan(&vk); gbm_device_destroy(gbm); close(fd);
    return 1;
  }
  printf("  [PASS] Full pipeline buffer created.\n\n");

  // --- Fill buffer ---
  // ARGB8888: green = 0xFF00FF00
  printf("[TEST] Filling buffer via %s...\n", cfg.fill.c_str());
  if (cfg.fill == "cuda") {
    // cuMemsetD32 fills 32-bit values; ARGB8888 green = 0xFF00FF00
    err = cuMemsetD32(pbuf.cu_ptr, 0xFF00FF00,
                      (pbuf.stride / 4) * h);
    if (err != CUDA_SUCCESS) {
      const char* errStr = "unknown";
      cuGetErrorString(err, &errStr);
      printf("  [FAIL] cuMemsetD32: %s\n", errStr);
    } else {
      err = cuCtxSynchronize();
      if (err != CUDA_SUCCESS) {
        const char* errStr = "unknown";
        cuGetErrorString(err, &errStr);
        printf("  [FAIL] cuCtxSynchronize: %s\n", errStr);
      } else {
        printf("  [PASS] CUDA fill complete (green 0xFF00FF00).\n");
      }
    }
  } else {
    // CPU fill: allocate host buffer, fill, copy to device
    uint32_t num_pixels = (pbuf.stride / 4) * h;
    uint32_t* host_buf = static_cast<uint32_t*>(
        malloc(num_pixels * sizeof(uint32_t)));
    if (!host_buf) {
      printf("  [FAIL] malloc for host buffer.\n");
    } else {
      for (uint32_t i = 0; i < num_pixels; i++)
        host_buf[i] = 0xFF00FF00;
      err = cuMemcpyHtoD(pbuf.cu_ptr, host_buf,
                         num_pixels * sizeof(uint32_t));
      free(host_buf);
      if (err != CUDA_SUCCESS) {
        const char* errStr = "unknown";
        cuGetErrorString(err, &errStr);
        printf("  [FAIL] cuMemcpyHtoD: %s\n", errStr);
      } else {
        cuCtxSynchronize();
        printf("  [PASS] CPU fill complete (green 0xFF00FF00).\n");
      }
    }
  }
  printf("\n");

  // --- Modeset + display ---
  printf("[TEST] Setting mode and displaying...\n");
  if (drmModeSetCrtc(fd, crtc_id, pbuf.fb_id, 0, 0,
                     &conn->connector_id, 1, mode) != 0) {
    printf("  [FAIL] drmModeSetCrtc: %s\n", strerror(errno));
  } else {
    printf("  [PASS] CRTC modeset succeeded.\n");
  }

  drmModeAtomicReqPtr req = drmModeAtomicAlloc();
  auto add_prop = [&](const char* name, uint64_t val) {
    uint32_t pid = find_plane_prop(fd, primary_plane, name);
    if (pid) drmModeAtomicAddProperty(req, primary_plane, pid, val);
  };
  add_prop("FB_ID", pbuf.fb_id);
  add_prop("CRTC_ID", crtc_id);
  add_prop("SRC_X", 0);
  add_prop("SRC_Y", 0);
  add_prop("SRC_W", static_cast<uint64_t>(w) << 16);
  add_prop("SRC_H", static_cast<uint64_t>(h) << 16);
  add_prop("CRTC_X", 0);
  add_prop("CRTC_Y", 0);
  add_prop("CRTC_W", w);
  add_prop("CRTC_H", h);

  int ret = drmModeAtomicCommit(fd, req,
                                DRM_MODE_ATOMIC_ALLOW_MODESET, nullptr);
  drmModeAtomicFree(req);
  if (ret != 0) {
    printf("  [FAIL] Atomic commit: %s\n", strerror(errno));
  } else {
    printf("  [PASS] Atomic commit succeeded.\n");
  }
  printf("\n");

  // --- Summary ---
  printf("==========================================\n");
  printf("  Pipeline Test Results\n");
  printf("==========================================\n");
  printf("  Mode:       %ux%u @ %u Hz\n",
         mode->hdisplay, mode->vdisplay, mode_refresh_hz(mode));
  printf("  Connector:  %s\n", connector_name_str(conn).c_str());
  if (!selected_monitor.empty())
    printf("  Monitor:    %s\n", selected_monitor.c_str());
  printf("  CUDA dev:   %s\n", devName);
  printf("  Vulkan dev: %s\n", vkProps.deviceName);
  printf("  Fill:       %s\n", cfg.fill.c_str());
  printf("  Buffer:     stride=%u, cuda_ptr=0x%llx, fb=%u\n",
         pbuf.stride, (unsigned long long)pbuf.cu_ptr, pbuf.fb_id);
  printf("==========================================\n\n");

  printf(">>> A GREEN screen should be visible on the display.     <<<\n");
  printf(">>> If black or corrupt, the interop pipeline has issues. <<<\n");
  printf("\n");
  printf("Holding display for %d seconds...\n", cfg.duration);
  sleep(cfg.duration);

  // --- Cleanup ---
  printf("\n[INFO] Cleaning up...\n");
  destroy_pipeline_buffer(fd, &vk, &pbuf);
  drmModeFreeConnector(conn);
  drmModeFreeResources(res);
  cleanup_vulkan(&vk);
  gbm_device_destroy(gbm);
  close(fd);
  cuCtxDestroy(cuCtx);

  printf("[DONE] Pipeline validation test completed.\n");
  return 0;
}
