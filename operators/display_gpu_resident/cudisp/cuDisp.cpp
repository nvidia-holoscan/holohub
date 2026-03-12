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

#include "cuDisp.h"
#include "cuDispDeviceInternal.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <atomic>

#include <cuda.h>
#include <drm_fourcc.h>
#include <gbm.h>
#include <vulkan/vulkan.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#define MAX_PLANES_PER_CRTC 10
#define MAX_BUFFERS 10

/* Set to 1 to enable verbose init/present debug prints */
#ifndef CUDISP_DEBUG
#define CUDISP_DEBUG 0
#endif
/* Set to 1 to enable present-thread benchmarking (timing, counters, env CUDISP_BENCH_*) */
#ifndef CUDISP_BENCH
#define CUDISP_BENCH 0
#endif

// Internal plane type enumeration
typedef enum {
  CUDISP_PLANE_TYPE_PRIMARY = 0,
  CUDISP_PLANE_TYPE_OVERLAY,
  CUDISP_PLANE_TYPE_CURSOR,
  CUDISP_PLANE_TYPE_COUNT
} cuDispPlaneType;

// Internal structures
struct cuDispPlane {
  drmModePlanePtr info;
  cuDispPlaneType type;
};

struct cuDispCrtc {
  drmModeCrtcPtr info;
  struct cuDispPlane planes[MAX_PLANES_PER_CRTC];
  int num_planes;
};

struct cuDispConnector {
  drmModeConnectorPtr info;
  struct cuDispCrtc crtc;
};

// Cached property IDs for fast atomic commits
struct CachedPlaneProps {
  uint32_t plane_id;
  uint32_t fb_id_prop;
  uint32_t crtc_id_prop;
  uint32_t crtc_x_prop;
  uint32_t crtc_y_prop;
  uint32_t crtc_w_prop;
  uint32_t crtc_h_prop;
  uint32_t src_x_prop;
  uint32_t src_y_prop;
  uint32_t src_w_prop;
  uint32_t src_h_prop;
  uint32_t alpha_prop;
};

// Vulkan context
struct VulkanContext {
  VkInstance instance;
  VkPhysicalDevice physicalDevice;
  VkDevice device;
  uint32_t queueFamilyIndex;
};

static void releaseImportedCudaBuffer(CUdeviceptr* d_ptr, CUexternalMemory* ext_mem, int* dmabuf_fd) {
  // CUDA requires the mapped buffer to be freed before releasing the external memory object.
  if (d_ptr && *d_ptr) {
    cuMemFree(*d_ptr);
    *d_ptr = 0;
  }
  if (ext_mem && *ext_mem) {
    cuDestroyExternalMemory(*ext_mem);
    *ext_mem = nullptr;
  }
  if (dmabuf_fd && *dmabuf_fd >= 0) {
    close(*dmabuf_fd);
    *dmabuf_fd = -1;
  }
}

// Main swapchain context structure
struct cuDispSwapchain_t {
  // DRM objects
  int fd;
  uint32_t crtc_id;
  uint32_t connector_id;
  struct cuDispConnector connector;
  drmModeModeInfoPtr mode;
  drmModeRes* res_info;

  // Buffer management
  uint32_t num_buffers;
  CUdeviceptr d_ptrs[MAX_BUFFERS];
  int dmabuf_fds[MAX_BUFFERS];
  uint32_t fb_ids[MAX_BUFFERS];
  CUexternalMemory ext_mems[MAX_BUFFERS];

  // Vulkan/GBM
  struct VulkanContext vk_ctx;
  struct gbm_device* gbm;

  // Cached for fast flips
  struct CachedPlaneProps cached_props;
  uint32_t src_w_fixed;  // hdisplay << 16
  uint32_t src_h_fixed;  // vdisplay << 16

  // VRR support
  uint32_t vrr_enabled_prop_id;
  bool vrr_supported;
  bool vrr_capable;
  bool enable_vrr;

  // State tracking
  bool first_flip;
  int conn_index;

  // Present thread (for continuous flip mode)
  pthread_t present_thread;
  std::atomic<bool> exit_thread;
  bool thread_started;
  bool front_buffer_render;
  cuDispSwapchainGPU* swapchain_gpu;

  // Configuration (parsed from attributes)
  uint32_t width;
  uint32_t height;
  uint32_t refresh_rate_millihz;  // Stored in milliHz
  uint32_t drm_format;            // DRM fourcc code
  uint32_t stride;                // Row pitch in bytes (from GBM)
  uint64_t buffer_size;           // Size of each buffer in bytes
};

//=============================================================================
// Internal Helper Functions
//=============================================================================

static uint32_t mode_refresh_millihz(const drmModeModeInfo* m) {
  if (m->htotal == 0 || m->vtotal == 0)
    return 0;
  return (uint32_t)((uint64_t)m->clock * 1000000ULL / ((uint64_t)m->htotal * m->vtotal));
}

static int getDrmPlaneType(int drmFd, uint32_t objectID) {
  uint32_t i;
  int j;
  uint64_t value = 0;
  int planeType = -1;

  drmModeObjectPropertiesPtr ModeObjectProperties =
      drmModeObjectGetProperties(drmFd, objectID, DRM_MODE_OBJECT_PLANE);
  if (!ModeObjectProperties) {
    return planeType;
  }

  for (i = 0; i < ModeObjectProperties->count_props; i++) {
    drmModePropertyPtr Property = drmModeGetProperty(drmFd, ModeObjectProperties->props[i]);

    if (Property == NULL) {
      continue;
    }
    if (strcmp("type", Property->name) == 0) {
      value = ModeObjectProperties->prop_values[i];
      for (j = 0; j < Property->count_enums; j++) {
        if (value == (Property->enums[j]).value) {
          if (strcmp("Primary", (Property->enums[j]).name) == 0) {
            planeType = CUDISP_PLANE_TYPE_PRIMARY;
          } else if (strcmp("Overlay", (Property->enums[j]).name) == 0) {
            planeType = CUDISP_PLANE_TYPE_OVERLAY;
          } else if (strcmp("Cursor", (Property->enums[j]).name) == 0) {
            planeType = CUDISP_PLANE_TYPE_CURSOR;
          }
        }
      }
      drmModeFreeProperty(Property);
      break;
    }
    drmModeFreeProperty(Property);
  }
  drmModeFreeObjectProperties(ModeObjectProperties);

  return planeType;
}

static int get_connector_crtc_plane_info(struct cuDispSwapchain_t* swapchain) {
  drmModeConnector* conn = NULL;
  drmModeCrtc* crtc = NULL;
  drmModePlaneRes* plane_res = NULL;
  int crtc_index = -1;
  int i = 0;
  int found = 0;

  swapchain->res_info = drmModeGetResources(swapchain->fd);
  if (!swapchain->res_info) {
    printf("Failed to get DRM resources\n");
    return 0;
  }

#if CUDISP_DEBUG
  printf("[cuDisp] Found %d connectors\n", swapchain->res_info->count_connectors);
#endif

  for (i = 0; i < swapchain->res_info->count_connectors; i++) {
    conn = drmModeGetConnector(swapchain->fd, swapchain->res_info->connectors[i]);
    if (!conn) {
      continue;
    }

#if CUDISP_DEBUG
    printf("[cuDisp] Connector %d: ID=%u, type=%u, connection=%s\n",
           i,
           conn->connector_id,
           conn->connector_type,
           (conn->connection == DRM_MODE_CONNECTED)      ? "CONNECTED"
           : (conn->connection == DRM_MODE_DISCONNECTED) ? "DISCONNECTED"
                                                         : "UNKNOWN");
#endif
    if ((conn->count_modes > 0) && (conn->connection == DRM_MODE_CONNECTED)) {
      found = 1;
      swapchain->conn_index = i;
#if CUDISP_DEBUG
      printf("[cuDisp] Using connector %d (ID=%u) - has %d modes\n",
             i,
             conn->connector_id,
             conn->count_modes);
#endif
      break;
    }
    drmModeFreeConnector(conn);
    conn = NULL;
  }

  if (!found || !conn) {
    printf("Suitable connector not found\n");
    return 0;
  }

  swapchain->connector.info = conn;

  if (conn->encoder_id) {
    drmModeEncoder* enc = drmModeGetEncoder(swapchain->fd, conn->encoder_id);
    if (enc && enc->crtc_id) {
      crtc = drmModeGetCrtc(swapchain->fd, enc->crtc_id);
#if CUDISP_DEBUG
      printf("[cuDisp] Found CRTC via encoder: CRTC_ID=%u\n", enc->crtc_id);
#endif
    }
    if (enc)
      drmModeFreeEncoder(enc);
  }

  if (!crtc && swapchain->res_info->count_crtcs > 0) {
    crtc = drmModeGetCrtc(swapchain->fd, swapchain->res_info->crtcs[0]);
#if CUDISP_DEBUG
    printf("[cuDisp] Using first CRTC: CRTC_ID=%u\n", swapchain->res_info->crtcs[0]);
#endif
  }

  if (!crtc) {
    printf("CRTC not found\n");
    return 0;
  }

  for (i = 0; i < swapchain->res_info->count_crtcs; i++) {
    if (swapchain->res_info->crtcs[i] == crtc->crtc_id) {
      crtc_index = i;
      break;
    }
  }
  if (crtc_index < 0) {
    printf("Failed to find CRTC index for CRTC_ID=%u\n", crtc->crtc_id);
    drmModeFreeCrtc(crtc);
    return 0;
  }

  swapchain->connector.crtc.info = crtc;
  swapchain->connector.crtc.num_planes = 0;

  plane_res = drmModeGetPlaneResources(swapchain->fd);
  if (!plane_res) {
    printf("Failed to get plane resources\n");
    return 0;
  }

#if CUDISP_DEBUG
  printf("[cuDisp] Found %d planes\n", plane_res->count_planes);
#endif

  for (i = 0; i < (int)plane_res->count_planes; i++) {
    drmModePlane* plane = drmModeGetPlane(swapchain->fd, plane_res->planes[i]);
    if (!plane) {
      continue;
    }

    if (plane->possible_crtcs & (1u << crtc_index)) {
      int plane_type = getDrmPlaneType(swapchain->fd, plane->plane_id);
#if CUDISP_DEBUG
      const char* type_str = (plane_type == CUDISP_PLANE_TYPE_PRIMARY)   ? "PRIMARY"
                             : (plane_type == CUDISP_PLANE_TYPE_OVERLAY) ? "OVERLAY"
                                                                         : "CURSOR";
      printf("[cuDisp] Plane %d: ID=%u, type=%s, possible for CRTC index %d\n",
             i,
             plane->plane_id,
             type_str,
             crtc_index);
#endif

      if (swapchain->connector.crtc.num_planes < MAX_PLANES_PER_CRTC) {
        swapchain->connector.crtc.planes[swapchain->connector.crtc.num_planes].info = plane;
        swapchain->connector.crtc.planes[swapchain->connector.crtc.num_planes].type =
            (cuDispPlaneType)plane_type;
        swapchain->connector.crtc.num_planes++;
      } else {
        drmModeFreePlane(plane);
      }
    } else {
      drmModeFreePlane(plane);
    }
  }

  drmModeFreePlaneResources(plane_res);
  return 1;
}

static int drm_cache_plane_properties(struct cuDispSwapchain_t* swapchain, uint32_t plane_id) {
  drmModeObjectPropertiesPtr props =
      drmModeObjectGetProperties(swapchain->fd, plane_id, DRM_MODE_OBJECT_PLANE);
  if (!props) {
    return 0;
  }

  swapchain->cached_props.plane_id = plane_id;

  for (uint32_t i = 0; i < props->count_props; i++) {
    drmModePropertyPtr prop = drmModeGetProperty(swapchain->fd, props->props[i]);
    if (!prop)
      continue;

    if (strcmp(prop->name, "FB_ID") == 0) {
      swapchain->cached_props.fb_id_prop = prop->prop_id;
    } else if (strcmp(prop->name, "CRTC_ID") == 0) {
      swapchain->cached_props.crtc_id_prop = prop->prop_id;
    } else if (strcmp(prop->name, "CRTC_X") == 0) {
      swapchain->cached_props.crtc_x_prop = prop->prop_id;
    } else if (strcmp(prop->name, "CRTC_Y") == 0) {
      swapchain->cached_props.crtc_y_prop = prop->prop_id;
    } else if (strcmp(prop->name, "CRTC_W") == 0) {
      swapchain->cached_props.crtc_w_prop = prop->prop_id;
    } else if (strcmp(prop->name, "CRTC_H") == 0) {
      swapchain->cached_props.crtc_h_prop = prop->prop_id;
    } else if (strcmp(prop->name, "SRC_X") == 0) {
      swapchain->cached_props.src_x_prop = prop->prop_id;
    } else if (strcmp(prop->name, "SRC_Y") == 0) {
      swapchain->cached_props.src_y_prop = prop->prop_id;
    } else if (strcmp(prop->name, "SRC_W") == 0) {
      swapchain->cached_props.src_w_prop = prop->prop_id;
    } else if (strcmp(prop->name, "SRC_H") == 0) {
      swapchain->cached_props.src_h_prop = prop->prop_id;
    } else if (strcmp(prop->name, "alpha") == 0) {
      swapchain->cached_props.alpha_prop = prop->prop_id;
    }

    drmModeFreeProperty(prop);
  }

  drmModeFreeObjectProperties(props);
  return 1;
}

static void drm_check_vrr_support(struct cuDispSwapchain_t* swapchain) {
  swapchain->vrr_supported = false;
  swapchain->vrr_capable = false;

  // Check CRTC for VRR_ENABLED property
  drmModeObjectPropertiesPtr crtc_props =
      drmModeObjectGetProperties(swapchain->fd, swapchain->crtc_id, DRM_MODE_OBJECT_CRTC);

  if (crtc_props) {
    for (uint32_t i = 0; i < crtc_props->count_props; i++) {
      drmModePropertyPtr prop = drmModeGetProperty(swapchain->fd, crtc_props->props[i]);
      if (prop && strcmp(prop->name, "VRR_ENABLED") == 0) {
        swapchain->vrr_enabled_prop_id = prop->prop_id;
        swapchain->vrr_supported = true;
#if CUDISP_DEBUG
        printf("[cuDisp] VRR_ENABLED property found on CRTC (prop_id=%u)\n", prop->prop_id);
#endif
      }
      if (prop)
        drmModeFreeProperty(prop);
    }
    drmModeFreeObjectProperties(crtc_props);
  }

  // Check Connector for vrr_capable property
  drmModeObjectPropertiesPtr conn_props =
      drmModeObjectGetProperties(swapchain->fd, swapchain->connector_id, DRM_MODE_OBJECT_CONNECTOR);

  if (conn_props) {
    for (uint32_t i = 0; i < conn_props->count_props; i++) {
      drmModePropertyPtr prop = drmModeGetProperty(swapchain->fd, conn_props->props[i]);
      if (prop && strcmp(prop->name, "vrr_capable") == 0) {
        swapchain->vrr_capable = (conn_props->prop_values[i] != 0);
#if CUDISP_DEBUG
        printf("[cuDisp] Connector vrr_capable=%d\n", swapchain->vrr_capable);
#endif
      }
      if (prop)
        drmModeFreeProperty(prop);
    }
    drmModeFreeObjectProperties(conn_props);
  }

  if (swapchain->vrr_supported && swapchain->vrr_capable) {
    printf("VRR/G-Sync is supported and available\n");
  } else if (swapchain->vrr_supported) {
    printf("VRR/G-Sync property exists but display is not capable\n");
  } else {
    printf("VRR/G-Sync is not supported\n");
  }
}

// Returns 0 if the DRM fd has usable KMS resources (e.g. at least one connector).
static int drm_util_mode_get_resources(int fd) {
  drmModeRes* res_info = drmModeGetResources(fd);
  if (!res_info) {
    return -1;
  }
  if (res_info->count_connectors < 1) {
    drmModeFreeResources(res_info);
    return -1;
  }
  drmModeFreeResources(res_info);
  return 0;
}

static int drm_util_find_device(char* buf, size_t bufsize) {
  DIR* dir = opendir("/dev/dri");
  if (!dir) {
    return -1;
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != NULL) {
    if (strncmp(entry->d_name, "card", 4) != 0) {
      continue;
    }

    char path[256];
    snprintf(path, sizeof(path), "/dev/dri/%.*s", 246, entry->d_name);

    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
      continue;
    }

    if (drm_util_mode_get_resources(fd) != 0) {
      close(fd);
      continue;
    }

    close(fd);
    snprintf(buf, bufsize, "/dev/dri/%.*s",
             static_cast<int>(bufsize - 11), entry->d_name);
    closedir(dir);
    return 0;
  }

  closedir(dir);
  return -1;
}

static int initVulkanContext(struct VulkanContext* vkCtx) {
  vkCtx->instance = VK_NULL_HANDLE;
  vkCtx->physicalDevice = VK_NULL_HANDLE;
  vkCtx->device = VK_NULL_HANDLE;
  vkCtx->queueFamilyIndex = 0;

  // 1. Create Vulkan instance
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "cuDisp";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  const char* instanceExtensions[] = {VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME};

  VkInstanceCreateInfo instanceInfo = {};
  instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceInfo.pApplicationInfo = &appInfo;
  instanceInfo.enabledExtensionCount = 1;
  instanceInfo.ppEnabledExtensionNames = instanceExtensions;

  if (vkCreateInstance(&instanceInfo, nullptr, &vkCtx->instance) != VK_SUCCESS) {
    printf("Failed to create Vulkan instance\n");
    return -1;
  }

  // 2. Pick physical device
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(vkCtx->instance, &deviceCount, nullptr);
  if (deviceCount == 0) {
    printf("No Vulkan physical devices found\n");
    vkDestroyInstance(vkCtx->instance, nullptr);
    vkCtx->instance = VK_NULL_HANDLE;
    return -1;
  }

  VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(deviceCount * sizeof(VkPhysicalDevice));
  if (!devices) {
    printf("Failed to allocate memory for Vulkan physical devices\n");
    vkDestroyInstance(vkCtx->instance, nullptr);
    vkCtx->instance = VK_NULL_HANDLE;
    return -1;
  }
  vkEnumeratePhysicalDevices(vkCtx->instance, &deviceCount, devices);
  vkCtx->physicalDevice = devices[0];
  free(devices);

  // 3. Find queue family
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(vkCtx->physicalDevice, &queueFamilyCount, nullptr);
  VkQueueFamilyProperties* queueFamilies =
      (VkQueueFamilyProperties*)malloc(queueFamilyCount * sizeof(VkQueueFamilyProperties));
  if (!queueFamilies) {
    printf("Failed to allocate memory for Vulkan queue families\n");
    vkDestroyInstance(vkCtx->instance, nullptr);
    vkCtx->instance = VK_NULL_HANDLE;
    return -1;
  }
  vkGetPhysicalDeviceQueueFamilyProperties(vkCtx->physicalDevice, &queueFamilyCount, queueFamilies);
  vkCtx->queueFamilyIndex = 0;
  free(queueFamilies);

  // 4. Create logical device
  const char* deviceExtensions[] = {VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
                                    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                                    VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME};

  float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo queueInfo = {};
  queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueInfo.queueFamilyIndex = vkCtx->queueFamilyIndex;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &queuePriority;

  VkDeviceCreateInfo deviceInfo = {};
  deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceInfo.queueCreateInfoCount = 1;
  deviceInfo.pQueueCreateInfos = &queueInfo;
  deviceInfo.enabledExtensionCount = 3;
  deviceInfo.ppEnabledExtensionNames = deviceExtensions;

  if (vkCreateDevice(vkCtx->physicalDevice, &deviceInfo, nullptr, &vkCtx->device) != VK_SUCCESS) {
    printf("Failed to create Vulkan device\n");
    vkDestroyInstance(vkCtx->instance, nullptr);
    vkCtx->instance = VK_NULL_HANDLE;
    return -1;
  }

  return 0;
}

// Supported formats in this implementation (all others return NotSupported)
static int isSupportedSurfaceFormat(cuDispSurfaceFormat format) {
  return (format == CUDISP_SURFACE_FORMAT_ARGB8888 ||
          format == CUDISP_SURFACE_FORMAT_XRGB8888 ||
          format == CUDISP_SURFACE_FORMAT_ABGR16161616);
}

// Map cuDisp surface format to DRM fourcc code (only for supported formats)
static uint32_t mapSurfaceFormatToDrmFourcc(cuDispSurfaceFormat format) {
  switch (format) {
    case CUDISP_SURFACE_FORMAT_ARGB8888:
      return DRM_FORMAT_ARGB8888;
    case CUDISP_SURFACE_FORMAT_XRGB8888:
      return DRM_FORMAT_XRGB8888;
    case CUDISP_SURFACE_FORMAT_ABGR16161616:
      return DRM_FORMAT_ABGR16161616;
    default:
      return DRM_FORMAT_ARGB8888;
  }
}

// Map cuDisp surface format to GBM format
static uint32_t mapSurfaceFormatToGbmFormat(cuDispSurfaceFormat format) {
  switch (format) {
    case CUDISP_SURFACE_FORMAT_ARGB8888:
      return GBM_FORMAT_ARGB8888;
    case CUDISP_SURFACE_FORMAT_XRGB8888:
      return GBM_FORMAT_XRGB8888;
    case CUDISP_SURFACE_FORMAT_ABGR16161616:
      return GBM_FORMAT_ABGR16161616;
    default:
      return GBM_FORMAT_ARGB8888;
  }
}

// Map cuDisp surface format to Vulkan format
static VkFormat mapSurfaceFormatToVulkanFormat(cuDispSurfaceFormat format) {
  switch (format) {
    case CUDISP_SURFACE_FORMAT_ARGB8888:
      return VK_FORMAT_B8G8R8A8_UNORM;
    case CUDISP_SURFACE_FORMAT_XRGB8888:
      return VK_FORMAT_B8G8R8A8_UNORM;
    case CUDISP_SURFACE_FORMAT_ABGR16161616:
      return VK_FORMAT_R16G16B16A16_UNORM;
    default:
      return VK_FORMAT_B8G8R8A8_UNORM;
  }
}

// Calculate bytes per pixel for format
static uint32_t getBytesPerPixel(cuDispSurfaceFormat format) {
  switch (format) {
    case CUDISP_SURFACE_FORMAT_ARGB8888:
    case CUDISP_SURFACE_FORMAT_XRGB8888:
      return 4;
    case CUDISP_SURFACE_FORMAT_ABGR16161616:
      return 8;
    default:
      return 4;
  }
}

static int createBufferGbmVulkanCuda(struct gbm_device* gbm_dev, struct VulkanContext* vkCtx,
                                     int width, int height, cuDispSurfaceFormat surfaceFormat,
                                     CUdeviceptr* d_ptr, int* dmabuf_fd, CUexternalMemory* extMem,
                                     uint32_t* out_stride) {
  uint32_t gbmFormat = mapSurfaceFormatToGbmFormat(surfaceFormat);
  VkFormat vkFormat = mapSurfaceFormatToVulkanFormat(surfaceFormat);

  // 1. Allocate GBM buffer with SCANOUT flag
  struct gbm_bo* bo =
      gbm_bo_create(gbm_dev, width, height, gbmFormat, GBM_BO_USE_SCANOUT | GBM_BO_USE_LINEAR);

  if (!bo) {
    printf("Failed to create GBM buffer\n");
    return -1;
  }

  // 2. Export dmabuf fd from GBM
  *dmabuf_fd = gbm_bo_get_fd(bo);

  uint32_t bytesPerPixel = getBytesPerPixel(surfaceFormat);
  uint32_t gbmStride = gbm_bo_get_stride(bo);
  uint32_t calculatedStride = width * bytesPerPixel;
  uint32_t bufferSize = gbmStride * height;

#if CUDISP_DEBUG
  printf("[cuDisp] GBM buffer: expected=%u, dmabuf fd=%d\n", width * bytesPerPixel, *dmabuf_fd);
#endif

  if (gbmStride != calculatedStride) {
    printf("[cuDisp] WARNING: GBM stride (%u) differs from calculated (%u = %d * %u)\n",
           gbmStride,
           calculatedStride,
           width,
           bytesPerPixel);
  }
#if CUDISP_DEBUG
  printf("[cuDisp] GBM buffer: %dx%d, stride=%u (calculated=%u), size=%u, dmabuf fd=%d\n",
         width,
         height,
         gbmStride,
         calculatedStride,
         bufferSize,
         *dmabuf_fd);
#endif

  if (out_stride) {
    *out_stride = gbmStride;
  }

  if (*dmabuf_fd < 0) {
    printf("Failed to export GBM buffer to dmabuf fd\n");
    gbm_bo_destroy(bo);
    return -1;
  }

  gbm_bo_destroy(bo);
  bo = NULL;

  // Duplicate FD for Vulkan
  int vulkan_fd = dup(*dmabuf_fd);
  if (vulkan_fd < 0) {
    printf("Failed to duplicate dmabuf fd\n");
    close(*dmabuf_fd);
    return -1;
  }

  // 3. Import dmabuf fd into Vulkan
  VkExternalMemoryImageCreateInfo externalMemInfo = {};
  externalMemInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
  externalMemInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

  VkImageCreateInfo imageInfo = {};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.pNext = &externalMemInfo;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.format = vkFormat;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
  imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VkImage vkImage;
  if (vkCreateImage(vkCtx->device, &imageInfo, nullptr, &vkImage) != VK_SUCCESS) {
    printf("Failed to create Vulkan image\n");
    close(vulkan_fd);
    close(*dmabuf_fd);
    return -1;
  }

  // Get memory requirements
  VkMemoryRequirements memReqs;
  vkGetImageMemoryRequirements(vkCtx->device, vkImage, &memReqs);

  if (memReqs.size != bufferSize) {
    printf(
        "[cuDisp warning] NOTE: Vulkan memReqs.size=%llu differs from bufferSize=%u "
        "(stride=%u, height=%d)\n",
        (unsigned long long)memReqs.size,
        bufferSize,
        gbmStride,
        height);
  }

  // Import dmabuf fd as Vulkan memory
  VkImportMemoryFdInfoKHR importInfo = {};
  importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
  importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
  importInfo.fd = vulkan_fd;

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.pNext = &importInfo;
  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex = 0;

  VkDeviceMemory vkMemory;
  if (vkAllocateMemory(vkCtx->device, &allocInfo, nullptr, &vkMemory) != VK_SUCCESS) {
    printf("Failed to import dmabuf into Vulkan memory\n");
    vkDestroyImage(vkCtx->device, vkImage, nullptr);
    close(vulkan_fd);
    close(*dmabuf_fd);
    return -1;
  }

  // After a successful import, Vulkan owns vulkan_fd and closes it via vkFreeMemory.
  vkBindImageMemory(vkCtx->device, vkImage, vkMemory, 0);

  // 4. Export Vulkan memory as new fd for CUDA
  VkMemoryGetFdInfoKHR getFdInfo = {};
  getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  getFdInfo.memory = vkMemory;
  getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  int cuda_fd;
  PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR =
      (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(vkCtx->device, "vkGetMemoryFdKHR");

  if (!vkGetMemoryFdKHR ||
      vkGetMemoryFdKHR(vkCtx->device, &getFdInfo, &cuda_fd) != VK_SUCCESS) {
    printf("Failed to export Vulkan memory to fd\n");
    vkFreeMemory(vkCtx->device, vkMemory, nullptr);
    vkDestroyImage(vkCtx->device, vkImage, nullptr);
    close(*dmabuf_fd);
    return -1;
  }

  // 5. Import fd into CUDA using actual Vulkan allocation size
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC externalMemDesc = {};
  externalMemDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
  externalMemDesc.handle.fd = cuda_fd;
  externalMemDesc.size = memReqs.size;
  externalMemDesc.flags = 0;

  // CUDA takes ownership of cuda_fd and closes it before this call returns.
  CUresult cuErr = cuImportExternalMemory(extMem, &externalMemDesc);
  if (cuErr != CUDA_SUCCESS) {
    const char* errStr;
    cuGetErrorString(cuErr, &errStr);
    printf("Failed to import memory into CUDA: %s (memReqs.size=%llu)\n",
           errStr,
           (unsigned long long)memReqs.size);
    vkFreeMemory(vkCtx->device, vkMemory, nullptr);
    vkDestroyImage(vkCtx->device, vkImage, nullptr);
    close(*dmabuf_fd);
    return -1;
  }

  // Map the buffer using the actual allocation size
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc = {};
  bufferDesc.offset = 0;
  bufferDesc.size = memReqs.size;
  bufferDesc.flags = 0;

  cuErr = cuExternalMemoryGetMappedBuffer(d_ptr, *extMem, &bufferDesc);
  if (cuErr != CUDA_SUCCESS) {
    const char* errStr;
    cuGetErrorString(cuErr, &errStr);
    printf("Failed to map CUDA buffer: %s\n", errStr);
    cuDestroyExternalMemory(*extMem);
    vkFreeMemory(vkCtx->device, vkMemory, nullptr);
    vkDestroyImage(vkCtx->device, vkImage, nullptr);
    close(*dmabuf_fd);
    return -1;
  }

#if CUDISP_DEBUG
  printf("[cuDisp] CUDA buffer mapped: ptr=%p, memReqs.size=%llu, bufferSize=%u\n",
         (void*)*d_ptr,
         (unsigned long long)memReqs.size,
         bufferSize);
#endif
  return 0;
}

//=============================================================================
// Present Thread for Continuous Flip Mode
//=============================================================================

static void* cuDisp_present_thread_func(void* arg) {
  struct cuDispSwapchain_t* swapchain = (struct cuDispSwapchain_t*)arg;
  cuDispSwapchainGPU* swapchain_gpu = swapchain->swapchain_gpu;
#if CUDISP_BENCH
  const char* bench_present_env = getenv("CUDISP_BENCH_PRESENT");
  const bool bench_present_enabled = (bench_present_env != NULL && bench_present_env[0] != '\0' &&
                                      strcmp(bench_present_env, "0") != 0);
  const char* bench_every_success_env = getenv("CUDISP_BENCH_EVERY_SUCCESS");
  const bool bench_every_success =
      (bench_every_success_env != NULL && bench_every_success_env[0] != '\0' &&
       strcmp(bench_every_success_env, "0") != 0);
  uint64_t bench_success_count = 0;
  uint64_t bench_ebusy_count = 0;
  uint64_t bench_error_count = 0;
  uint64_t bench_total_ns = 0;
  uint64_t bench_min_ns = UINT64_MAX;
  uint64_t bench_max_ns = 0;
  uint64_t bench_interval_success_count = 0;
  uint64_t bench_interval_ebusy_count = 0;
  uint64_t bench_interval_error_count = 0;
  uint64_t bench_interval_total_ns = 0;
  uint64_t bench_interval_min_ns = UINT64_MAX;
  uint64_t bench_interval_max_ns = 0;
  uint64_t bench_report_interval = 120;
  const char* bench_report_interval_env = getenv("CUDISP_BENCH_REPORT_INTERVAL");
  if (bench_report_interval_env && bench_report_interval_env[0] != '\0') {
    char* end = NULL;
    unsigned long parsed = strtoul(bench_report_interval_env, &end, 10);
    if (end != bench_report_interval_env && *end == '\0' && parsed > 0) {
      bench_report_interval = (uint64_t)parsed;
    }
  }
#endif

  // Pin to CPU 0 for consistent timing
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(0, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

  // Wait for GPU to write first frame
  while (!swapchain->exit_thread.load(std::memory_order_acquire)) {
    if (swapchain_gpu->prevSlot == 0xffffffff) {
      usleep(1000);
      continue;  // GPU hasn't started yet
    } else {
      break;  // GPU has written first frame
    }
  }

#if CUDISP_DEBUG
  printf("cuDisp present thread started, prevSlot = %x\n", swapchain_gpu->prevSlot);
#endif
#if CUDISP_BENCH
  if (bench_present_enabled) {
    printf("[cuDisp BENCH] present benchmark enabled (interval=%llu, per_success=%d)\n",
           (unsigned long long)bench_report_interval,
           bench_every_success ? 1 : 0);
  }
#endif
  uint32_t prevSlot = swapchain_gpu->prevSlot;

  while (!swapchain->exit_thread.load(std::memory_order_acquire)) {
    uint32_t requestedGPUPresent = swapchain_gpu->notifySlots[prevSlot];
#if CUDISP_DEBUG
    printf("prevSlot: %d, requestedGPUPresent: %d\n", prevSlot, requestedGPUPresent);
#endif

    if (requestedGPUPresent == 1) {
      // Find which buffer to flip to
      void* requested_buf = swapchain_gpu->bufSlots[prevSlot][0];
      CUdeviceptr deviceptr = (CUdeviceptr)(uintptr_t)requested_buf;
      cuDispBufferMemory bufMem = {};
      bufMem.devicePtr = &deviceptr;
      bufMem.size = NULL;
      bufMem.pHDRMetadata = NULL;

#if CUDISP_BENCH
      struct timespec bench_start = {};
      struct timespec bench_end = {};
      if (bench_present_enabled) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &bench_start);
      }
#endif
      // Use the cuDispPresent API
      cuDispStatus status = cuDispPresent((cuDispSwapchain)swapchain, NULL, &bufMem, 1, 0);
#if CUDISP_BENCH
      if (bench_present_enabled) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &bench_end);
      }
      uint64_t elapsed_ns = 0;
      if (bench_present_enabled) {
        int64_t sec_delta = (int64_t)bench_end.tv_sec - (int64_t)bench_start.tv_sec;
        int64_t nsec_delta = (int64_t)bench_end.tv_nsec - (int64_t)bench_start.tv_nsec;
        elapsed_ns = (uint64_t)(sec_delta * 1000000000LL + nsec_delta);
      }
#endif
      if (status == cuDispErrorDisplay) {
        // Transient display busy (e.g. -EBUSY); retry same slot.
#if CUDISP_BENCH
        if (bench_present_enabled) {
          bench_ebusy_count++;
          bench_interval_ebusy_count++;
        }
#endif
        continue;
      }
      if (status != cuDispSuccess) {
#if CUDISP_BENCH
        if (bench_present_enabled) {
          bench_error_count++;
          bench_interval_error_count++;
        }
#endif
        printf("[cuDisp FLIP ERROR] cuDispPresent failed for buffer %p\n", requested_buf);
      }
#if CUDISP_BENCH
      if (status == cuDispSuccess && bench_present_enabled) {
        bench_success_count++;
        bench_total_ns += elapsed_ns;
        if (elapsed_ns < bench_min_ns) {
          bench_min_ns = elapsed_ns;
        }
        if (elapsed_ns > bench_max_ns) {
          bench_max_ns = elapsed_ns;
        }
        bench_interval_success_count++;
        bench_interval_total_ns += elapsed_ns;
        if (elapsed_ns < bench_interval_min_ns) {
          bench_interval_min_ns = elapsed_ns;
        }
        if (elapsed_ns > bench_interval_max_ns) {
          bench_interval_max_ns = elapsed_ns;
        }

        if (bench_every_success) {
          printf("[cuDisp BENCH] success=%llu slot=%u latency_us=%.1f\n",
                 (unsigned long long)bench_success_count,
                 prevSlot,
                 (double)elapsed_ns / 1000.0);
        }

        if (bench_interval_success_count >= bench_report_interval) {
          const double avg_us =
              ((double)bench_interval_total_ns / (double)bench_interval_success_count) / 1000.0;
          const double min_us = (double)bench_interval_min_ns / 1000.0;
          const double max_us = (double)bench_interval_max_ns / 1000.0;
          const double jitter_us = max_us - min_us;
          printf(
              "[cuDisp BENCH] interval_success=%llu total_success=%llu avg_us=%.1f min_us=%.1f "
              "max_us=%.1f jitter_us=%.1f interval_ebusy=%llu interval_errors=%llu\n",
              (unsigned long long)bench_interval_success_count,
              (unsigned long long)bench_success_count,
              avg_us,
              min_us,
              max_us,
              jitter_us,
              (unsigned long long)bench_interval_ebusy_count,
              (unsigned long long)bench_interval_error_count);

          bench_interval_success_count = 0;
          bench_interval_ebusy_count = 0;
          bench_interval_error_count = 0;
          bench_interval_total_ns = 0;
          bench_interval_min_ns = UINT64_MAX;
          bench_interval_max_ns = 0;
        }
      }
#endif

      // Cleanup (non-critical)
      swapchain_gpu->notifySlots[prevSlot] = 0;
      swapchain_gpu->bufSlots[prevSlot][0] = NULL;
      prevSlot = (prevSlot + 1) % CUDISP_PRESENT_NOTIFY_SLOTS;
    }
  }

#if CUDISP_BENCH
  if (bench_present_enabled && bench_interval_success_count > 0) {
    const double avg_us =
        ((double)bench_interval_total_ns / (double)bench_interval_success_count) / 1000.0;
    const double min_us = (double)bench_interval_min_ns / 1000.0;
    const double max_us = (double)bench_interval_max_ns / 1000.0;
    const double jitter_us = max_us - min_us;
    printf(
        "[cuDisp BENCH] final_interval interval_success=%llu avg_us=%.1f min_us=%.1f "
        "max_us=%.1f jitter_us=%.1f interval_ebusy=%llu interval_errors=%llu\n",
        (unsigned long long)bench_interval_success_count,
        avg_us,
        min_us,
        max_us,
        jitter_us,
        (unsigned long long)bench_interval_ebusy_count,
        (unsigned long long)bench_interval_error_count);
  }
  if (bench_present_enabled && bench_success_count > 0) {
    const double avg_us = ((double)bench_total_ns / (double)bench_success_count) / 1000.0;
    const double min_us = (double)bench_min_ns / 1000.0;
    const double max_us = (double)bench_max_ns / 1000.0;
    const double jitter_us = max_us - min_us;
    printf(
        "[cuDisp BENCH] final success=%llu avg_us=%.1f min_us=%.1f max_us=%.1f "
        "jitter_us=%.1f ebusy=%llu errors=%llu\n",
        (unsigned long long)bench_success_count,
        avg_us,
        min_us,
        max_us,
        jitter_us,
        (unsigned long long)bench_ebusy_count,
        (unsigned long long)bench_error_count);
  }
#endif
#if CUDISP_DEBUG
  printf("cuDisp present thread exiting\n");
#endif
  return NULL;
}

//=============================================================================
// Attribute Parsing
//=============================================================================

// Parse attributes and extract configuration
static cuDispStatus parseAttributes(const cuDispCreateAttribute* attributes, uint32_t numAttributes,
                                    uint32_t* width, uint32_t* height, uint32_t* numBuffers,
                                    cuDispSurfaceFormat* surfaceFormat,
                                    uint32_t* refreshRateMilliHz, bool* enableVrr,
                                    void*** gpuPresentHandle) {
  bool hasPrimaryBufferInfo = false;
  *refreshRateMilliHz = 0;  // 0 = use display native
  *enableVrr = false;
  *gpuPresentHandle = NULL;

  for (uint32_t i = 0; i < numAttributes; i++) {
    switch (attributes[i].id) {
      case CUDISP_CREATE_ATTRIBUTE_IGNORE:
        break;

      case CUDISP_CREATE_ATTRIBUTE_MODE_INFO: {
        const cuDispModeInfo* mi = &attributes[i].value.modeInfo;
        *refreshRateMilliHz = mi->refreshRateMilliHz;
        *enableVrr = (mi->enableVrr != 0);
        if (mi->maxBpc != CUDISP_MAX_BPC_DEFAULT) {
          printf("maxBpc other than default is not supported\n");
          return cuDispErrorNotSupported;
        }
        break;
      }

      case CUDISP_CREATE_ATTRIBUTE_BUFFER_INFO: {
        const cuDispBufferInfo* bi = &attributes[i].value.bufferInfo;
        if (bi->layerIndex != 0) {
          printf("Only layer 0 (primary) is supported\n");
          return cuDispErrorNotSupported;
        }
        if (!isSupportedSurfaceFormat(bi->format)) {
          printf("Unsupported surface format: %u\n", bi->format);
          return cuDispErrorNotSupported;
        }
        if (bi->scaleWidth != 0 || bi->scaleHeight != 0) {
          printf("Scaling is not supported\n");
          return cuDispErrorNotSupported;
        }
        if (bi->posX != 0 || bi->posY != 0) {
          printf("Windowed mode (non-zero position) is not supported\n");
          return cuDispErrorNotSupported;
        }
        if (bi->alpha != 0xFFFF) {
          printf("Non-default alpha is not supported\n");
          return cuDispErrorNotSupported;
        }
        if (bi->blendMode != CUDISP_BLEND_MODE_DEFAULT) {
          printf("Non-default blend mode is not supported\n");
          return cuDispErrorNotSupported;
        }
        if (bi->rotation != CUDISP_ROTATE_0) {
          printf("Non-default rotation is not supported\n");
          return cuDispErrorNotSupported;
        }
        if (bi->colorEncoding != CUDISP_COLOR_ENCODING_DEFAULT) {
          printf("Non-default color encoding is not supported\n");
          return cuDispErrorNotSupported;
        }
        if (bi->colorRange != CUDISP_COLOR_RANGE_DEFAULT) {
          printf("Non-default color range is not supported\n");
          return cuDispErrorNotSupported;
        }
        *width = bi->width;
        *height = bi->height;
        *numBuffers = bi->numBuffers;
        *surfaceFormat = bi->format;
        hasPrimaryBufferInfo = true;
        break;
      }

      case CUDISP_CREATE_ATTRIBUTE_GPU_PRESENT:
        *gpuPresentHandle = attributes[i].value.gpuPresent.handleGPUPresent;
        break;

      case CUDISP_CREATE_ATTRIBUTE_HDR_METADATA:
        printf("HDR metadata is not supported\n");
        return cuDispErrorNotSupported;

      case CUDISP_CREATE_ATTRIBUTE_COLORSPACE:
        printf("Colorspace configuration is not supported\n");
        return cuDispErrorNotSupported;

      case CUDISP_CREATE_ATTRIBUTE_DEGAMMA_LUT:
        printf("Degamma LUT is not supported\n");
        return cuDispErrorNotSupported;

      case CUDISP_CREATE_ATTRIBUTE_GAMMA_LUT:
        printf("Gamma LUT is not supported\n");
        return cuDispErrorNotSupported;

      case CUDISP_CREATE_ATTRIBUTE_CTM:
        printf("CTM is not supported\n");
        return cuDispErrorNotSupported;

      case CUDISP_CREATE_ATTRIBUTE_DISPLAY_SELECT:
        printf("Display selection is not supported\n");
        return cuDispErrorNotSupported;

      default:
        printf("Unknown attribute ID: %u\n", attributes[i].id);
        return cuDispErrorInvalidParam;
    }
  }

  // Validate required attributes
  if (!hasPrimaryBufferInfo) {
    printf("Missing required attribute: CUDISP_CREATE_ATTRIBUTE_BUFFER_INFO for layer 0\n");
    return cuDispErrorInvalidParam;
  }

  if (*numBuffers == 0 || *numBuffers > MAX_BUFFERS) {
    printf("Invalid numBuffers: %u (must be 1-%d)\n", *numBuffers, MAX_BUFFERS);
    return cuDispErrorInvalidParam;
  }

  return cuDispSuccess;
}

//=============================================================================
// Error Code Mapping
//=============================================================================

static cuDispStatus mapErrorCode(int error) {
  if (error == 0) {
    return cuDispSuccess;
  }

  switch (error) {
    case -EBUSY:
      return cuDispErrorDisplay;
    case -ENOMEM:
      return cuDispErrorOutOfResources;
    case -EINVAL:
      return cuDispErrorInvalidParam;
    default:
      return cuDispErrorUnknown;
  }
}

//=============================================================================
// Public API Implementation
//=============================================================================

cuDispStatus cuDispGetVersion(uint64_t* version) {
  if (!version) {
    return cuDispErrorInvalidParam;
  }

  *version = ((uint64_t)CUDISP_VER_MAJOR << 32) | ((uint64_t)CUDISP_VER_MINOR << 16) |
             (uint64_t)CUDISP_VER_PATCH;

  return cuDispSuccess;
}

cuDispStatus cuDispCreateSwapchain(cuDispSwapchain* swapchain, cuDispCreateAttribute* attributes,
                                   uint32_t numAttributes, uint32_t flags) {
  if (!swapchain || !attributes) {
    return cuDispErrorInvalidParam;
  }

  if (flags != 0) {
    printf("Non-zero flags not supported in v1.0\n");
    return cuDispErrorInvalidParam;
  }

  // Parse attributes
  uint32_t width = 0, height = 0, numBuffers = 0;
  cuDispSurfaceFormat surfaceFormat = CUDISP_SURFACE_FORMAT_ARGB8888;
  uint32_t refreshRateMilliHz = 0;
  bool enableVrr = false;
  void** gpuPresentHandle = NULL;

  cuDispStatus status = parseAttributes(attributes,
                                        numAttributes,
                                        &width,
                                        &height,
                                        &numBuffers,
                                        &surfaceFormat,
                                        &refreshRateMilliHz,
                                        &enableVrr,
                                        &gpuPresentHandle);
  if (status != cuDispSuccess) {
    return status;
  }

  // Allocate swapchain context
  struct cuDispSwapchain_t* sc =
      (struct cuDispSwapchain_t*)calloc(1, sizeof(struct cuDispSwapchain_t));
  if (!sc) {
    return cuDispErrorOutOfResources;
  }

  for (uint32_t i = 0; i < MAX_BUFFERS; i++) {
    sc->dmabuf_fds[i] = -1;
  }

  sc->num_buffers = numBuffers;
  sc->first_flip = true;
  sc->exit_thread.store(false);
  sc->thread_started = false;
  sc->front_buffer_render = (gpuPresentHandle == NULL);  // FBR if no GPU present handle
  sc->width = width;
  sc->height = height;
  sc->refresh_rate_millihz = refreshRateMilliHz;
  sc->drm_format = mapSurfaceFormatToDrmFourcc(surfaceFormat);
  printf("[cuDisp] Creating swapchain: %ux%u, surface_format=%d, drm_format=0x%08x\n",
         width,
         height,
         surfaceFormat,
         sc->drm_format);

  // Allocate swapchain GPU structure internally if continuous flip mode
  if (!sc->front_buffer_render) {
    CUresult cu_err = cuMemHostAlloc((void**)&sc->swapchain_gpu,
                                     sizeof(cuDispSwapchainGPU),
                                     CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE);
    if (cu_err != CUDA_SUCCESS) {
      printf("Failed to allocate swapchain GPU structure\n");
      free(sc);
      return cuDispErrorOutOfResources;
    }
    memset(sc->swapchain_gpu, 0, sizeof(cuDispSwapchainGPU));
    sc->swapchain_gpu->prevSlot = 0xffffffffU;

    // Return pointer to app via output attribute
    if (gpuPresentHandle) {
      *gpuPresentHandle = sc->swapchain_gpu;
    }
  } else {
    sc->swapchain_gpu = NULL;
  }

  // Find and open DRM device (CUDISP_DRM_DEVICE overrides which card: use 1, 2, 3 for card1, card2,
  // card3)
  char driNode[256];
  const char* env_dev = getenv("CUDISP_DRM_DEVICE");
  if (env_dev && env_dev[0] != '\0') {
    size_t cap = sizeof(driNode);
    if (env_dev[0] >= '0' && env_dev[0] <= '9') {
      snprintf(driNode, cap, "/dev/dri/card%.*s", (int)(cap - 14), env_dev);
    } else if (strchr(env_dev, '/') != NULL) {
      snprintf(driNode, cap, "%.*s", (int)(cap - 1), env_dev);
    } else {
      snprintf(driNode, cap, "/dev/dri/%.*s", (int)(cap - 11), env_dev);
    }
  } else if (drm_util_find_device(driNode, sizeof(driNode)) != 0) {
    printf("Failed to find DRM device\n");
    if (sc->swapchain_gpu) {
      cuMemFreeHost(sc->swapchain_gpu);
    }
    free(sc);
    return cuDispErrorDisplay;
  }

  printf("[cuDisp] Opening DRM device: %s\n", driNode);
  sc->fd = open(driNode, O_RDWR | O_CLOEXEC);
  if (sc->fd < 0) {
    printf("Failed to open node %s\n", driNode);
    if (sc->swapchain_gpu) {
      cuMemFreeHost(sc->swapchain_gpu);
    }
    free(sc);
    return cuDispErrorOs;
  }

  // Set DRM capabilities
  if (drmSetClientCap(sc->fd, DRM_CLIENT_CAP_ATOMIC, 1)) {
    printf("Failed to set atomic cap\n");
    close(sc->fd);
    if (sc->swapchain_gpu) {
      cuMemFreeHost(sc->swapchain_gpu);
    }
    free(sc);
    return cuDispErrorDisplay;
  }
  if (drmSetClientCap(sc->fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1)) {
    printf("Failed to set Universal planes cap\n");
    close(sc->fd);
    if (sc->swapchain_gpu) {
      cuMemFreeHost(sc->swapchain_gpu);
    }
    free(sc);
    return cuDispErrorDisplay;
  }

  // Get connector, CRTC, and plane info
  if (get_connector_crtc_plane_info(sc) == 0) {
    printf("Failed to initialize connector_crtc_plane_info\n");
    close(sc->fd);
    if (sc->swapchain_gpu) {
      cuMemFreeHost(sc->swapchain_gpu);
    }
    free(sc);
    return cuDispErrorDisplay;
  }

  // Find suitable mode.
  //   Priority (highest first):
  //     1. Exact resolution + closest refresh rate  (precise milliHz from pixel clock)
  //     When VRR is enabled:
  //       2. Exact resolution + highest available refresh rate
  //       3. Exact resolution + preferred flag
  //     When VRR is disabled:
  //       2. Exact resolution + preferred flag
  //       3. Exact resolution + highest available refresh rate
  //     4. No resolution match: error
  drmModeModeInfo* mode = NULL;

#if CUDISP_DEBUG
  printf("[cuDisp] Available modes (%d total), target %ux%u",
         sc->connector.info->count_modes,
         width,
         height);
  if (refreshRateMilliHz > 0)
    printf(" @ %.3f Hz", refreshRateMilliHz / 1000.0);
  printf(" (vrr=%s):\n", enableVrr ? "on" : "off");
  for (int i = 0; i < sc->connector.info->count_modes; i++) {
    drmModeModeInfo* m = &sc->connector.info->modes[i];
    printf("  [%2d] %ux%u @ %.3f Hz (vrefresh=%u, clock=%u kHz, type=0x%x%s)\n",
           i,
           m->hdisplay,
           m->vdisplay,
           mode_refresh_millihz(m) / 1000.0,
           m->vrefresh,
           m->clock,
           m->type,
           (m->type & DRM_MODE_TYPE_PREFERRED) ? " PREFERRED" : "");
  }
#endif

  {
    drmModeModeInfo* res_closest_refresh = NULL;
    uint32_t best_refresh_delta = UINT32_MAX;
    drmModeModeInfo* res_preferred = NULL;
    drmModeModeInfo* res_highest = NULL;
    uint32_t highest_millihz = 0;

    for (int i = 0; i < sc->connector.info->count_modes; i++) {
      drmModeModeInfo* m = &sc->connector.info->modes[i];
      if (m->hdisplay != width || m->vdisplay != height)
        continue;

      // Only accept progressive scan modes (reject interlaced and double-scan)
      const uint32_t non_progressive_flags = DRM_MODE_FLAG_INTERLACE | DRM_MODE_FLAG_DBLSCAN;
      if (m->flags & non_progressive_flags) {
#if CUDISP_DEBUG
        printf("[cuDisp] Skipping non-progressive mode [%d]: %ux%u @ %u mHz (flags=0x%x)\n",
               i,
               m->hdisplay,
               m->vdisplay,
               mode_refresh_millihz(m),
               m->flags);
#endif
        continue;
      }

      uint32_t actual_mhz = mode_refresh_millihz(m);

#if CUDISP_DEBUG
      printf(
          "[cuDisp] Candidate mode [%d]: %ux%u @ %u mHz (vrefresh=%u, clock=%u kHz, "
          "flags=0x%x, type=0x%x)\n",
          i,
          m->hdisplay,
          m->vdisplay,
          actual_mhz,
          m->vrefresh,
          m->clock,
          m->flags,
          m->type);
#endif
      if (!res_preferred && (m->type & DRM_MODE_TYPE_PREFERRED))
        res_preferred = m;

      if (actual_mhz > highest_millihz) {
        highest_millihz = actual_mhz;
        res_highest = m;
      }

      if (refreshRateMilliHz > 0) {
        uint32_t delta = (actual_mhz > refreshRateMilliHz) ? (actual_mhz - refreshRateMilliHz)
                                                           : (refreshRateMilliHz - actual_mhz);
        if (delta < best_refresh_delta) {
          best_refresh_delta = delta;
          res_closest_refresh = m;
        }
      }
    }

    if (res_closest_refresh) {
      mode = res_closest_refresh;
    } else if (enableVrr) {
      if (res_highest) {
        mode = res_highest;
      } else if (res_preferred) {
        mode = res_preferred;
      }
    } else {
      if (res_preferred) {
        mode = res_preferred;
      } else if (res_highest) {
        mode = res_highest;
      }
    }
  }

  if (!mode) {
    printf("[cuDisp] ERROR: no mode matching requested resolution %ux%u\n", width, height);
    close(sc->fd);
    if (sc->swapchain_gpu) {
      cuMemFreeHost(sc->swapchain_gpu);
    }
    free(sc);
    return cuDispErrorDisplay;
  }

  sc->crtc_id = sc->connector.crtc.info->crtc_id;
  sc->mode = mode;
  sc->connector_id = sc->connector.info->connector_id;

#if CUDISP_DEBUG
  printf("[cuDisp] Selected mode: %ux%u @ %u mHz (vrefresh=%u, clock=%u kHz, flags=0x%x)\n",
         mode->hdisplay,
         mode->vdisplay,
         mode_refresh_millihz(mode),
         mode->vrefresh,
         mode->clock,
         mode->flags);
#endif

  // Cache primary plane properties
  bool found_primary = false;
  for (int i = 0; i < sc->connector.crtc.num_planes; i++) {
    if (sc->connector.crtc.planes[i].type == CUDISP_PLANE_TYPE_PRIMARY) {
      uint32_t plane_id = sc->connector.crtc.planes[i].info->plane_id;
      printf("[cuDisp] Using PRIMARY plane: plane_id=%u\n", plane_id);
      if (!drm_cache_plane_properties(sc, plane_id)) {
        printf("Failed to cache plane properties\n");
        close(sc->fd);
        if (sc->swapchain_gpu) {
          cuMemFreeHost(sc->swapchain_gpu);
        }
        free(sc);
        return cuDispErrorDisplay;
      }
      found_primary = true;
      break;
    }
  }
  if (!found_primary) {
    printf("[cuDisp WARNING] No PRIMARY plane found! num_planes=%d\n",
           sc->connector.crtc.num_planes);
  }

  // Pre-calculate fixed values
  sc->src_w_fixed = sc->mode->hdisplay << 16;
  sc->src_h_fixed = sc->mode->vdisplay << 16;

#if CUDISP_DEBUG
  printf("cuDisp optimizations initialized\n");
#endif

  // Check VRR support
  drm_check_vrr_support(sc);
  sc->enable_vrr = enableVrr && sc->vrr_supported && sc->vrr_capable;
  if (enableVrr && !sc->enable_vrr) {
    printf("VRR requested but not available\n");
  }

  // Initialize Vulkan context
  if (initVulkanContext(&sc->vk_ctx) != 0) {
    printf("Failed to initialize Vulkan\n");
    close(sc->fd);
    if (sc->swapchain_gpu) {
      cuMemFreeHost(sc->swapchain_gpu);
    }
    free(sc);
    return cuDispErrorCreationFailed;
  }

  // Initialize GBM
  sc->gbm = gbm_create_device(sc->fd);
  if (!sc->gbm) {
    printf("Failed to create GBM device\n");
    vkDestroyDevice(sc->vk_ctx.device, nullptr);
    vkDestroyInstance(sc->vk_ctx.instance, nullptr);
    close(sc->fd);
    if (sc->swapchain_gpu) {
      cuMemFreeHost(sc->swapchain_gpu);
    }
    free(sc);
    return cuDispErrorCreationFailed;
  }

  // Create buffers via GBM->Vulkan->CUDA path
  uint32_t gbmStride = 0;
  for (uint32_t i = 0; i < numBuffers; i++) {
    uint32_t bufStride = 0;
    if (createBufferGbmVulkanCuda(sc->gbm,
                                  &sc->vk_ctx,
                                  width,
                                  height,
                                  surfaceFormat,
                                  &sc->d_ptrs[i],
                                  &sc->dmabuf_fds[i],
                                  &sc->ext_mems[i],
                                  &bufStride) != 0) {
      printf("Failed to create buffer %d\n", i);
      // Cleanup already created buffers
      for (uint32_t j = 0; j < i; j++) {
        releaseImportedCudaBuffer(&sc->d_ptrs[j], &sc->ext_mems[j], &sc->dmabuf_fds[j]);
      }
      gbm_device_destroy(sc->gbm);
      vkDestroyDevice(sc->vk_ctx.device, nullptr);
      vkDestroyInstance(sc->vk_ctx.instance, nullptr);
      close(sc->fd);
      if (sc->swapchain_gpu) {
        cuMemFreeHost(sc->swapchain_gpu);
      }
      free(sc);
      return cuDispErrorCreationFailed;
    }
    if (i == 0) {
      gbmStride = bufStride;
    }
  }

  sc->stride = gbmStride;
  sc->buffer_size = static_cast<uint64_t>(gbmStride) * height;

  // Create DRM framebuffers using actual GBM stride
  for (uint32_t i = 0; i < numBuffers; i++) {
    uint32_t handle;
    int ret = drmPrimeFDToHandle(sc->fd, sc->dmabuf_fds[i], &handle);
    if (ret) {
      printf("drmPrimeFDToHandle failed for buffer %d (errno=%d)\n", i, errno);
      // Cleanup
      for (uint32_t j = 0; j < numBuffers; j++) {
        releaseImportedCudaBuffer(&sc->d_ptrs[j], &sc->ext_mems[j], &sc->dmabuf_fds[j]);
      }
      gbm_device_destroy(sc->gbm);
      vkDestroyDevice(sc->vk_ctx.device, nullptr);
      vkDestroyInstance(sc->vk_ctx.instance, nullptr);
      close(sc->fd);
      if (sc->swapchain_gpu) {
        cuMemFreeHost(sc->swapchain_gpu);
      }
      free(sc);
      return cuDispErrorDisplay;
    }

    uint32_t handles[4] = {handle, 0, 0, 0};
    uint32_t pitches[4] = {gbmStride, 0, 0, 0};
    uint32_t offsets[4] = {0, 0, 0, 0};

#if CUDISP_DEBUG
    printf("[cuDisp FB] buffer[%d]: stride=%u, width=%u, height=%u, format=0x%08x\n",
           i,
           gbmStride,
           width,
           height,
           sc->drm_format);
#endif

    ret = drmModeAddFB2(
        sc->fd, width, height, sc->drm_format, handles, pitches, offsets, &sc->fb_ids[i], 0);

    if (ret) {
      printf(
          "Failed to create framebuffer %d: drmModeAddFB2 returned %d (errno=%d), "
          "width=%u height=%u format=0x%08x pitch=%u handle=%u\n",
          i,
          ret,
          errno,
          width,
          height,
          sc->drm_format,
          gbmStride,
          handle);
      // Cleanup
      for (uint32_t j = 0; j < numBuffers; j++) {
        releaseImportedCudaBuffer(&sc->d_ptrs[j], &sc->ext_mems[j], &sc->dmabuf_fds[j]);
      }
      gbm_device_destroy(sc->gbm);
      vkDestroyDevice(sc->vk_ctx.device, nullptr);
      vkDestroyInstance(sc->vk_ctx.instance, nullptr);
      close(sc->fd);
      if (sc->swapchain_gpu) {
        cuMemFreeHost(sc->swapchain_gpu);
      }
      free(sc);
      return cuDispErrorDisplay;
    }

#if CUDISP_DEBUG
    printf("Created framebuffer %d: fb_id=%u\n", i, sc->fb_ids[i]);
#endif
  }

  printf("[cuDisp] Setting CRTC: crtc_id=%u, fb_id=%u, connector=%u, mode=%dx%d@%uHz\n",
         sc->crtc_id,
         sc->fb_ids[0],
         sc->res_info->connectors[sc->conn_index],
         sc->mode->hdisplay,
         sc->mode->vdisplay,
         sc->mode->vrefresh);
  if (drmModeSetCrtc(sc->fd,
                     sc->crtc_id,
                     sc->fb_ids[0],
                     0,
                     0,
                     &sc->res_info->connectors[sc->conn_index],
                     1,
                     sc->mode)) {
    printf("Failed to set crtc (errno=%d)\n", errno);
    // Cleanup
    for (uint32_t i = 0; i < numBuffers; i++) {
      drmModeRmFB(sc->fd, sc->fb_ids[i]);
      releaseImportedCudaBuffer(&sc->d_ptrs[i], &sc->ext_mems[i], &sc->dmabuf_fds[i]);
    }
    gbm_device_destroy(sc->gbm);
    vkDestroyDevice(sc->vk_ctx.device, nullptr);
    vkDestroyInstance(sc->vk_ctx.instance, nullptr);
    close(sc->fd);
    if (sc->swapchain_gpu) {
      cuMemFreeHost(sc->swapchain_gpu);
    }
    free(sc);
    return cuDispErrorDisplay;
  }

  // Start present thread for continuous flip mode
  if (!sc->front_buffer_render) {
    if (pthread_create(&sc->present_thread, NULL, cuDisp_present_thread_func, sc) != 0) {
      printf("Failed to create present thread\n");
      // Cleanup
      for (uint32_t i = 0; i < numBuffers; i++) {
        drmModeRmFB(sc->fd, sc->fb_ids[i]);
        releaseImportedCudaBuffer(&sc->d_ptrs[i], &sc->ext_mems[i], &sc->dmabuf_fds[i]);
      }
      gbm_device_destroy(sc->gbm);
      vkDestroyDevice(sc->vk_ctx.device, nullptr);
      vkDestroyInstance(sc->vk_ctx.instance, nullptr);
      close(sc->fd);
      if (sc->swapchain_gpu) {
        cuMemFreeHost(sc->swapchain_gpu);
      }
      free(sc);
      return cuDispErrorCreationFailed;
    }
    sc->thread_started = true;
#if CUDISP_DEBUG
    printf("cuDisp present thread created successfully\n");
#endif
  }

#if CUDISP_DEBUG
  printf("cuDisp initialization complete\n");
#endif
  *swapchain = (cuDispSwapchain)sc;
  return cuDispSuccess;
}

cuDispStatus cuDispGetBuffer(cuDispSwapchain swapchain, uint32_t layerIndex,
                             uint64_t bufNum,
                             cuDispBufferMemory* outBufferMemory, uint32_t flags) {
  if (!swapchain || !outBufferMemory) {
    return cuDispErrorInvalidParam;
  }

  if (layerIndex != 0) {
    return cuDispErrorNotSupported;
  }

  if (flags != 0) {
    return cuDispErrorInvalidParam;
  }

  struct cuDispSwapchain_t* sc = (struct cuDispSwapchain_t*)swapchain;
  if (bufNum >= sc->num_buffers) {
    printf("Invalid buffer index: %lu\n", (unsigned long)bufNum);
    return cuDispErrorInvalidParam;
  }

  if (outBufferMemory->devicePtr == NULL) {
    return cuDispErrorInvalidParam;
  }

  *outBufferMemory->devicePtr = sc->d_ptrs[bufNum];

  if (outBufferMemory->size != NULL) {
    *outBufferMemory->size = sc->buffer_size;
  }

  if (outBufferMemory->stride != NULL) {
    *outBufferMemory->stride = sc->stride;
  }

  if (outBufferMemory->pHDRMetadata != NULL) {
    memset(outBufferMemory->pHDRMetadata, 0, sizeof(cuDispHDRMetadata));
  }

  return cuDispSuccess;
}

cuDispStatus cuDispPresent(cuDispSwapchain swapchain, void* stream,
                           const cuDispBufferMemory* bufferMemory,
                           uint32_t numLayers, uint32_t flags) {
  if (!swapchain || !bufferMemory) {
    return cuDispErrorInvalidParam;
  }

  if (numLayers != 1) {
    return cuDispErrorNotSupported;
  }

  if (flags & CUDISP_PRESENT_FLAG_VSYNC_OFF) {
    return cuDispErrorNotSupported;
  }

  if (flags != 0) {
    return cuDispErrorInvalidParam;
  }

  if (bufferMemory[0].devicePtr == NULL) {
    return cuDispErrorInvalidParam;
  }

  // Stream parameter is reserved for future use (currently ignored)
  // When implemented, stream != NULL will synchronize present with stream completion
  (void)stream;  // Suppress unused parameter warning

  struct cuDispSwapchain_t* sc = (struct cuDispSwapchain_t*)swapchain;
  CUdeviceptr deviceptr = *bufferMemory[0].devicePtr;

  // Find buffer index from pointer
  uint32_t buffer_index = (uint32_t)-1;
  for (uint32_t i = 0; i < sc->num_buffers; i++) {
    if (sc->d_ptrs[i] == deviceptr) {
      buffer_index = i;
      break;
    }
  }

  if (buffer_index == (uint32_t)-1) {
    printf("Invalid buffer pointer: %llu\n", (unsigned long long)deviceptr);
    return cuDispErrorInvalidParam;
  }

  int ret;
  uint32_t drm_flags = 0;
  struct CachedPlaneProps* cache = &sc->cached_props;
  uint32_t fb_id = sc->fb_ids[buffer_index];

  // Allocate request locally for each flip
  drmModeAtomicReqPtr req = drmModeAtomicAlloc();
  if (!req) {
    printf("Failed to allocate DRM atomic request\n");
    return cuDispErrorOutOfResources;
  }

  if (sc->first_flip) {
#if CUDISP_DEBUG
    printf(
        "[cuDisp] First flip debug: plane_id=%u, crtc_id=%u, fb_id=%u\n"
        "  props: FB_ID=%u, CRTC_ID=%u, SRC_X=%u, SRC_Y=%u, SRC_W=%u, SRC_H=%u\n"
        "  props: CRTC_X=%u, CRTC_Y=%u, CRTC_W=%u, CRTC_H=%u, alpha=%u\n"
        "  src_w_fixed=%u (mode.hdisplay=%u), src_h_fixed=%u (mode.vdisplay=%u)\n",
        cache->plane_id,
        sc->crtc_id,
        fb_id,
        cache->fb_id_prop,
        cache->crtc_id_prop,
        cache->src_x_prop,
        cache->src_y_prop,
        cache->src_w_prop,
        cache->src_h_prop,
        cache->crtc_x_prop,
        cache->crtc_y_prop,
        cache->crtc_w_prop,
        cache->crtc_h_prop,
        cache->alpha_prop,
        sc->src_w_fixed,
        sc->mode->hdisplay,
        sc->src_h_fixed,
        sc->mode->vdisplay);
#endif
    // First flip: set ALL properties
    drmModeAtomicAddProperty(req, cache->plane_id, cache->fb_id_prop, fb_id);
    drmModeAtomicAddProperty(req, cache->plane_id, cache->crtc_id_prop, sc->crtc_id);
    drmModeAtomicAddProperty(req, cache->plane_id, cache->src_x_prop, 0);
    drmModeAtomicAddProperty(req, cache->plane_id, cache->src_y_prop, 0);
    drmModeAtomicAddProperty(req, cache->plane_id, cache->src_w_prop, sc->src_w_fixed);
    drmModeAtomicAddProperty(req, cache->plane_id, cache->src_h_prop, sc->src_h_fixed);
    drmModeAtomicAddProperty(req, cache->plane_id, cache->crtc_x_prop, 0);
    drmModeAtomicAddProperty(req, cache->plane_id, cache->crtc_y_prop, 0);
    drmModeAtomicAddProperty(req, cache->plane_id, cache->crtc_w_prop, sc->mode->hdisplay);
    drmModeAtomicAddProperty(req, cache->plane_id, cache->crtc_h_prop, sc->mode->vdisplay);
    if (cache->alpha_prop) {
      drmModeAtomicAddProperty(req, cache->plane_id, cache->alpha_prop, 0xffff);
    }

    // Enable VRR if requested and supported
    if (sc->enable_vrr && sc->vrr_supported) {
#if CUDISP_DEBUG
      printf("[cuDisp] Adding VRR_ENABLED property: CRTC_ID=%u, prop_id=%u, value=1\n",
             sc->crtc_id,
             sc->vrr_enabled_prop_id);
#endif
      drmModeAtomicAddProperty(req, sc->crtc_id, sc->vrr_enabled_prop_id, 1);
#if CUDISP_DEBUG
      printf("[cuDisp] VRR enabled in atomic commit\n");
#endif
    }

    drm_flags = DRM_MODE_ATOMIC_ALLOW_MODESET | DRM_MODE_ATOMIC_NONBLOCK;
    sc->first_flip = false;
  } else {
    // Subsequent flips: only change FB_ID
    drmModeAtomicAddProperty(req, cache->plane_id, cache->fb_id_prop, fb_id);
    drm_flags = DRM_MODE_ATOMIC_NONBLOCK;
  }

  // Single commit
  ret = drmModeAtomicCommit(sc->fd, req, drm_flags, NULL);
  drmModeAtomicFree(req);

  if (ret && ret != -EBUSY) {
    printf("[cuDisp FLIP ERROR] Atomic commit failed, error code %d (fb_id=%u)\n", ret, fb_id);
    return mapErrorCode(ret);
  } else if (ret == -EBUSY) {
#if CUDISP_DEBUG
    printf("[cuDisp FLIP EBUSY] Flip busy: previous flip still in progress (fb_id=%u)\n", fb_id);
#endif
    return cuDispErrorDisplay;
  }

  return cuDispSuccess;
}

cuDispStatus cuDispDestroySwapchain(cuDispSwapchain swapchain) {
  if (!swapchain) {
    return cuDispErrorInvalidParam;
  }

  struct cuDispSwapchain_t* sc = (struct cuDispSwapchain_t*)swapchain;

  // Stop present thread if running
  if (sc->thread_started) {
#if CUDISP_DEBUG
    printf("Stopping cuDisp present thread...\n");
#endif
    sc->exit_thread.store(true, std::memory_order_release);
    pthread_join(sc->present_thread, NULL);
#if CUDISP_DEBUG
    printf("cuDisp present thread stopped\n");
#endif
  }

  // Remove framebuffers
  for (uint32_t i = 0; i < sc->num_buffers; i++) {
    if (sc->fb_ids[i]) {
      drmModeRmFB(sc->fd, sc->fb_ids[i]);
    }
  }

  // Free CUDA external memory
  for (uint32_t i = 0; i < sc->num_buffers; i++) {
    releaseImportedCudaBuffer(&sc->d_ptrs[i], &sc->ext_mems[i], &sc->dmabuf_fds[i]);
  }

  // Free planes
  for (int i = 0; i < sc->connector.crtc.num_planes; i++) {
    if (sc->connector.crtc.planes[i].info) {
      drmModeFreePlane(sc->connector.crtc.planes[i].info);
    }
  }

  // Free connector, CRTC, resources
  if (sc->connector.info) {
    drmModeFreeConnector(sc->connector.info);
  }
  if (sc->connector.crtc.info) {
    drmModeFreeCrtc(sc->connector.crtc.info);
  }
  if (sc->res_info) {
    drmModeFreeResources(sc->res_info);
  }

  // Destroy GBM
  if (sc->gbm) {
    gbm_device_destroy(sc->gbm);
  }

  // Destroy Vulkan
  if (sc->vk_ctx.device) {
    vkDestroyDevice(sc->vk_ctx.device, nullptr);
  }
  if (sc->vk_ctx.instance) {
    vkDestroyInstance(sc->vk_ctx.instance, nullptr);
  }

  // Close DRM fd
  if (sc->fd >= 0) {
    close(sc->fd);
  }

  // Free swapchain GPU structure if allocated
  if (sc->swapchain_gpu) {
    cuMemFreeHost(sc->swapchain_gpu);
  }

  free(sc);
#if CUDISP_DEBUG
  printf("cuDisp cleanup complete\n");
#endif
  return cuDispSuccess;
}
