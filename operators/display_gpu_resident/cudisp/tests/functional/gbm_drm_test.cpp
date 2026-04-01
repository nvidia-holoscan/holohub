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
// GBM + DRM Platform Validation Test
//
// Validates the GBM buffer allocation and DRM scanout path on this
// platform. Allocates GBM buffers with various usage flags, exports
// them as dma-buf FDs, imports into DRM as framebuffers, and displays.
//
// Usage:
//   gbm_drm_test [options]
//
// Options:
//   --card <N>           DRI card number (default: auto-detect)
//   --display <name>     Connector or monitor name (default: first connected)
//   --mode <WxH>         Display mode (default: preferred)
//   --flags <flag,...>    GBM usage flags: scanout,linear,render,write
//                         (default: scanout,linear)
//   --num-buffers <N>    Number of GBM buffers to allocate (default: 2)
//   --duration <sec>     Hold duration (default: 3)
//   --help
//

#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>
#include <gbm.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
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
#include <sys/select.h>
#include <unistd.h>
#include <vector>

// ============================================================================
// Connector helpers (shared pattern with drm_basic_test)
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
    if (strcmp(prop->name, "EDID") == 0 &&
        (prop->flags & DRM_MODE_PROP_BLOB)) {
      drmModePropertyBlobPtr blob =
          drmModeGetPropertyBlob(fd, props->prop_values[i]);
      if (blob && blob->length >= 128) {
        auto* edid = static_cast<const uint8_t*>(blob->data);
        for (int d = 0; d < 4; d++) {
          int off = 54 + d * 18;
          if (off + 18 > static_cast<int>(blob->length)) break;
          if (edid[off] == 0 && edid[off + 1] == 0 &&
              edid[off + 2] == 0 && edid[off + 3] == 0xFC) {
            char name[14] = {};
            memcpy(name, &edid[off + 5], 13);
            for (int k = 12; k >= 0; k--) {
              if (name[k] == '\n' || name[k] == ' ' || name[k] == '\0')
                name[k] = '\0';
              else
                break;
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
      (uint64_t)m->clock * 1000ULL /
      ((uint64_t)m->htotal * m->vtotal));
}

// ============================================================================
// Plane helpers
// ============================================================================
static uint32_t find_plane_prop(int fd, uint32_t plane_id,
                                const char* name) {
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

// ============================================================================
// DRM device open
// ============================================================================
static int open_drm_device(int card_num) {
  char path[64];
  if (card_num >= 0) {
    snprintf(path, sizeof(path), "/dev/dri/card%d", card_num);
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0)
      printf("[ERROR] Cannot open %s: %s\n", path, strerror(errno));
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
// GBM buffer wrapper
// ============================================================================
struct GbmBuffer {
  struct gbm_bo* bo = nullptr;
  int dmabuf_fd = -1;
  uint32_t drm_handle = 0;
  uint32_t fb_id = 0;
  uint32_t stride = 0;
  uint32_t width = 0;
  uint32_t height = 0;
};

static bool create_gbm_buffer(struct gbm_device* gbm, int drm_fd,
                               GbmBuffer* buf, uint32_t w, uint32_t h,
                               uint32_t gbm_format, uint32_t gbm_flags,
                               uint32_t drm_format) {
  buf->width = w;
  buf->height = h;

  buf->bo = gbm_bo_create(gbm, w, h, gbm_format, gbm_flags);
  if (!buf->bo) {
    printf("  [FAIL] gbm_bo_create(%ux%u, flags=0x%x): %s\n",
           w, h, gbm_flags, strerror(errno));
    return false;
  }

  buf->dmabuf_fd = gbm_bo_get_fd(buf->bo);
  if (buf->dmabuf_fd < 0) {
    printf("  [FAIL] gbm_bo_get_fd: %s\n", strerror(errno));
    gbm_bo_destroy(buf->bo);
    buf->bo = nullptr;
    return false;
  }

  buf->stride = gbm_bo_get_stride(buf->bo);

  if (drmPrimeFDToHandle(drm_fd, buf->dmabuf_fd, &buf->drm_handle) != 0) {
    printf("  [FAIL] drmPrimeFDToHandle: %s\n", strerror(errno));
    close(buf->dmabuf_fd);
    gbm_bo_destroy(buf->bo);
    buf->bo = nullptr;
    return false;
  }

  uint32_t handles[4] = {buf->drm_handle, 0, 0, 0};
  uint32_t strides[4] = {buf->stride, 0, 0, 0};
  uint32_t offsets[4] = {0, 0, 0, 0};
  if (drmModeAddFB2(drm_fd, w, h, drm_format,
                    handles, strides, offsets,
                    &buf->fb_id, 0) != 0) {
    printf("  [FAIL] drmModeAddFB2: %s\n", strerror(errno));
    close(buf->dmabuf_fd);
    gbm_bo_destroy(buf->bo);
    buf->bo = nullptr;
    return false;
  }

  return true;
}

static void destroy_gbm_buffer(int drm_fd, GbmBuffer* buf) {
  if (buf->fb_id) {
    drmModeRmFB(drm_fd, buf->fb_id);
    buf->fb_id = 0;
  }
  if (buf->dmabuf_fd >= 0) {
    close(buf->dmabuf_fd);
    buf->dmabuf_fd = -1;
  }
  if (buf->bo) {
    gbm_bo_destroy(buf->bo);
    buf->bo = nullptr;
  }
}

// Write a solid color to a GBM buffer via mmap
static bool fill_gbm_buffer(GbmBuffer* buf, uint32_t argb) {
  uint32_t* map = nullptr;
  void* map_data = nullptr;
  uint32_t map_stride = 0;
  map = static_cast<uint32_t*>(
      gbm_bo_map(buf->bo, 0, 0, buf->width, buf->height,
                 GBM_BO_TRANSFER_WRITE, &map_stride, &map_data));
  if (!map) {
    printf("  [WARN] gbm_bo_map failed: %s. Buffer may show garbage.\n",
           strerror(errno));
    return false;
  }
  for (uint32_t y = 0; y < buf->height; y++) {
    uint32_t* row = reinterpret_cast<uint32_t*>(
        reinterpret_cast<uint8_t*>(map) + y * map_stride);
    for (uint32_t x = 0; x < buf->width; x++)
      row[x] = argb;
  }
  gbm_bo_unmap(buf->bo, map_data);
  return true;
}

// ============================================================================
// Config
// ============================================================================
struct Config {
  int card_num = -1;
  std::string display_name;
  uint32_t mode_w = 0;
  uint32_t mode_h = 0;
  uint32_t gbm_flags = GBM_BO_USE_SCANOUT | GBM_BO_USE_LINEAR;
  int num_buffers = 2;
  int duration = 3;
};

static uint32_t parse_gbm_flags(const std::string& str) {
  uint32_t flags = 0;
  std::string s = str;
  size_t pos;
  while ((pos = s.find(',')) != std::string::npos) {
    std::string tok = s.substr(0, pos);
    s = s.substr(pos + 1);
    if (tok == "scanout") flags |= GBM_BO_USE_SCANOUT;
    else if (tok == "linear") flags |= GBM_BO_USE_LINEAR;
    else if (tok == "render") flags |= GBM_BO_USE_RENDERING;
    else if (tok == "write") flags |= GBM_BO_USE_WRITE;
    else printf("[WARN] Unknown GBM flag: %s\n", tok.c_str());
  }
  if (!s.empty()) {
    if (s == "scanout") flags |= GBM_BO_USE_SCANOUT;
    else if (s == "linear") flags |= GBM_BO_USE_LINEAR;
    else if (s == "render") flags |= GBM_BO_USE_RENDERING;
    else if (s == "write") flags |= GBM_BO_USE_WRITE;
    else printf("[WARN] Unknown GBM flag: %s\n", s.c_str());
  }
  return flags;
}

static std::string gbm_flags_to_str(uint32_t flags) {
  std::string s;
  if (flags & GBM_BO_USE_SCANOUT)   { if (!s.empty()) s += ","; s += "scanout"; }
  if (flags & GBM_BO_USE_LINEAR)    { if (!s.empty()) s += ","; s += "linear"; }
  if (flags & GBM_BO_USE_RENDERING) { if (!s.empty()) s += ","; s += "render"; }
  if (flags & GBM_BO_USE_WRITE)     { if (!s.empty()) s += ","; s += "write"; }
  if (s.empty()) s = "none";
  return s;
}

static void print_help() {
  printf(
      "GBM + DRM Platform Validation Test\n"
      "\n"
      "Validates GBM buffer allocation and DRM scanout on this platform.\n"
      "\n"
      "Options:\n"
      "  --card <N>           DRI card number (default: auto)\n"
      "  --display <name>     Connector or monitor name\n"
      "  --mode <WxH>         Resolution (default: preferred)\n"
      "  --flags <f,...>      GBM flags: scanout,linear,render,write\n"
      "                       (default: scanout,linear)\n"
      "  --num-buffers <N>    Buffers to allocate (default: 2)\n"
      "  --duration <sec>     Hold duration (default: 3)\n"
      "  --help\n");
}

static bool parse_args(int argc, char** argv, Config* cfg) {
  static struct option long_opts[] = {
    {"card",        required_argument, nullptr, 'c'},
    {"display",     required_argument, nullptr, 'D'},
    {"mode",        required_argument, nullptr, 'm'},
    {"flags",       required_argument, nullptr, 'f'},
    {"num-buffers", required_argument, nullptr, 'n'},
    {"duration",    required_argument, nullptr, 'd'},
    {"help",        no_argument,       nullptr, 'h'},
    {nullptr, 0, nullptr, 0}
  };
  int opt;
  while ((opt = getopt_long(argc, argv, "hc:D:m:f:n:d:",
                            long_opts, nullptr)) != -1) {
    switch (opt) {
      case 'c': cfg->card_num = atoi(optarg); break;
      case 'D': cfg->display_name = optarg; break;
      case 'm':
        if (sscanf(optarg, "%ux%u", &cfg->mode_w, &cfg->mode_h) != 2) {
          printf("[ERROR] Invalid --mode format. Use WxH.\n");
          return false;
        }
        break;
      case 'f':
        cfg->gbm_flags = parse_gbm_flags(optarg);
        if (cfg->gbm_flags == 0) {
          printf("[ERROR] No valid GBM flags specified.\n");
          return false;
        }
        break;
      case 'n': cfg->num_buffers = atoi(optarg); break;
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
  printf("  GBM + DRM Platform Validation Test\n");
  printf("  Validates GBM buffer allocation and DRM"
         " scanout on this platform.\n");
  printf("=========================================="
         "==========================================\n\n");

  Config cfg;
  if (!parse_args(argc, argv, &cfg)) return 1;

  if (cfg.num_buffers < 1 || cfg.num_buffers > 16) {
    printf("[ERROR] --num-buffers must be 1..16.\n");
    return 1;
  }

  // --- Open DRM ---
  printf("[TEST] Opening DRM device...\n");
  int fd = open_drm_device(cfg.card_num);
  if (fd < 0) {
    printf("  [FAIL] No usable DRM device found.\n");
    return 1;
  }
  printf("  [PASS] DRM device opened (fd=%d)\n\n", fd);

  drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
  drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1);

  // --- Create GBM device ---
  printf("[TEST] Creating GBM device...\n");
  struct gbm_device* gbm = gbm_create_device(fd);
  if (!gbm) {
    printf("  [FAIL] gbm_create_device: %s\n", strerror(errno));
    close(fd);
    return 1;
  }
  printf("  [PASS] GBM device created.\n\n");

  // --- Find connector ---
  printf("[TEST] Finding display...\n");
  drmModeRes* res = drmModeGetResources(fd);
  if (!res) {
    printf("  [FAIL] drmModeGetResources: %s\n", strerror(errno));
    gbm_device_destroy(gbm);
    close(fd);
    return 1;
  }

  drmModeConnector* conn = nullptr;
  std::string selected_monitor;
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
        printf("  Selected: %s", cname.c_str());
        if (!mname.empty()) printf(" (%s)", mname.c_str());
        printf("\n");
        continue;
      }
    }
    drmModeFreeConnector(c);
  }
  if (!conn) {
    printf("  [FAIL] No connected display found.\n");
    drmModeFreeResources(res);
    gbm_device_destroy(gbm);
    close(fd);
    return 1;
  }
  printf("  [PASS] Display found.\n\n");

  // --- Select mode ---
  printf("[TEST] Selecting mode...\n");
  drmModeModeInfo* mode = nullptr;
  for (int i = 0; i < conn->count_modes; i++) {
    drmModeModeInfo* m = &conn->modes[i];
    if (cfg.mode_w > 0 && cfg.mode_h > 0) {
      if (m->hdisplay == cfg.mode_w && m->vdisplay == cfg.mode_h) {
        if (!mode || mode_refresh_hz(m) > mode_refresh_hz(mode))
          mode = m;
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
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    gbm_device_destroy(gbm);
    close(fd);
    return 1;
  }
  printf("  [PASS] Mode: %ux%u @ %u Hz\n\n",
         mode->hdisplay, mode->vdisplay, mode_refresh_hz(mode));

  // --- Find CRTC and primary plane ---
  uint32_t crtc_id = 0;
  if (conn->encoder_id) {
    drmModeEncoder* enc = drmModeGetEncoder(fd, conn->encoder_id);
    if (enc) { crtc_id = enc->crtc_id; drmModeFreeEncoder(enc); }
  }
  if (!crtc_id && res->count_crtcs > 0) {
    for (int i = 0; i < res->count_crtcs; i++) {
      if (conn->encoders && conn->count_encoders > 0) {
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
  for (int i = 0; i < res->count_crtcs; i++) {
    if (res->crtcs[i] == crtc_id) { crtc_index = i; break; }
  }

  uint32_t primary_plane_id = 0;
  drmModePlaneRes* plane_res = drmModeGetPlaneResources(fd);
  if (plane_res) {
    for (uint32_t i = 0; i < plane_res->count_planes; i++) {
      drmModePlane* p = drmModeGetPlane(fd, plane_res->planes[i]);
      if (!p) continue;
      if (crtc_index >= 0 && !(p->possible_crtcs & (1u << crtc_index))) {
        drmModeFreePlane(p);
        continue;
      }
      if (get_plane_type(fd, p->plane_id) == 1) {
        primary_plane_id = p->plane_id;
        drmModeFreePlane(p);
        break;
      }
      drmModeFreePlane(p);
    }
    drmModeFreePlaneResources(plane_res);
  }

  if (!crtc_id || !primary_plane_id) {
    printf("  [FAIL] No CRTC/primary plane found.\n");
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    gbm_device_destroy(gbm);
    close(fd);
    return 1;
  }
  printf("[INFO] CRTC: %u, Primary plane: %u\n\n", crtc_id, primary_plane_id);

  // --- Allocate GBM buffers ---
  uint32_t w = mode->hdisplay;
  uint32_t h = mode->vdisplay;
  std::string active_flags = gbm_flags_to_str(cfg.gbm_flags);
  printf("[TEST] Allocating %d GBM buffer(s) (%ux%u, flags=%s)...\n",
         cfg.num_buffers, w, h, active_flags.c_str());

  std::vector<GbmBuffer> buffers(cfg.num_buffers);
  // Alternate green shades so page flips are visible
  uint32_t colors[] = {0xFF00FF00, 0xFF00CC00, 0xFF00AA00, 0xFF008800,
                       0xFF00FF44, 0xFF00DD22, 0xFF00BB11, 0xFF009900,
                       0xFF00FF88, 0xFF00EE66, 0xFF00CC44, 0xFF00AA22,
                       0xFF00FF00, 0xFF00CC00, 0xFF00AA00, 0xFF008800};

  for (int i = 0; i < cfg.num_buffers; i++) {
    if (!create_gbm_buffer(gbm, fd, &buffers[i], w, h,
                           GBM_FORMAT_ARGB8888, cfg.gbm_flags,
                           DRM_FORMAT_ARGB8888)) {
      printf("  [FAIL] Buffer %d allocation failed.\n", i);
      for (int j = 0; j < i; j++) destroy_gbm_buffer(fd, &buffers[j]);
      drmModeFreeConnector(conn);
      drmModeFreeResources(res);
      gbm_device_destroy(gbm);
      close(fd);
      return 1;
    }
    printf("  Buffer %d: bo=%p, dmabuf_fd=%d, stride=%u, fb=%u\n",
           i, static_cast<void*>(buffers[i].bo),
           buffers[i].dmabuf_fd, buffers[i].stride, buffers[i].fb_id);
    fill_gbm_buffer(&buffers[i], colors[i % 16]);
  }
  printf("  [PASS] All %d GBM buffers allocated and mapped.\n\n",
         cfg.num_buffers);

  // --- Initial modeset ---
  printf("[TEST] Setting CRTC mode...\n");
  if (drmModeSetCrtc(fd, crtc_id, buffers[0].fb_id, 0, 0,
                     &conn->connector_id, 1, mode) != 0) {
    printf("  [FAIL] drmModeSetCrtc: %s\n", strerror(errno));
    for (auto& b : buffers) destroy_gbm_buffer(fd, &b);
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    gbm_device_destroy(gbm);
    close(fd);
    return 1;
  }
  printf("  [PASS] CRTC modeset succeeded.\n\n");

  // --- Atomic commit ---
  printf("[TEST] Atomic commit with GBM framebuffer...\n");
  drmModeAtomicReqPtr req = drmModeAtomicAlloc();
  auto add_plane_prop = [&](const char* name, uint64_t val) {
    uint32_t pid = find_plane_prop(fd, primary_plane_id, name);
    if (pid) drmModeAtomicAddProperty(req, primary_plane_id, pid, val);
  };
  add_plane_prop("FB_ID", buffers[0].fb_id);
  add_plane_prop("CRTC_ID", crtc_id);
  add_plane_prop("SRC_X", 0);
  add_plane_prop("SRC_Y", 0);
  add_plane_prop("SRC_W", static_cast<uint64_t>(w) << 16);
  add_plane_prop("SRC_H", static_cast<uint64_t>(h) << 16);
  add_plane_prop("CRTC_X", 0);
  add_plane_prop("CRTC_Y", 0);
  add_plane_prop("CRTC_W", w);
  add_plane_prop("CRTC_H", h);

  int ret = drmModeAtomicCommit(fd, req, DRM_MODE_ATOMIC_ALLOW_MODESET,
                                nullptr);
  drmModeAtomicFree(req);
  if (ret != 0) {
    printf("  [FAIL] drmModeAtomicCommit: %s\n", strerror(errno));
    for (auto& b : buffers) destroy_gbm_buffer(fd, &b);
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    gbm_device_destroy(gbm);
    close(fd);
    return 1;
  }
  printf("  [PASS] Atomic commit succeeded.\n\n");

  // --- Page-flip loop ---
  printf("[TEST] Page-flip loop for %d seconds (%d buffers)...\n",
         cfg.duration, cfg.num_buffers);
  printf(">>> A GREEN screen should be visible on the display. <<<\n\n");

  auto t_start = std::chrono::steady_clock::now();
  auto t_end = t_start + std::chrono::seconds(cfg.duration);
  uint32_t flip_count = 0;
  uint32_t flip_errors = 0;
  double min_ms = 1e9, max_ms = 0, sum_ms = 0;
  auto t_prev = t_start;
  int buf_idx = 0;

  while (std::chrono::steady_clock::now() < t_end) {
    buf_idx = (buf_idx + 1) % cfg.num_buffers;

    drmModeAtomicReqPtr flip = drmModeAtomicAlloc();
    if (!flip) break;
    uint32_t fb_prop = find_plane_prop(fd, primary_plane_id, "FB_ID");
    if (fb_prop)
      drmModeAtomicAddProperty(flip, primary_plane_id,
                               fb_prop, buffers[buf_idx].fb_id);

    int flip_ret = drmModeAtomicCommit(fd, flip,
                                       DRM_MODE_PAGE_FLIP_EVENT, nullptr);
    drmModeAtomicFree(flip);

    if (flip_ret == 0) {
      drmEventContext ev_ctx = {};
      ev_ctx.version = 2;
      ev_ctx.page_flip_handler = [](int, unsigned int,
                                    unsigned int, unsigned int,
                                    void*) {};
      fd_set fds;
      FD_ZERO(&fds);
      FD_SET(fd, &fds);
      struct timeval tv = {1, 0};
      if (select(fd + 1, &fds, nullptr, nullptr, &tv) > 0)
        drmHandleEvent(fd, &ev_ctx);

      auto t_now = std::chrono::steady_clock::now();
      double dt = std::chrono::duration<double, std::milli>(
                      t_now - t_prev).count();
      t_prev = t_now;
      if (flip_count > 0) {
        min_ms = std::min(min_ms, dt);
        max_ms = std::max(max_ms, dt);
        sum_ms += dt;
      }
      flip_count++;
    } else {
      flip_errors++;
      usleep(1000);
    }
  }

  double elapsed_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t_start).count();

  // --- Summary ---
  printf("==========================================\n");
  printf("  GBM + DRM Platform Test Results\n");
  printf("==========================================\n");
  printf("  Mode:       %ux%u @ %u Hz\n",
         mode->hdisplay, mode->vdisplay, mode_refresh_hz(mode));
  printf("  Connector:  %s\n", connector_name_str(conn).c_str());
  if (!selected_monitor.empty())
    printf("  Monitor:    %s\n", selected_monitor.c_str());
  printf("  GBM flags:  %s (0x%x)\n",
         active_flags.c_str(), cfg.gbm_flags);
  printf("  Buffers:    %d\n", cfg.num_buffers);
  for (int i = 0; i < cfg.num_buffers; i++) {
    printf("    [%d] stride=%u dmabuf_fd=%d fb=%u\n",
           i, buffers[i].stride, buffers[i].dmabuf_fd, buffers[i].fb_id);
  }
  printf("  ------------------------------------------\n");
  printf("  Page-Flip Summary\n");
  printf("  ------------------------------------------\n");
  printf("  Duration:     %.1f s\n", elapsed_s);
  printf("  Total flips:  %u\n", flip_count);
  printf("  Flip errors:  %u\n", flip_errors);
  if (flip_count > 1) {
    double avg_ms = sum_ms / (flip_count - 1);
    printf("  Interval:     min=%.2f ms  avg=%.2f ms  max=%.2f ms\n",
           min_ms, avg_ms, max_ms);
    printf("  Avg FPS:      %.1f\n", 1000.0 / avg_ms);
  }
  printf("==========================================\n");
  if (flip_count > 0)
    printf("[PASS] GBM + DRM page-flip loop completed (%u flips).\n",
           flip_count);
  else
    printf("[FAIL] No successful page flips.\n");

  // --- Cleanup ---
  printf("\n[INFO] Cleaning up...\n");
  for (auto& b : buffers) destroy_gbm_buffer(fd, &b);
  drmModeFreeConnector(conn);
  drmModeFreeResources(res);
  gbm_device_destroy(gbm);
  close(fd);

  printf("[DONE] GBM + DRM platform validation test completed.\n");
  return 0;
}
