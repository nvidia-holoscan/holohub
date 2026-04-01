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
// DRM Platform Validation Test
//
// Validates that the DRM/KMS subsystem on this platform supports the
// display features required by cuDisp. This is a platform capability
// test, not a cuDisp library test.
//
// Usage:
//   drm_basic_test [options]
//
// Options:
//   --card <N>               DRI card number (default: auto-detect)
//   --display <name>         Connector (e.g. DP-2) or monitor name (e.g. "ASUS VG27A")
//   --mode <WxH>             Display mode resolution (default: preferred mode)
//   --refresh <Hz>           Refresh rate in Hz (default: highest available)
//   --vrr                    Enable Variable Refresh Rate on the CRTC
//   --overlay                Render an overlay plane on top of primary
//   --scale <WxH>            Scale source buffer to given output size
//   --alpha <0-65535>        Plane alpha value (default: 65535 = opaque)
//   --blend <none|premul|coverage>  Pixel blend mode
//   --rotate <0|90|180|270>  Rotation angle in degrees
//   --color-encoding <bt601|bt709|bt2020>  (accepted but not tested)
//   --color-range <limited|full>            (accepted but not tested)
//   --duration <seconds>     How long to hold the display (default: 5)
//   --help                   Show this help
//

#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <getopt.h>
#include <algorithm>
#include <chrono>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>
#include <vector>

// ============================================================================
// DRM connector type name table
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

// ============================================================================
// Dumb buffer helper
// ============================================================================
struct DumbBuffer {
  uint32_t handle = 0;
  uint32_t fb_id = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t stride = 0;
  uint64_t size = 0;
  void* map = nullptr;
  int fd = -1;
};

static bool create_dumb_buffer(int drm_fd, DumbBuffer* buf,
                               uint32_t w, uint32_t h, uint32_t bpp) {
  struct drm_mode_create_dumb create = {};
  create.width = w;
  create.height = h;
  create.bpp = bpp;
  if (drmIoctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create) < 0) {
    printf("  [FAIL] DRM_IOCTL_MODE_CREATE_DUMB: %s\n", strerror(errno));
    return false;
  }
  buf->handle = create.handle;
  buf->stride = create.pitch;
  buf->size = create.size;
  buf->width = w;
  buf->height = h;
  buf->fd = drm_fd;

  struct drm_mode_map_dumb map_req = {};
  map_req.handle = buf->handle;
  if (drmIoctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map_req) < 0) {
    printf("  [FAIL] DRM_IOCTL_MODE_MAP_DUMB: %s\n", strerror(errno));
    return false;
  }
  buf->map = mmap(nullptr, buf->size, PROT_READ | PROT_WRITE,
                  MAP_SHARED, drm_fd, map_req.offset);
  if (buf->map == MAP_FAILED) {
    printf("  [FAIL] mmap: %s\n", strerror(errno));
    buf->map = nullptr;
    return false;
  }
  return true;
}

static bool add_framebuffer(int drm_fd, DumbBuffer* buf) {
  uint32_t handles[4] = {buf->handle, 0, 0, 0};
  uint32_t strides[4] = {buf->stride, 0, 0, 0};
  uint32_t offsets[4] = {0, 0, 0, 0};
  if (drmModeAddFB2(drm_fd, buf->width, buf->height,
                    DRM_FORMAT_ARGB8888,
                    handles, strides, offsets,
                    &buf->fb_id, 0) != 0) {
    printf("  [FAIL] drmModeAddFB2: %s\n", strerror(errno));
    return false;
  }
  return true;
}

static void fill_solid(DumbBuffer* buf, uint32_t argb) {
  auto* px = static_cast<uint32_t*>(buf->map);
  uint32_t count = buf->stride / 4 * buf->height;
  for (uint32_t i = 0; i < count; i++) px[i] = argb;
}

static void destroy_dumb_buffer(DumbBuffer* buf) {
  if (buf->fb_id) {
    drmModeRmFB(buf->fd, buf->fb_id);
    buf->fb_id = 0;
  }
  if (buf->map) {
    munmap(buf->map, buf->size);
    buf->map = nullptr;
  }
  if (buf->handle) {
    struct drm_mode_destroy_dumb destroy = {};
    destroy.handle = buf->handle;
    drmIoctl(buf->fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
    buf->handle = 0;
  }
}

// ============================================================================
// Plane property helpers
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

static uint32_t find_crtc_prop(int fd, uint32_t crtc_id,
                               const char* name) {
  drmModeObjectPropertiesPtr props =
      drmModeObjectGetProperties(fd, crtc_id, DRM_MODE_OBJECT_CRTC);
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

// Extract the monitor model name from the EDID blob property on a connector.
// EDID descriptor tag 0xFC = "Monitor Name". Returns empty string if not found.
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
        // Standard EDID: 4 descriptor blocks starting at offset 54, 18 bytes each
        for (int d = 0; d < 4; d++) {
          int off = 54 + d * 18;
          if (off + 18 > static_cast<int>(blob->length)) break;
          // Tag 0xFC = Monitor Name descriptor
          if (edid[off] == 0 && edid[off + 1] == 0 &&
              edid[off + 2] == 0 && edid[off + 3] == 0xFC) {
            char name[14] = {};
            memcpy(name, &edid[off + 5], 13);
            // Trim trailing newline/spaces
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
// Auto-detect DRI card
// ============================================================================
static int open_drm_device(int card_num) {
  char path[64];
  if (card_num >= 0) {
    snprintf(path, sizeof(path), "/dev/dri/card%d", card_num);
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
      printf("[ERROR] Cannot open %s: %s\n", path, strerror(errno));
    }
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
// Config
// ============================================================================
struct Config {
  int card_num = -1;
  std::string display_name;
  uint32_t mode_w = 0;
  uint32_t mode_h = 0;
  int refresh_hz = 0;
  bool vrr = false;
  bool overlay = false;
  uint32_t scale_w = 0;
  uint32_t scale_h = 0;
  uint16_t alpha = 0xFFFF;
  std::string blend;
  int rotate = 0;
  std::string color_encoding;
  std::string color_range;
  int duration = 5;
};

static void print_help() {
  printf(
      "DRM Platform Validation Test\n"
      "\n"
      "Validates DRM/KMS display features on this platform.\n"
      "\n"
      "Options:\n"
      "  --card <N>             DRI card number (default: auto)\n"
      "  --display <name>       Connector (DP-2) or monitor name\n"
      "  --mode <WxH>           Resolution (default: preferred)\n"
      "  --refresh <Hz>         Refresh rate (default: highest)\n"
      "  --vrr                  Enable Variable Refresh Rate\n"
      "  --overlay              Render overlay plane on primary\n"
      "  --scale <WxH>          Scale to output size\n"
      "  --alpha <0-65535>      Plane alpha (default: 65535)\n"
      "  --blend <none|premul|coverage>\n"
      "  --rotate <0|90|180|270>\n"
      "  --color-encoding <bt601|bt709|bt2020>  (not tested)\n"
      "  --color-range <limited|full>            (not tested)\n"
      "  --duration <sec>       Hold duration (default: 5)\n"
      "  --help\n");
}

static bool parse_args(int argc, char** argv, Config* cfg) {
  static struct option long_opts[] = {
    {"card",           required_argument, nullptr, 'c'},
    {"display",        required_argument, nullptr, 'D'},
    {"mode",           required_argument, nullptr, 'm'},
    {"refresh",        required_argument, nullptr, 'r'},
    {"vrr",            no_argument,       nullptr, 'v'},
    {"overlay",        no_argument,       nullptr, 'o'},
    {"scale",          required_argument, nullptr, 's'},
    {"alpha",          required_argument, nullptr, 'a'},
    {"blend",          required_argument, nullptr, 'b'},
    {"rotate",         required_argument, nullptr, 'R'},
    {"color-encoding", required_argument, nullptr, 'E'},
    {"color-range",    required_argument, nullptr, 'G'},
    {"duration",       required_argument, nullptr, 'd'},
    {"help",           no_argument,       nullptr, 'h'},
    {nullptr, 0, nullptr, 0}
  };
  int opt;
  while ((opt = getopt_long(argc, argv, "hc:D:m:r:vos:a:b:R:d:",
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
      case 'r': cfg->refresh_hz = atoi(optarg); break;
      case 'v': cfg->vrr = true; break;
      case 'o': cfg->overlay = true; break;
      case 's':
        if (sscanf(optarg, "%ux%u", &cfg->scale_w, &cfg->scale_h) != 2) {
          printf("[ERROR] Invalid --scale format. Use WxH.\n");
          return false;
        }
        break;
      case 'a': cfg->alpha = static_cast<uint16_t>(atoi(optarg)); break;
      case 'b': cfg->blend = optarg; break;
      case 'R': cfg->rotate = atoi(optarg); break;
      case 'E': cfg->color_encoding = optarg; break;
      case 'G': cfg->color_range = optarg; break;
      case 'd': cfg->duration = atoi(optarg); break;
      case 'h': print_help(); exit(0);
      default: return false;
    }
  }
  return true;
}

// ============================================================================
// Main test
// ============================================================================
int main(int argc, char** argv) {
  printf("=========================================="
         "==========================================\n");
  printf("  DRM Platform Validation Test\n");
  printf("  Validates DRM/KMS subsystem capabilities"
         " on this platform.\n");
  printf("=========================================="
         "==========================================\n\n");

  Config cfg;
  if (!parse_args(argc, argv, &cfg)) return 1;

  if (!cfg.color_encoding.empty()) {
    printf("[NOTE] --color-encoding=%s accepted but "
           "not tested in this version.\n",
           cfg.color_encoding.c_str());
  }
  if (!cfg.color_range.empty()) {
    printf("[NOTE] --color-range=%s accepted but "
           "not tested in this version.\n",
           cfg.color_range.c_str());
  }

  // --- Open DRM device ---
  printf("[TEST] Opening DRM device...\n");
  int fd = open_drm_device(cfg.card_num);
  if (fd < 0) {
    printf("  [FAIL] No usable DRM device found.\n");
    return 1;
  }
  printf("  [PASS] DRM device opened (fd=%d)\n\n", fd);

  // --- Enable atomic modesetting and universal planes ---
  printf("[TEST] Enabling DRM atomic modesetting...\n");
  if (drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) != 0) {
    printf("  [FAIL] DRM_CLIENT_CAP_UNIVERSAL_PLANES: %s\n",
           strerror(errno));
    close(fd);
    return 1;
  }
  if (drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1) != 0) {
    printf("  [FAIL] DRM_CLIENT_CAP_ATOMIC: %s\n", strerror(errno));
    close(fd);
    return 1;
  }
  printf("  [PASS] Atomic modesetting enabled.\n\n");

  // --- Find connector ---
  printf("[TEST] Enumerating connectors...\n");
  drmModeRes* res = drmModeGetResources(fd);
  if (!res) {
    printf("  [FAIL] drmModeGetResources: %s\n", strerror(errno));
    close(fd);
    return 1;
  }

  drmModeConnector* conn = nullptr;
  std::string selected_monitor_name;
  for (int i = 0; i < res->count_connectors; i++) {
    drmModeConnector* c =
        drmModeGetConnector(fd, res->connectors[i]);
    if (!c) continue;
    std::string cname = connector_name_str(c);
    std::string mname = get_edid_monitor_name(fd, c->connector_id);
    printf("  Connector: %-12s  monitor=\"%s\"  status=%s  modes=%d\n",
           cname.c_str(),
           mname.empty() ? "(none)" : mname.c_str(),
           c->connection == DRM_MODE_CONNECTED ? "connected" :
           c->connection == DRM_MODE_DISCONNECTED ? "disconnected" :
           "unknown",
           c->count_modes);
    if (c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) {
      bool match = cfg.display_name.empty() ||
                   cfg.display_name == cname ||
                   (!mname.empty() && cfg.display_name == mname);
      if (match) {
        conn = c;
        selected_monitor_name = mname;
        printf("  --> Selected: %s", cname.c_str());
        if (!mname.empty()) printf(" (%s)", mname.c_str());
        printf("\n");
        continue;
      }
    }
    drmModeFreeConnector(c);
  }
  if (!conn) {
    if (!cfg.display_name.empty()) {
      printf("  [FAIL] No connected display matching \"%s\".\n",
             cfg.display_name.c_str());
      printf("         Specify a connector name (e.g. DP-2) or "
             "monitor name from the list above.\n");
    } else {
      printf("  [FAIL] No connected display found.\n");
    }
    drmModeFreeResources(res);
    close(fd);
    return 1;
  }
  printf("  [PASS] Connector %s", connector_name_str(conn).c_str());
  if (!selected_monitor_name.empty())
    printf(" (%s)", selected_monitor_name.c_str());
  printf(" found with %d modes.\n\n", conn->count_modes);

  // --- Select mode ---
  printf("[TEST] Selecting display mode...\n");
  drmModeModeInfo* mode = nullptr;
  drmModeModeInfo* preferred = nullptr;
  drmModeModeInfo* best_match = nullptr;
  uint32_t best_refresh = 0;

  for (int i = 0; i < conn->count_modes; i++) {
    drmModeModeInfo* m = &conn->modes[i];
    uint32_t hz = mode_refresh_hz(m);
    printf("  Mode: %ux%u @ %u Hz%s\n",
           m->hdisplay, m->vdisplay, hz,
           (m->type & DRM_MODE_TYPE_PREFERRED) ? " (preferred)" : "");

    if (m->type & DRM_MODE_TYPE_PREFERRED) preferred = m;

    bool res_match = (cfg.mode_w == 0 && cfg.mode_h == 0) ||
                     (m->hdisplay == cfg.mode_w &&
                      m->vdisplay == cfg.mode_h);
    if (!res_match) continue;

    bool refresh_match = (cfg.refresh_hz == 0) ||
        (static_cast<int>(hz) >= cfg.refresh_hz - 1 &&
         static_cast<int>(hz) <= cfg.refresh_hz + 1);

    if (cfg.refresh_hz > 0 && refresh_match) {
      best_match = m;
    } else if (cfg.refresh_hz == 0 && hz > best_refresh) {
      best_refresh = hz;
      best_match = m;
    }
  }

  if (best_match) {
    mode = best_match;
  } else if (cfg.mode_w == 0 && preferred) {
    mode = preferred;
  }

  if (!mode) {
    printf("  [FAIL] No matching mode found for %ux%u",
           cfg.mode_w, cfg.mode_h);
    if (cfg.refresh_hz > 0) printf(" @ %d Hz", cfg.refresh_hz);
    printf(".\n");
    printf("  Available modes on this display:\n");
    for (int i = 0; i < conn->count_modes; i++) {
      drmModeModeInfo* m = &conn->modes[i];
      printf("    %ux%u @ %u Hz%s\n",
             m->hdisplay, m->vdisplay, mode_refresh_hz(m),
             (m->type & DRM_MODE_TYPE_PREFERRED) ? " (preferred)" : "");
    }
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    close(fd);
    return 1;
  }
  printf("  [PASS] Selected mode: %ux%u @ %u Hz\n\n",
         mode->hdisplay, mode->vdisplay, mode_refresh_hz(mode));

  // --- Find CRTC ---
  printf("[TEST] Finding CRTC for connector...\n");
  uint32_t crtc_id = 0;
  if (conn->encoder_id) {
    drmModeEncoder* enc = drmModeGetEncoder(fd, conn->encoder_id);
    if (enc) {
      crtc_id = enc->crtc_id;
      drmModeFreeEncoder(enc);
    }
  }
  if (!crtc_id && res->count_crtcs > 0) {
    for (int i = 0; i < res->count_crtcs; i++) {
      if (conn->encoders && conn->count_encoders > 0) {
        drmModeEncoder* enc =
            drmModeGetEncoder(fd, conn->encoders[0]);
        if (enc && (enc->possible_crtcs & (1u << i))) {
          crtc_id = res->crtcs[i];
          drmModeFreeEncoder(enc);
          break;
        }
        if (enc) drmModeFreeEncoder(enc);
      }
    }
  }
  if (!crtc_id) {
    printf("  [FAIL] No CRTC available for this connector.\n");
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    close(fd);
    return 1;
  }
  printf("  [PASS] CRTC ID: %u\n\n", crtc_id);

  // --- Find primary plane (and overlay if requested) ---
  printf("[TEST] Finding planes...\n");
  drmModePlaneRes* plane_res = drmModeGetPlaneResources(fd);
  if (!plane_res) {
    printf("  [FAIL] drmModeGetPlaneResources: %s\n", strerror(errno));
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    close(fd);
    return 1;
  }

  uint32_t primary_plane_id = 0;
  uint32_t overlay_plane_id = 0;
  int crtc_index = -1;
  for (int i = 0; i < res->count_crtcs; i++) {
    if (res->crtcs[i] == crtc_id) { crtc_index = i; break; }
  }

  for (uint32_t i = 0; i < plane_res->count_planes; i++) {
    drmModePlane* p = drmModeGetPlane(fd, plane_res->planes[i]);
    if (!p) continue;
    if (crtc_index >= 0 &&
        !(p->possible_crtcs & (1u << crtc_index))) {
      drmModeFreePlane(p);
      continue;
    }
    int ptype = get_plane_type(fd, p->plane_id);
    const char* tname = ptype == 0 ? "Overlay" :
                        ptype == 1 ? "Primary" :
                        ptype == 2 ? "Cursor" : "Unknown";
    printf("  Plane %u: type=%s\n", p->plane_id, tname);
    if (ptype == 1 && !primary_plane_id) primary_plane_id = p->plane_id;
    if (ptype == 0 && !overlay_plane_id) overlay_plane_id = p->plane_id;
    drmModeFreePlane(p);
  }
  drmModeFreePlaneResources(plane_res);

  if (!primary_plane_id) {
    printf("  [FAIL] No primary plane found for CRTC %u.\n", crtc_id);
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    close(fd);
    return 1;
  }
  printf("  [PASS] Primary plane: %u\n", primary_plane_id);
  if (cfg.overlay) {
    if (!overlay_plane_id) {
      printf("  [FAIL] --overlay requested but no overlay "
             "plane found for CRTC %u.\n", crtc_id);
      drmModeFreeConnector(conn);
      drmModeFreeResources(res);
      close(fd);
      return 1;
    }
    printf("  [PASS] Overlay plane: %u\n", overlay_plane_id);
  }
  printf("\n");

  // --- Create dumb buffers ---
  uint32_t buf_w = mode->hdisplay;
  uint32_t buf_h = mode->vdisplay;

  printf("[TEST] Creating primary dumb buffer (%ux%u)...\n", buf_w, buf_h);
  DumbBuffer primary_buf;
  if (!create_dumb_buffer(fd, &primary_buf, buf_w, buf_h, 32)) {
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    close(fd);
    return 1;
  }
  // Green = success indicator
  fill_solid(&primary_buf, 0xFF00FF00);
  if (!add_framebuffer(fd, &primary_buf)) {
    destroy_dumb_buffer(&primary_buf);
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    close(fd);
    return 1;
  }
  printf("  [PASS] Primary buffer: handle=%u fb=%u stride=%u\n\n",
         primary_buf.handle, primary_buf.fb_id, primary_buf.stride);

  DumbBuffer overlay_buf;
  if (cfg.overlay) {
    uint32_t ov_w = buf_w / 2;
    uint32_t ov_h = buf_h / 2;
    printf("[TEST] Creating overlay dumb buffer (%ux%u)...\n", ov_w, ov_h);
    if (!create_dumb_buffer(fd, &overlay_buf, ov_w, ov_h, 32)) {
      destroy_dumb_buffer(&primary_buf);
      drmModeFreeConnector(conn);
      drmModeFreeResources(res);
      close(fd);
      return 1;
    }
    // Semi-transparent blue
    fill_solid(&overlay_buf, 0x800000FF);
    if (!add_framebuffer(fd, &overlay_buf)) {
      destroy_dumb_buffer(&overlay_buf);
      destroy_dumb_buffer(&primary_buf);
      drmModeFreeConnector(conn);
      drmModeFreeResources(res);
      close(fd);
      return 1;
    }
    printf("  [PASS] Overlay buffer: handle=%u fb=%u stride=%u\n\n",
           overlay_buf.handle, overlay_buf.fb_id, overlay_buf.stride);
  }

  // --- Initial modeset ---
  printf("[TEST] Setting CRTC mode (%ux%u @ %u Hz)...\n",
         mode->hdisplay, mode->vdisplay, mode_refresh_hz(mode));
  if (drmModeSetCrtc(fd, crtc_id, primary_buf.fb_id, 0, 0,
                     &conn->connector_id, 1, mode) != 0) {
    printf("  [FAIL] drmModeSetCrtc: %s\n", strerror(errno));
    if (cfg.overlay) destroy_dumb_buffer(&overlay_buf);
    destroy_dumb_buffer(&primary_buf);
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    close(fd);
    return 1;
  }
  printf("  [PASS] CRTC modeset succeeded.\n\n");

  // --- Atomic commit with plane properties ---
  printf("[TEST] Atomic commit with plane properties...\n");

  drmModeAtomicReqPtr req = drmModeAtomicAlloc();
  if (!req) {
    printf("  [FAIL] drmModeAtomicAlloc failed.\n");
    if (cfg.overlay) destroy_dumb_buffer(&overlay_buf);
    destroy_dumb_buffer(&primary_buf);
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    close(fd);
    return 1;
  }

  uint32_t out_w = (cfg.scale_w > 0) ? cfg.scale_w : mode->hdisplay;
  uint32_t out_h = (cfg.scale_h > 0) ? cfg.scale_h : mode->vdisplay;

  // Primary plane properties
  auto add_prop = [&](uint32_t obj, const char* name,
                      uint64_t val, const char* obj_type) {
    uint32_t prop_id = 0;
    if (strcmp(obj_type, "plane") == 0) {
      prop_id = find_plane_prop(fd, obj, name);
    } else {
      prop_id = find_crtc_prop(fd, obj, name);
    }
    if (!prop_id) {
      printf("  [WARN] Property '%s' not found on %s %u\n",
             name, obj_type, obj);
      return false;
    }
    drmModeAtomicAddProperty(req, obj, prop_id, val);
    return true;
  };

  add_prop(primary_plane_id, "FB_ID", primary_buf.fb_id, "plane");
  add_prop(primary_plane_id, "CRTC_ID", crtc_id, "plane");
  add_prop(primary_plane_id, "SRC_X", 0, "plane");
  add_prop(primary_plane_id, "SRC_Y", 0, "plane");
  add_prop(primary_plane_id, "SRC_W",
           static_cast<uint64_t>(buf_w) << 16, "plane");
  add_prop(primary_plane_id, "SRC_H",
           static_cast<uint64_t>(buf_h) << 16, "plane");
  add_prop(primary_plane_id, "CRTC_X", 0, "plane");
  add_prop(primary_plane_id, "CRTC_Y", 0, "plane");
  add_prop(primary_plane_id, "CRTC_W", out_w, "plane");
  add_prop(primary_plane_id, "CRTC_H", out_h, "plane");

  printf("  Primary: src=%ux%u -> crtc=%ux%u%s\n",
         buf_w, buf_h, out_w, out_h,
         (cfg.scale_w > 0) ? " (SCALED)" : "");

  // Alpha
  if (cfg.alpha != 0xFFFF) {
    if (add_prop(primary_plane_id, "alpha", cfg.alpha, "plane")) {
      printf("  Alpha: %u\n", cfg.alpha);
    }
  }

  // Blend mode
  if (!cfg.blend.empty()) {
    uint64_t blend_val = 0;
    if (cfg.blend == "none") blend_val = 0;
    else if (cfg.blend == "premul") blend_val = 1;
    else if (cfg.blend == "coverage") blend_val = 2;
    if (add_prop(primary_plane_id, "pixel blend mode",
                 blend_val, "plane")) {
      printf("  Blend mode: %s\n", cfg.blend.c_str());
    }
  }

  // Rotation
  if (cfg.rotate != 0) {
    uint64_t rot_val = 0;
    if (cfg.rotate == 90) rot_val = 1 << 1;       // DRM_MODE_ROTATE_90
    else if (cfg.rotate == 180) rot_val = 1 << 2;  // DRM_MODE_ROTATE_180
    else if (cfg.rotate == 270) rot_val = 1 << 3;  // DRM_MODE_ROTATE_270
    if (add_prop(primary_plane_id, "rotation", rot_val, "plane")) {
      printf("  Rotation: %d degrees\n", cfg.rotate);
    }
  }

  // VRR: requires both CRTC support (VRR_ENABLED property) and
  // monitor capability (vrr_capable connector property).
  bool vrr_supported = false;
  bool vrr_capable = false;
  bool vrr_active = false;
  uint32_t vrr_enabled_prop_id = 0;
  {
    drmModeObjectPropertiesPtr crtc_props =
        drmModeObjectGetProperties(fd, crtc_id, DRM_MODE_OBJECT_CRTC);
    if (crtc_props) {
      for (uint32_t i = 0; i < crtc_props->count_props; i++) {
        drmModePropertyPtr prop =
            drmModeGetProperty(fd, crtc_props->props[i]);
        if (prop && strcmp(prop->name, "VRR_ENABLED") == 0) {
          vrr_enabled_prop_id = prop->prop_id;
          vrr_supported = true;
        }
        if (prop) drmModeFreeProperty(prop);
      }
      drmModeFreeObjectProperties(crtc_props);
    }

    drmModeObjectPropertiesPtr conn_props =
        drmModeObjectGetProperties(fd, conn->connector_id,
                                   DRM_MODE_OBJECT_CONNECTOR);
    if (conn_props) {
      for (uint32_t i = 0; i < conn_props->count_props; i++) {
        drmModePropertyPtr prop =
            drmModeGetProperty(fd, conn_props->props[i]);
        if (prop && strcmp(prop->name, "vrr_capable") == 0) {
          vrr_capable = (conn_props->prop_values[i] != 0);
        }
        if (prop) drmModeFreeProperty(prop);
      }
      drmModeFreeObjectProperties(conn_props);
    }

    if (vrr_supported && vrr_capable) {
      printf("  VRR/G-Sync is supported and available.\n");
    } else if (vrr_supported) {
      printf("  VRR/G-Sync property exists but display "
             "is not capable.\n");
    } else {
      printf("  VRR/G-Sync is not supported.\n");
    }

    if (cfg.vrr) {
      vrr_active = cfg.vrr && vrr_supported && vrr_capable;
      if (vrr_active) {
        drmModeAtomicAddProperty(req, crtc_id,
                                 vrr_enabled_prop_id, 1);
        printf("  VRR: enabled on CRTC %u\n", crtc_id);
      } else {
        printf("  VRR requested but not available. Skipping.\n");
      }
    }
  }

  // Overlay plane
  if (cfg.overlay) {
    uint32_t ov_w = buf_w / 2;
    uint32_t ov_h = buf_h / 2;
    uint32_t ov_x = buf_w / 4;
    uint32_t ov_y = buf_h / 4;
    add_prop(overlay_plane_id, "FB_ID", overlay_buf.fb_id, "plane");
    add_prop(overlay_plane_id, "CRTC_ID", crtc_id, "plane");
    add_prop(overlay_plane_id, "SRC_X", 0, "plane");
    add_prop(overlay_plane_id, "SRC_Y", 0, "plane");
    add_prop(overlay_plane_id, "SRC_W",
             static_cast<uint64_t>(ov_w) << 16, "plane");
    add_prop(overlay_plane_id, "SRC_H",
             static_cast<uint64_t>(ov_h) << 16, "plane");
    add_prop(overlay_plane_id, "CRTC_X", ov_x, "plane");
    add_prop(overlay_plane_id, "CRTC_Y", ov_y, "plane");
    add_prop(overlay_plane_id, "CRTC_W", ov_w, "plane");
    add_prop(overlay_plane_id, "CRTC_H", ov_h, "plane");
    printf("  Overlay: %ux%u at (%u,%u) - semi-transparent blue\n",
           ov_w, ov_h, ov_x, ov_y);
  }

  uint32_t flags = DRM_MODE_ATOMIC_ALLOW_MODESET;
  int ret = drmModeAtomicCommit(fd, req, flags, nullptr);
  drmModeAtomicFree(req);

  if (ret != 0) {
    printf("  [FAIL] drmModeAtomicCommit: %s (errno=%d)\n",
           strerror(errno), errno);

    // Fall back to red to signal failure visually
    fill_solid(&primary_buf, 0xFFFF0000);
    drmModeSetCrtc(fd, crtc_id, primary_buf.fb_id, 0, 0,
                   &conn->connector_id, 1, mode);
    printf("\n");
    printf("  >>> A RED screen should be visible on the display. <<<\n");
    printf("  >>> This indicates the atomic commit FAILED.       <<<\n");
    printf("  Holding for %d seconds...\n", cfg.duration);
    sleep(cfg.duration);

    if (cfg.overlay) destroy_dumb_buffer(&overlay_buf);
    destroy_dumb_buffer(&primary_buf);
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    close(fd);
    return 1;
  }
  printf("  [PASS] Atomic commit succeeded.\n\n");

  // --- Display result summary ---
  printf("==========================================\n");
  printf("  DRM Platform Test Results\n");
  printf("==========================================\n");
  printf("  Mode:       %ux%u @ %u Hz\n",
         mode->hdisplay, mode->vdisplay, mode_refresh_hz(mode));
  printf("  Connector:  %s\n", connector_name_str(conn).c_str());
  if (!selected_monitor_name.empty())
    printf("  Monitor:    %s\n", selected_monitor_name.c_str());
  printf("  CRTC:       %u\n", crtc_id);
  printf("  Primary:    plane %u, fb %u\n",
         primary_plane_id, primary_buf.fb_id);

  if (cfg.scale_w > 0) {
    printf("  Scaling:    %ux%u -> %ux%u  [TESTED]\n",
           buf_w, buf_h, out_w, out_h);
  }
  if (cfg.alpha != 0xFFFF) {
    printf("  Alpha:      %u  [TESTED]\n", cfg.alpha);
  }
  if (!cfg.blend.empty()) {
    printf("  Blend:      %s  [TESTED]\n", cfg.blend.c_str());
  }
  if (cfg.rotate != 0) {
    printf("  Rotation:   %d deg  [TESTED]\n", cfg.rotate);
  }
  if (cfg.vrr) {
    if (vrr_active)
      printf("  VRR:        enabled  [TESTED]\n");
    else
      printf("  VRR:        requested but not supported  [SKIPPED]\n");
  }
  if (cfg.overlay) {
    printf("  Overlay:    plane %u, fb %u  [TESTED]\n",
           overlay_plane_id, overlay_buf.fb_id);
  }
  if (!cfg.color_encoding.empty()) {
    printf("  Color enc:  %s  [NOT TESTED]\n",
           cfg.color_encoding.c_str());
  }
  if (!cfg.color_range.empty()) {
    printf("  Color range: %s  [NOT TESTED]\n",
           cfg.color_range.c_str());
  }
  printf("==========================================\n\n");

  printf(">>> A GREEN screen should be visible on the display.  <<<\n");
  if (cfg.overlay) {
    printf(">>> A semi-transparent BLUE rectangle should appear   <<<\n");
    printf(">>> centered on top of the green background.          <<<\n");
  }
  if (cfg.scale_w > 0) {
    printf(">>> The image is scaled from %ux%u to %ux%u.%s <<<\n",
           buf_w, buf_h, out_w, out_h,
           std::string(11, ' ').c_str());
  }
  printf(">>> If the screen is RED, the atomic commit failed.   <<<\n");
  printf(">>> If nothing is displayed, DRM modeset failed.      <<<\n");
  printf("\n");

  // Page-flip loop: alternate between two buffers for the full duration.
  // Validates the DRM page-flip path and reports timing statistics.
  printf("[TEST] Page-flip loop for %d seconds%s...\n",
         cfg.duration, vrr_active ? " (VRR enabled)" : "");

  DumbBuffer primary_buf2;
  bool have_buf2 = false;
  if (create_dumb_buffer(fd, &primary_buf2, buf_w, buf_h, 32)) {
    fill_solid(&primary_buf2, 0xFF00EE00);
    if (add_framebuffer(fd, &primary_buf2)) {
      have_buf2 = true;
    } else {
      destroy_dumb_buffer(&primary_buf2);
    }
  }
  if (!have_buf2) {
    printf("  [WARN] Could not allocate second buffer; "
           "flipping same buffer.\n");
  }

  uint32_t fb_a = primary_buf.fb_id;
  uint32_t fb_b = have_buf2 ? primary_buf2.fb_id : primary_buf.fb_id;
  bool use_a = true;

  auto t_start = std::chrono::steady_clock::now();
  auto t_end = t_start + std::chrono::seconds(cfg.duration);
  uint32_t flip_count = 0;
  uint32_t flip_errors = 0;
  double min_ms = 1e9, max_ms = 0, sum_ms = 0;
  auto t_prev = t_start;

  while (std::chrono::steady_clock::now() < t_end) {
    drmModeAtomicReqPtr flip_req = drmModeAtomicAlloc();
    if (!flip_req) break;

    uint32_t fb = use_a ? fb_a : fb_b;
    uint32_t fb_prop = find_plane_prop(fd, primary_plane_id, "FB_ID");
    if (fb_prop)
      drmModeAtomicAddProperty(flip_req, primary_plane_id, fb_prop, fb);

    int flip_ret = drmModeAtomicCommit(
        fd, flip_req, DRM_MODE_PAGE_FLIP_EVENT, nullptr);
    drmModeAtomicFree(flip_req);

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
      use_a = !use_a;
    } else {
      flip_errors++;
      usleep(1000);
    }
  }

  double elapsed_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t_start).count();
  printf("  ------------------------------------------\n");
  printf("  Page-Flip Summary%s\n",
         vrr_active ? " (VRR)" : "");
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
  printf("  ------------------------------------------\n");
  if (flip_count > 0)
    printf("  [PASS] Page-flip loop completed (%u flips).\n", flip_count);
  else
    printf("  [FAIL] No successful page flips.\n");

  if (have_buf2) destroy_dumb_buffer(&primary_buf2);

  // --- Cleanup ---
  printf("\n[INFO] Cleaning up...\n");
  if (cfg.overlay) destroy_dumb_buffer(&overlay_buf);
  destroy_dumb_buffer(&primary_buf);
  drmModeFreeConnector(conn);
  drmModeFreeResources(res);
  close(fd);

  printf("[DONE] DRM platform validation test completed "
         "successfully.\n");
  return 0;
}
