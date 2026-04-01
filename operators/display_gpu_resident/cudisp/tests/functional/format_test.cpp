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
// DRM Format Validation Test
//
// Queries the primary plane for all supported pixel formats, then
// allocates a dumb buffer, fills it with color bars, creates a
// framebuffer, and displays each one. Reports PASS/FAIL/SKIP per
// format. Use --format to test a specific format, even one not
// advertised by the driver.
//
// Usage:
//   format_test [options]
//
// Options:
//   --card <N>           DRI card number (default: auto-detect)
//   --display <name>     Connector or monitor name
//   --mode <WxH>         Display mode (default: preferred)
//   --duration <sec>     Per-format hold time (default: 2)
//   --format <name>      Test only this format (e.g. DRM_FORMAT_ARGB8888)
//   --list               List supported formats and exit
//   --help
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
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

// ============================================================================
// DRM fourcc helpers
// ============================================================================

// Convert a DRM fourcc to a printable 4-character string
static std::string fourcc_str(uint32_t fourcc) {
  char s[5] = {
    static_cast<char>(fourcc & 0xFF),
    static_cast<char>((fourcc >> 8) & 0xFF),
    static_cast<char>((fourcc >> 16) & 0xFF),
    static_cast<char>((fourcc >> 24) & 0xFF),
    '\0'
  };
  return std::string(s);
}

struct FormatInfo {
  uint32_t fourcc;
  uint32_t bpp;        // bits per pixel for single-plane dumb buffer alloc
  const char* name;    // DRM_FORMAT_xxx string
  bool can_fill;       // whether we know how to fill this format
};

static FormatInfo get_format_info(uint32_t fourcc) {
  switch (fourcc) {
    case DRM_FORMAT_RGB565:
      return {fourcc, 16, "DRM_FORMAT_RGB565", true};
    case DRM_FORMAT_ARGB1555:
      return {fourcc, 16, "DRM_FORMAT_ARGB1555", true};
    case DRM_FORMAT_XRGB1555:
      return {fourcc, 16, "DRM_FORMAT_XRGB1555", true};
    case DRM_FORMAT_ARGB8888:
      return {fourcc, 32, "DRM_FORMAT_ARGB8888", true};
    case DRM_FORMAT_XRGB8888:
      return {fourcc, 32, "DRM_FORMAT_XRGB8888", true};
    case DRM_FORMAT_ABGR8888:
      return {fourcc, 32, "DRM_FORMAT_ABGR8888", true};
    case DRM_FORMAT_XBGR8888:
      return {fourcc, 32, "DRM_FORMAT_XBGR8888", true};
    case DRM_FORMAT_ABGR2101010:
      return {fourcc, 32, "DRM_FORMAT_ABGR2101010", true};
    case DRM_FORMAT_XBGR2101010:
      return {fourcc, 32, "DRM_FORMAT_XBGR2101010", true};
    case DRM_FORMAT_ARGB2101010:
      return {fourcc, 32, "DRM_FORMAT_ARGB2101010", true};
    case DRM_FORMAT_XRGB2101010:
      return {fourcc, 32, "DRM_FORMAT_XRGB2101010", true};
    case DRM_FORMAT_ABGR16161616:
      return {fourcc, 64, "DRM_FORMAT_ABGR16161616", true};
    case DRM_FORMAT_XBGR16161616:
      return {fourcc, 64, "DRM_FORMAT_XBGR16161616", true};
    case DRM_FORMAT_ABGR16161616F:
      return {fourcc, 64, "DRM_FORMAT_ABGR16161616F", true};
    case DRM_FORMAT_XBGR16161616F:
      return {fourcc, 64, "DRM_FORMAT_XBGR16161616F", true};
    case DRM_FORMAT_YUYV:
      return {fourcc, 16, "DRM_FORMAT_YUYV", true};
    case DRM_FORMAT_UYVY:
      return {fourcc, 16, "DRM_FORMAT_UYVY", true};
    case DRM_FORMAT_NV12:
      return {fourcc, 12, "DRM_FORMAT_NV12", true};
    case DRM_FORMAT_NV21:
      return {fourcc, 12, "DRM_FORMAT_NV21", true};
    case DRM_FORMAT_NV16:
      return {fourcc, 16, "DRM_FORMAT_NV16", true};
    case DRM_FORMAT_NV61:
      return {fourcc, 16, "DRM_FORMAT_NV61", true};
    case DRM_FORMAT_NV24:
      return {fourcc, 24, "DRM_FORMAT_NV24", true};
    case DRM_FORMAT_NV42:
      return {fourcc, 24, "DRM_FORMAT_NV42", true};
    case DRM_FORMAT_P010:
      return {fourcc, 15, "DRM_FORMAT_P010", true};
    case DRM_FORMAT_P012:
      return {fourcc, 15, "DRM_FORMAT_P012", true};
    case DRM_FORMAT_P210:
      return {fourcc, 20, "DRM_FORMAT_P210", true};
    default:
      return {fourcc, 0, nullptr, false};
  }
}

// Fill a dumb buffer with color bars: Red | Green | Blue | White | Black
// Each bar is 1/5 of the width.
static void fill_color_bars_32bpp(void* map, uint32_t stride, uint32_t w,
                                  uint32_t h, uint32_t fourcc) {
  auto* pixels = static_cast<uint8_t*>(map);
  uint32_t bar_w = w / 5;

  // Encode ARGB/XRGB colors based on channel order
  // DRM_FORMAT_ARGB8888: memory layout B G R A (little-endian: 0xAARRGGBB)
  // DRM_FORMAT_ABGR8888: memory layout R G B A (little-endian: 0xAABBGGRR)
  uint32_t red, green, blue, white, black;
  if (fourcc == DRM_FORMAT_ABGR8888 || fourcc == DRM_FORMAT_XBGR8888) {
    red   = 0xFF0000FF; green = 0xFF00FF00; blue  = 0xFFFF0000;
    white = 0xFFFFFFFF; black = 0xFF000000;
  } else if (fourcc == DRM_FORMAT_ABGR2101010 ||
             fourcc == DRM_FORMAT_XBGR2101010) {
    // 10-bit: A(2) B(10) G(10) R(10) -> pack as 0xC0000000 | B<<20 | G<<10 | R
    red   = 0xC00003FF; green = 0xC00FFC00; blue  = 0xFFF00000;
    white = 0xFFFFFFFF; black = 0xC0000000;
  } else if (fourcc == DRM_FORMAT_ARGB2101010 ||
             fourcc == DRM_FORMAT_XRGB2101010) {
    // A(2) R(10) G(10) B(10)
    red   = 0xFFF00000; green = 0xC00FFC00; blue  = 0xC00003FF;
    white = 0xFFFFFFFF; black = 0xC0000000;
  } else {
    // ARGB8888 / XRGB8888 (default)
    red   = 0xFFFF0000; green = 0xFF00FF00; blue  = 0xFF0000FF;
    white = 0xFFFFFFFF; black = 0xFF000000;
  }

  uint32_t colors[5] = {red, green, blue, white, black};
  for (uint32_t y = 0; y < h; y++) {
    auto* row = reinterpret_cast<uint32_t*>(pixels + y * stride);
    for (uint32_t x = 0; x < w; x++) {
      int bar = (x / bar_w);
      if (bar > 4) bar = 4;
      row[x] = colors[bar];
    }
  }
}

static void fill_color_bars_16bpp(void* map, uint32_t stride, uint32_t w,
                                  uint32_t h, uint32_t fourcc) {
  auto* pixels = static_cast<uint8_t*>(map);
  uint32_t bar_w = w / 5;

  uint16_t red, green, blue, white, black;
  if (fourcc == DRM_FORMAT_RGB565) {
    // R(5) G(6) B(5)
    red   = 0xF800; green = 0x07E0; blue  = 0x001F;
    white = 0xFFFF; black = 0x0000;
  } else if (fourcc == DRM_FORMAT_ARGB1555 ||
             fourcc == DRM_FORMAT_XRGB1555) {
    // A(1) R(5) G(5) B(5)
    red   = 0xFC00; green = 0x83E0; blue  = 0x801F;
    white = 0xFFFF; black = 0x8000;
  } else {
    red = 0xFFFF; green = 0; blue = 0; white = 0xFFFF; black = 0;
  }

  uint16_t colors[5] = {red, green, blue, white, black};
  for (uint32_t y = 0; y < h; y++) {
    auto* row = reinterpret_cast<uint16_t*>(pixels + y * stride);
    for (uint32_t x = 0; x < w; x++) {
      int bar = (x / bar_w);
      if (bar > 4) bar = 4;
      row[x] = colors[bar];
    }
  }
}

static void fill_color_bars_64bpp(void* map, uint32_t stride, uint32_t w,
                                  uint32_t h, uint32_t fourcc) {
  auto* pixels = static_cast<uint8_t*>(map);
  uint32_t bar_w = w / 5;

  // ABGR16161616 / XBGR16161616: R(16) G(16) B(16) A(16) in memory
  // Pack as two uint32_t per pixel: [RG, BA] little-endian
  // Simpler: treat as uint64_t
  uint64_t red, green, blue, white, black;
  if (fourcc == DRM_FORMAT_ABGR16161616 ||
      fourcc == DRM_FORMAT_XBGR16161616) {
    // Memory: R16 G16 B16 A16
    red   = 0xFFFF00000000FFFFULL;  // R=FFFF, G=0, B=0, A=FFFF
    green = 0xFFFF0000FFFF0000ULL;  // R=0, G=FFFF, B=0, A=FFFF
    blue  = 0xFFFFFFFF00000000ULL;  // R=0, G=0, B=FFFF, A=FFFF
    white = 0xFFFFFFFFFFFFFFFFULL;
    black = 0xFFFF000000000000ULL;  // R=0, G=0, B=0, A=FFFF
  } else {
    // ABGR16161616F: same layout but half-float values
    // half-float 1.0 = 0x3C00, 0.0 = 0x0000
    uint16_t one = 0x3C00;
    uint16_t zero = 0x0000;
    auto pack = [](uint16_t r, uint16_t g, uint16_t b, uint16_t a) -> uint64_t {
      return static_cast<uint64_t>(r) |
             (static_cast<uint64_t>(g) << 16) |
             (static_cast<uint64_t>(b) << 32) |
             (static_cast<uint64_t>(a) << 48);
    };
    red   = pack(one, zero, zero, one);
    green = pack(zero, one, zero, one);
    blue  = pack(zero, zero, one, one);
    white = pack(one, one, one, one);
    black = pack(zero, zero, zero, one);
  }

  uint64_t colors[5] = {red, green, blue, white, black};
  for (uint32_t y = 0; y < h; y++) {
    auto* row = reinterpret_cast<uint64_t*>(pixels + y * stride);
    for (uint32_t x = 0; x < w; x++) {
      int bar = (x / bar_w);
      if (bar > 4) bar = 4;
      row[x] = colors[bar];
    }
  }
}

// BT.601 RGB->YCbCr conversion for color bar values
struct YuvColor { uint8_t y; uint8_t cb; uint8_t cr; };
static YuvColor rgb_to_yuv(uint8_t r, uint8_t g, uint8_t b) {
  int y  = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
  int cb = ((-38 * r -  74 * g + 112 * b + 128) >> 8) + 128;
  int cr = ((112 * r -  94 * g -  18 * b + 128) >> 8) + 128;
  auto clamp = [](int v) -> uint8_t {
    return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
  };
  return {clamp(y), clamp(cb), clamp(cr)};
}

// Fill packed YUYV: each 4 bytes = Y0 Cb Y1 Cr (2 pixels)
static void fill_color_bars_yuyv(void* map, uint32_t stride,
                                 uint32_t w, uint32_t h) {
  auto* pixels = static_cast<uint8_t*>(map);
  uint32_t bar_w = w / 5;
  // Red, Green, Blue, White, Black
  uint8_t rgb[5][3] = {
      {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 255}, {0, 0, 0}};
  YuvColor yuv[5];
  for (int i = 0; i < 5; i++)
    yuv[i] = rgb_to_yuv(rgb[i][0], rgb[i][1], rgb[i][2]);

  for (uint32_t y = 0; y < h; y++) {
    auto* row = pixels + y * stride;
    for (uint32_t x = 0; x < w; x += 2) {
      int bar = (x / bar_w);
      if (bar > 4) bar = 4;
      row[x*2 + 0] = yuv[bar].y;
      row[x*2 + 1] = yuv[bar].cb;
      row[x*2 + 2] = yuv[bar].y;
      row[x*2 + 3] = yuv[bar].cr;
    }
  }
}

// Fill packed UYVY: each 4 bytes = Cb Y0 Cr Y1
static void fill_color_bars_uyvy(void* map, uint32_t stride,
                                 uint32_t w, uint32_t h) {
  auto* pixels = static_cast<uint8_t*>(map);
  uint32_t bar_w = w / 5;
  uint8_t rgb[5][3] = {
      {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 255}, {0, 0, 0}};
  YuvColor yuv[5];
  for (int i = 0; i < 5; i++)
    yuv[i] = rgb_to_yuv(rgb[i][0], rgb[i][1], rgb[i][2]);

  for (uint32_t y = 0; y < h; y++) {
    auto* row = pixels + y * stride;
    for (uint32_t x = 0; x < w; x += 2) {
      int bar = (x / bar_w);
      if (bar > 4) bar = 4;
      row[x*2 + 0] = yuv[bar].cb;
      row[x*2 + 1] = yuv[bar].y;
      row[x*2 + 2] = yuv[bar].cr;
      row[x*2 + 3] = yuv[bar].y;
    }
  }
}

// Fill 8-bit semi-planar NV formats (NV12/21/16/61/24/42).
// chroma_w/chroma_h control subsampling; cb_first selects CbCr vs CrCb.
static void fill_color_bars_nv(void* y_map, uint32_t y_stride,
                               void* uv_map, uint32_t uv_stride,
                               uint32_t w, uint32_t h,
                               uint32_t chroma_w, uint32_t chroma_h,
                               bool cb_first) {
  auto* y_pixels = static_cast<uint8_t*>(y_map);
  auto* uv_pixels = static_cast<uint8_t*>(uv_map);
  uint32_t bar_w = w / 5;
  uint8_t rgb[5][3] = {
      {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 255}, {0, 0, 0}};
  YuvColor yuv[5];
  for (int i = 0; i < 5; i++)
    yuv[i] = rgb_to_yuv(rgb[i][0], rgb[i][1], rgb[i][2]);

  for (uint32_t y = 0; y < h; y++) {
    auto* row = y_pixels + y * y_stride;
    for (uint32_t x = 0; x < w; x++) {
      int bar = (x / bar_w);
      if (bar > 4) bar = 4;
      row[x] = yuv[bar].y;
    }
  }

  uint32_t uv_h = h / chroma_h;
  uint32_t uv_w = w / chroma_w;
  for (uint32_t y = 0; y < uv_h; y++) {
    auto* row = uv_pixels + y * uv_stride;
    for (uint32_t x = 0; x < uv_w; x++) {
      int bar = ((x * chroma_w) / bar_w);
      if (bar > 4) bar = 4;
      if (cb_first) {
        row[x*2 + 0] = yuv[bar].cb;
        row[x*2 + 1] = yuv[bar].cr;
      } else {
        row[x*2 + 0] = yuv[bar].cr;
        row[x*2 + 1] = yuv[bar].cb;
      }
    }
  }
}

// Fill 16-bit semi-planar formats (P010, P012, P210).
// 10/12-bit values stored in the upper bits of 16-bit words.
// chroma_w/chroma_h specify the subsampling factors.
static void fill_color_bars_p0xx(void* y_map, uint32_t y_stride,
                                 void* uv_map, uint32_t uv_stride,
                                 uint32_t w, uint32_t h,
                                 uint32_t chroma_w, uint32_t chroma_h) {
  auto* y_pixels = static_cast<uint8_t*>(y_map);
  auto* uv_pixels = static_cast<uint8_t*>(uv_map);
  uint32_t bar_w = w / 5;
  uint8_t rgb[5][3] = {
      {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 255}, {0, 0, 0}};
  YuvColor yuv[5];
  for (int i = 0; i < 5; i++)
    yuv[i] = rgb_to_yuv(rgb[i][0], rgb[i][1], rgb[i][2]);

  for (uint32_t y = 0; y < h; y++) {
    auto* row = reinterpret_cast<uint16_t*>(y_pixels + y * y_stride);
    for (uint32_t x = 0; x < w; x++) {
      int bar = (x / bar_w);
      if (bar > 4) bar = 4;
      row[x] = static_cast<uint16_t>(yuv[bar].y) << 8;
    }
  }

  uint32_t uv_h = h / chroma_h;
  uint32_t uv_w = w / chroma_w;
  for (uint32_t y = 0; y < uv_h; y++) {
    auto* row = reinterpret_cast<uint16_t*>(uv_pixels + y * uv_stride);
    for (uint32_t x = 0; x < uv_w; x++) {
      int bar = ((x * chroma_w) / bar_w);
      if (bar > 4) bar = 4;
      row[x*2 + 0] = static_cast<uint16_t>(yuv[bar].cb) << 8;
      row[x*2 + 1] = static_cast<uint16_t>(yuv[bar].cr) << 8;
    }
  }
}

// Multi-plane dumb buffer layout. For single-plane formats, only
// plane 0 is used. For semi-planar YUV, both planes are laid out
// sequentially in a single tall dumb buffer with the correct strides
// and offsets for drmModeAddFB2.
struct DumbPlaneLayout {
  uint32_t alloc_bpp;       // bpp for CREATE_DUMB
  uint32_t alloc_height;    // total rows to allocate (Y + chroma)
  int      num_planes;
  uint32_t plane_stride[2];  // filled after allocation (from pitch)
  uint32_t plane_offset[2];  // byte offsets within the buffer
  uint32_t chroma_w;        // horizontal subsampling (1 or 2)
  uint32_t chroma_h;        // vertical subsampling (1 or 2)
  bool     is_16bit_yuv;    // P010/P012/P210 (16-bit per sample)
  bool     cb_first;        // CbCr vs CrCb ordering
};

static DumbPlaneLayout get_dumb_layout(uint32_t fourcc, uint32_t h,
                                       uint32_t bpp) {
  DumbPlaneLayout l = {};
  l.num_planes = 1;
  l.alloc_bpp = bpp;
  l.alloc_height = h;
  l.chroma_w = 1;
  l.chroma_h = 1;

  switch (fourcc) {
    case DRM_FORMAT_NV12: case DRM_FORMAT_NV21:  // 8-bit 4:2:0
      l.num_planes = 2; l.alloc_bpp = 8;
      l.chroma_w = 2; l.chroma_h = 2;
      l.alloc_height = h + h / 2;
      l.cb_first = (fourcc == DRM_FORMAT_NV12);
      break;
    case DRM_FORMAT_NV16: case DRM_FORMAT_NV61:  // 8-bit 4:2:2
      l.num_planes = 2; l.alloc_bpp = 8;
      l.chroma_w = 2; l.chroma_h = 1;
      l.alloc_height = h * 2;
      l.cb_first = (fourcc == DRM_FORMAT_NV16);
      break;
    case DRM_FORMAT_NV24: case DRM_FORMAT_NV42:  // 8-bit 4:4:4
      l.num_planes = 2; l.alloc_bpp = 8;
      l.chroma_w = 1; l.chroma_h = 1;
      l.alloc_height = h * 3;  // UV stride = 2 * Y stride
      l.cb_first = (fourcc == DRM_FORMAT_NV24);
      break;
    case DRM_FORMAT_P010: case DRM_FORMAT_P012:  // 16-bit 4:2:0
      l.num_planes = 2; l.alloc_bpp = 16;
      l.chroma_w = 2; l.chroma_h = 2;
      l.alloc_height = h + h / 2;
      l.is_16bit_yuv = true; l.cb_first = true;
      break;
    case DRM_FORMAT_P210:                         // 16-bit 4:2:2
      l.num_planes = 2; l.alloc_bpp = 16;
      l.chroma_w = 2; l.chroma_h = 1;
      l.alloc_height = h * 2;
      l.is_16bit_yuv = true; l.cb_first = true;
      break;
    default:
      break;
  }
  return l;
}

static void finalize_dumb_layout(DumbPlaneLayout* l, uint32_t pitch,
                                 uint32_t h) {
  l->plane_stride[0] = pitch;
  l->plane_offset[0] = 0;
  if (l->num_planes >= 2) {
    l->plane_offset[1] = pitch * h;
    if (l->chroma_w == 1 && !l->is_16bit_yuv) {
      l->plane_stride[1] = pitch * 2;  // 4:4:4 NV24/NV42
    } else {
      l->plane_stride[1] = pitch;
    }
  }
}

// Allocate a dumb buffer, fill it with color bars, create DRM FB, and display.
static bool test_format_dumb(int fd, uint32_t crtc_id, uint32_t conn_id,
                             drmModeModeInfo* mode, FormatInfo* fi,
                             int duration) {
  uint32_t w = mode->hdisplay;
  uint32_t h = mode->vdisplay;
  uint32_t fourcc = fi->fourcc;

  DumbPlaneLayout layout = get_dumb_layout(fourcc, h, fi->bpp);

  struct drm_mode_create_dumb create = {};
  create.width = w;
  create.height = layout.alloc_height;
  create.bpp = layout.alloc_bpp;
  if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &create) < 0) {
    printf("  [FAIL] CREATE_DUMB: %s\n", strerror(errno));
    return false;
  }

  finalize_dumb_layout(&layout, create.pitch, h);

  struct drm_mode_map_dumb map_req = {};
  map_req.handle = create.handle;
  if (drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &map_req) < 0) {
    printf("  [FAIL] MAP_DUMB: %s\n", strerror(errno));
    struct drm_mode_destroy_dumb d = {};
    d.handle = create.handle;
    drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
    return false;
  }

  void* map = mmap(nullptr, create.size, PROT_READ | PROT_WRITE,
                   MAP_SHARED, fd, map_req.offset);
  if (map == MAP_FAILED) {
    printf("  [FAIL] mmap: %s\n", strerror(errno));
    struct drm_mode_destroy_dumb d = {};
    d.handle = create.handle;
    drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
    return false;
  }

  // Fill the buffer
  auto* base = static_cast<uint8_t*>(map);
  if (fourcc == DRM_FORMAT_YUYV) {
    fill_color_bars_yuyv(base, create.pitch, w, h);
  } else if (fourcc == DRM_FORMAT_UYVY) {
    fill_color_bars_uyvy(base, create.pitch, w, h);
  } else if (layout.num_planes == 2 && layout.is_16bit_yuv) {
    fill_color_bars_p0xx(base + layout.plane_offset[0], layout.plane_stride[0],
                         base + layout.plane_offset[1], layout.plane_stride[1],
                         w, h, layout.chroma_w, layout.chroma_h);
  } else if (layout.num_planes == 2) {
    fill_color_bars_nv(base + layout.plane_offset[0], layout.plane_stride[0],
                       base + layout.plane_offset[1], layout.plane_stride[1],
                       w, h, layout.chroma_w, layout.chroma_h,
                       layout.cb_first);
  } else if (fi->bpp == 16) {
    fill_color_bars_16bpp(base, create.pitch, w, h, fourcc);
  } else if (fi->bpp == 32) {
    fill_color_bars_32bpp(base, create.pitch, w, h, fourcc);
  } else if (fi->bpp == 64) {
    fill_color_bars_64bpp(base, create.pitch, w, h, fourcc);
  }

  // Build per-plane handles/strides/offsets for AddFB2
  uint32_t handles[4] = {};
  uint32_t strides[4] = {};
  uint32_t offsets[4] = {};
  for (int p = 0; p < layout.num_planes; p++) {
    handles[p] = create.handle;
    strides[p] = layout.plane_stride[p];
    offsets[p] = layout.plane_offset[p];
  }

  printf("  Dumb buffer: %d plane(s), ", layout.num_planes);
  for (int p = 0; p < layout.num_planes; p++)
    printf("plane%d: stride=%u offset=%u ", p, strides[p], offsets[p]);
  printf("size=%llu\n", static_cast<unsigned long long>(create.size));

  uint32_t fb_id = 0;
  if (drmModeAddFB2(fd, w, h, fourcc,
                    handles, strides, offsets, &fb_id, 0) != 0) {
    printf("  [FAIL] drmModeAddFB2: %s\n", strerror(errno));
    munmap(map, create.size);
    struct drm_mode_destroy_dumb d = {};
    d.handle = create.handle;
    drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
    return false;
  }

  int ret = drmModeSetCrtc(fd, crtc_id, fb_id, 0, 0,
                           &conn_id, 1, mode);
  if (ret != 0) {
    printf("  [FAIL] drmModeSetCrtc: %s\n", strerror(errno));
    drmModeRmFB(fd, fb_id);
    munmap(map, create.size);
    struct drm_mode_destroy_dumb d = {};
    d.handle = create.handle;
    drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
    return false;
  }

  printf("  >>> Color bars: Red | Green | Blue | White | Black <<<\n");
  printf("  >>> If colors appear swapped, the channel order differs. <<<\n");
  printf("  [PASS] Displayed. Holding %d seconds...\n", duration);
  sleep(duration);

  drmModeRmFB(fd, fb_id);
  munmap(map, create.size);
  struct drm_mode_destroy_dumb d = {};
  d.handle = create.handle;
  drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
  return true;
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
// Config
// ============================================================================
struct Config {
  int card_num = -1;
  std::string display_name;
  uint32_t mode_w = 0;
  uint32_t mode_h = 0;
  int duration = 2;
  std::string filter_format;
  bool list_only = false;
};

static void print_help() {
  printf(
      "DRM Format Validation Test\n"
      "\n"
      "Tests all pixel formats supported by the primary plane.\n"
      "\n"
      "Options:\n"
      "  --card <N>         DRI card number (default: auto)\n"
      "  --display <name>   Connector or monitor name\n"
      "  --mode <WxH>       Resolution (default: preferred)\n"
      "  --duration <sec>   Per-format hold (default: 2)\n"
      "  --format <name>    Test single format (e.g. DRM_FORMAT_ARGB8888)\n"
      "  --list             List supported formats and exit\n"
      "  --help\n");
}

static bool parse_args(int argc, char** argv, Config* cfg) {
  static struct option long_opts[] = {
    {"card",     required_argument, nullptr, 'c'},
    {"display",  required_argument, nullptr, 'D'},
    {"mode",     required_argument, nullptr, 'm'},
    {"duration", required_argument, nullptr, 'd'},
    {"format",   required_argument, nullptr, 'f'},
    {"list",     no_argument,       nullptr, 'l'},
    {"help",     no_argument,       nullptr, 'h'},
    {nullptr, 0, nullptr, 0}
  };
  int opt;
  while ((opt = getopt_long(argc, argv, "hc:D:m:d:f:l",
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
      case 'd': cfg->duration = atoi(optarg); break;
      case 'f': cfg->filter_format = optarg; break;
      case 'l': cfg->list_only = true; break;
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
  printf("  DRM Format Validation Test\n");
  printf("  Tests pixel formats supported by the"
         " primary plane on this platform.\n");
  printf("=========================================="
         "==========================================\n\n");

  Config cfg;
  if (!parse_args(argc, argv, &cfg)) return 1;

  int fd = open_drm_device(cfg.card_num);
  if (fd < 0) { printf("[FAIL] No DRM device.\n"); return 1; }

  drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
  drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1);

  drmModeRes* res = drmModeGetResources(fd);
  if (!res) { printf("[FAIL] drmModeGetResources\n"); close(fd); return 1; }

  // Find connector
  drmModeConnector* conn = nullptr;
  for (int i = 0; i < res->count_connectors; i++) {
    drmModeConnector* c = drmModeGetConnector(fd, res->connectors[i]);
    if (!c) continue;
    std::string cname = connector_name_str(c);
    std::string mname = get_edid_monitor_name(fd, c->connector_id);
    if (c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) {
      bool match = cfg.display_name.empty() ||
                   cfg.display_name == cname ||
                   (!mname.empty() && cfg.display_name == mname);
      if (match && !conn) { conn = c; continue; }
    }
    drmModeFreeConnector(c);
  }
  if (!conn) {
    printf("[FAIL] No connected display.\n");
    drmModeFreeResources(res); close(fd); return 1;
  }

  // Find mode
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
    printf("[FAIL] No matching mode.\n");
    drmModeFreeConnector(conn); drmModeFreeResources(res); close(fd);
    return 1;
  }
  printf("[INFO] Mode: %ux%u @ %u Hz on %s\n\n",
         mode->hdisplay, mode->vdisplay, mode_refresh_hz(mode),
         connector_name_str(conn).c_str());

  // Find CRTC
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

  // Find primary plane and get its supported formats
  int crtc_index = -1;
  for (int i = 0; i < res->count_crtcs; i++)
    if (res->crtcs[i] == crtc_id) { crtc_index = i; break; }

  std::vector<uint32_t> plane_formats;
  uint32_t primary_plane = 0;
  drmModePlaneRes* pr = drmModeGetPlaneResources(fd);
  if (pr) {
    for (uint32_t i = 0; i < pr->count_planes; i++) {
      drmModePlane* p = drmModeGetPlane(fd, pr->planes[i]);
      if (!p) continue;
      if (crtc_index >= 0 && !(p->possible_crtcs & (1u << crtc_index))) {
        drmModeFreePlane(p); continue;
      }
      if (get_plane_type(fd, p->plane_id) == 1 && !primary_plane) {
        primary_plane = p->plane_id;
        for (uint32_t f = 0; f < p->count_formats; f++)
          plane_formats.push_back(p->formats[f]);
      }
      drmModeFreePlane(p);
    }
    drmModeFreePlaneResources(pr);
  }

  if (!primary_plane || plane_formats.empty()) {
    printf("[FAIL] No primary plane or no formats.\n");
    drmModeFreeConnector(conn); drmModeFreeResources(res); close(fd);
    return 1;
  }

  printf("[INFO] Primary plane %u supports %zu format(s):\n",
         primary_plane, plane_formats.size());
  for (auto fmt : plane_formats) {
    FormatInfo fi = get_format_info(fmt);
    printf("  %-28s %s (0x%08x)%s\n",
           fi.name ? fi.name : "unknown",
           fourcc_str(fmt).c_str(), fmt,
           fi.can_fill ? "" : "  [no fill support]");
  }
  printf("\n");

  if (cfg.list_only) {
    drmModeFreeConnector(conn); drmModeFreeResources(res); close(fd);
    return 0;
  }

  // If --format requests a format not in the driver's list, add it
  // so we can still attempt the test.
  if (!cfg.filter_format.empty()) {
    bool found = false;
    for (auto fmt : plane_formats) {
      std::string fcc = fourcc_str(fmt);
      FormatInfo fi = get_format_info(fmt);
      if (cfg.filter_format == fcc ||
          (fi.name && cfg.filter_format == fi.name)) {
        found = true;
        break;
      }
    }
    if (!found) {
      // Look up the fourcc from our format table by name
      static const uint32_t all_known[] = {
        DRM_FORMAT_RGB565, DRM_FORMAT_ARGB1555, DRM_FORMAT_XRGB1555,
        DRM_FORMAT_ARGB8888, DRM_FORMAT_XRGB8888,
        DRM_FORMAT_ABGR8888, DRM_FORMAT_XBGR8888,
        DRM_FORMAT_ABGR2101010, DRM_FORMAT_XBGR2101010,
        DRM_FORMAT_ARGB2101010, DRM_FORMAT_XRGB2101010,
        DRM_FORMAT_ABGR16161616, DRM_FORMAT_XBGR16161616,
        DRM_FORMAT_ABGR16161616F, DRM_FORMAT_XBGR16161616F,
        DRM_FORMAT_YUYV, DRM_FORMAT_UYVY,
        DRM_FORMAT_NV12, DRM_FORMAT_NV21, DRM_FORMAT_NV16, DRM_FORMAT_NV61,
        DRM_FORMAT_NV24, DRM_FORMAT_NV42,
        DRM_FORMAT_P010, DRM_FORMAT_P012, DRM_FORMAT_P210,
      };
      for (auto kf : all_known) {
        FormatInfo fi = get_format_info(kf);
        std::string fcc = fourcc_str(kf);
        if (cfg.filter_format == fcc ||
            (fi.name && cfg.filter_format == fi.name)) {
          printf("[INFO] %s is not advertised by the driver's primary plane.\n"
                 "       Attempting anyway...\n\n", fi.name);
          plane_formats.push_back(kf);
          break;
        }
      }
    }
  }

  // --- Test each format ---
  int tested = 0, passed = 0, failed = 0, skipped = 0;

  for (auto fmt : plane_formats) {
    std::string fcc = fourcc_str(fmt);
    FormatInfo fi = get_format_info(fmt);

    if (!cfg.filter_format.empty() &&
        cfg.filter_format != fcc &&
        (!fi.name || cfg.filter_format != fi.name)) {
      continue;
    }

    printf("[TEST] Format: %s (%s, 0x%08x)\n",
           fi.name ? fi.name : "unknown", fcc.c_str(), fmt);
    tested++;

    if (!fi.can_fill) {
      printf("  [SKIP] Unknown format, no fill support.\n\n");
      skipped++;
      continue;
    }

    if (test_format_dumb(fd, crtc_id, conn->connector_id,
                         mode, &fi, cfg.duration)) {
      passed++;
    } else {
      failed++;
    }
    printf("\n");
  }

  // --- Summary ---
  printf("==========================================\n");
  printf("  Format Test Summary\n");
  printf("==========================================\n");
  printf("  Tested:   %d\n", tested);
  printf("  Passed:   %d\n", passed);
  printf("  Failed:   %d\n", failed);
  printf("  Skipped:  %d\n", skipped);
  printf("==========================================\n");

  drmModeFreeConnector(conn);
  drmModeFreeResources(res);
  close(fd);

  printf("[DONE] Format validation test completed.\n");
  return (failed > 0) ? 1 : 0;
}
