// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "x11_display/x11_display.hpp"

#include <X11/keysym.h>

#include <array>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>

#include "v4l2_depth_common/tensor_utils.hpp"

namespace holoscan::examples::v4l2_depth {
namespace {

[[nodiscard]] holoscan::expected<void, holoscan::Error> failure(
    holoscan::ErrorCode code, std::string message) {
  return holoscan::make_unexpected(holoscan::Error{code, std::move(message)});
}

}  // namespace

X11DisplayOp::X11DisplayOp(int width, int height, std::string title)
    : width_(width), height_(height), title_(std::move(title)) {
  if (width_ <= 0 || height_ <= 0) {
    throw std::invalid_argument("X11 display width and height must be positive");
  }
  if (width_ > std::numeric_limits<int>::max() / 4) {
    throw std::invalid_argument("X11 display row byte count overflows int");
  }
  const auto unsigned_width = static_cast<std::size_t>(width_);
  const auto unsigned_height = static_cast<std::size_t>(height_);
  if (unsigned_width > std::numeric_limits<std::size_t>::max() / 4U ||
      unsigned_height >
          std::numeric_limits<std::size_t>::max() / (unsigned_width * 4U)) {
    throw std::invalid_argument("X11 image byte count overflows size_t");
  }
  frame_bytes_ = unsigned_width * unsigned_height * 4U;
}

X11DisplayOp::~X11DisplayOp() { cleanup(); }

void X11DisplayOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(input, "input")
      .queue_depth(2U)
      .expects_tensor(holoscan::TensorInputSpec{
          .representation = {
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = kUInt8Dtype,
              .rank = 3U,
          },
      });
}

holoscan::Contract X11DisplayOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{input});
  return result;
}

void X11DisplayOp::start() {
  cleanup();
  disabled_ = false;

  display_ = XOpenDisplay(nullptr);
  if (display_ == nullptr) {
    disable("XOpenDisplay failed; DISPLAY is unset or inaccessible");
    return;
  }

  const int screen = DefaultScreen(display_);
  Visual* visual = DefaultVisual(display_, screen);
  const int depth = DefaultDepth(display_, screen);
  window_ = XCreateSimpleWindow(display_,
                                RootWindow(display_, screen),
                                0,
                                0,
                                static_cast<unsigned int>(width_),
                                static_cast<unsigned int>(height_),
                                1U,
                                BlackPixel(display_, screen),
                                BlackPixel(display_, screen));
  if (window_ == 0U) {
    disable("XCreateSimpleWindow failed");
    return;
  }

  XStoreName(display_, window_, title_.c_str());
  XSelectInput(display_, window_, ExposureMask | KeyPressMask | StructureNotifyMask);
  wm_delete_window_ = XInternAtom(display_, "WM_DELETE_WINDOW", False);
  if (wm_delete_window_ != None) {
    Atom protocol = wm_delete_window_;
    static_cast<void>(XSetWMProtocols(display_, window_, &protocol, 1));
  }

  gc_ = XCreateGC(display_, window_, 0UL, nullptr);
  if (gc_ == nullptr) {
    disable("XCreateGC failed");
    return;
  }

  const int row_bytes = width_ * 4;
  image_ = XCreateImage(display_,
                        visual,
                        static_cast<unsigned int>(depth),
                        ZPixmap,
                        0,
                        nullptr,
                        static_cast<unsigned int>(width_),
                        static_cast<unsigned int>(height_),
                        32,
                        row_bytes);
  if (image_ == nullptr) {
    disable("XCreateImage failed");
    return;
  }

  // A little-endian 32-bpp TrueColor visual maps one BGRA byte quadruplet to
  // the conventional 0xAARRGGBB X pixel. Reject other visuals instead of
  // silently presenting swapped or row-skewed images.
  if (image_->bits_per_pixel != 32 || image_->byte_order != LSBFirst ||
      image_->bytes_per_line != row_bytes || visual->red_mask != 0x00ff0000UL ||
      visual->green_mask != 0x0000ff00UL || visual->blue_mask != 0x000000ffUL) {
    disable("X11 visual is not compatible with packed little-endian BGRA");
    return;
  }

  // XDestroyImage releases image_->data with free(). Allocate with calloc,
  // never new[] or cudaHostAlloc, so ownership remains ABI-compatible.
  image_->data = static_cast<char*>(std::calloc(1U, frame_bytes_));
  if (image_->data == nullptr) {
    disable("allocating XImage storage failed");
    return;
  }

  XMapWindow(display_, window_);
  XFlush(display_);
  std::fprintf(
      stdout, "[x11_display] opened \"%s\" (%dx%d)\n", title_.c_str(), width_, height_);
}

void X11DisplayOp::stop() { cleanup(); }

holoscan::expected<void, holoscan::Error> X11DisplayOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = input.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }

  const holoscan::Tensor& tensor = sample->data;
  const std::array<std::int64_t, 3U> expected_shape{
      static_cast<std::int64_t>(height_),
      static_cast<std::int64_t>(width_),
      4,
  };
  if (tensor.data() == nullptr || tensor.device().device_type != kDLCUDA ||
      !same_dtype(tensor.dtype(), kUInt8Dtype) ||
      !shape_equals(tensor, expected_shape) ||
      !tensor.is_contiguous() ||
      tensor.nbytes() != static_cast<std::int64_t>(frame_bytes_)) {
    return failure(holoscan::ErrorCode::kInvalidArgument,
                   "X11DisplayOp expects contiguous CUDA uint8 [height,width,4] BGRA input");
  }

  if (disabled_) {
    return {};
  }
  if (display_ == nullptr || image_ == nullptr || image_->data == nullptr || window_ == 0U ||
      gc_ == nullptr) {
    return failure(holoscan::ErrorCode::kNotReady,
                   "X11DisplayOp is enabled but its X11 resources are incomplete");
  }

  auto copied = tensor.copy_to_host(image_->data, frame_bytes_, context.cuda_stream());
  if (!copied) {
    return holoscan::make_unexpected(std::move(copied).error());
  }

  static_cast<void>(XPutImage(display_,
                              window_,
                              gc_,
                              image_,
                              0,
                              0,
                              0,
                              0,
                              static_cast<unsigned int>(width_),
                              static_cast<unsigned int>(height_)));
  XFlush(display_);
  process_events();
  return {};
}

void X11DisplayOp::cleanup() noexcept {
  if (image_ != nullptr) {
    // This also calls free(image_->data).
    XDestroyImage(image_);
    image_ = nullptr;
  }
  if (display_ != nullptr && gc_ != nullptr) {
    XFreeGC(display_, gc_);
  }
  gc_ = nullptr;
  if (display_ != nullptr && window_ != 0U) {
    XDestroyWindow(display_, window_);
  }
  window_ = 0U;
  wm_delete_window_ = None;
  if (display_ != nullptr) {
    XCloseDisplay(display_);
    display_ = nullptr;
  }
}

void X11DisplayOp::disable(const char* reason) noexcept {
  std::fprintf(stderr, "[x11_display] disabled: %s\n", reason);
  cleanup();
  disabled_ = true;
}

void X11DisplayOp::process_events() {
  while (display_ != nullptr && XPending(display_) > 0) {
    XEvent event{};
    XNextEvent(display_, &event);
    if (event.type == KeyPress) {
      const KeySym key = XLookupKeysym(&event.xkey, 0);
      if (key == XK_q || key == XK_Q || key == XK_Escape) {
        std::raise(SIGINT);
      }
    } else if (event.type == ClientMessage && wm_delete_window_ != None &&
               static_cast<Atom>(event.xclient.data.l[0]) == wm_delete_window_) {
      std::raise(SIGINT);
    }
  }
}

}  // namespace holoscan::examples::v4l2_depth
