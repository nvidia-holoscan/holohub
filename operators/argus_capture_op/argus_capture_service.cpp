// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format off
#include "argus_capture_op/argus_capture_service.hpp"

#include <Argus/Argus.h>
#include <Argus/Ext/SensorTimestampTsc.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cuda.h>
#include <cudaEGL.h>
#include <nvbufsurface.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
// clang-format on

namespace holoscan::holoscan_camera::argus {
namespace {
void check(Argus::Status status, const char* operation) {
  if (status != Argus::STATUS_OK) {
    throw std::runtime_error(std::string(operation) + ": Argus status " + std::to_string(status));
  }
}
void check_cuda(CUresult status, const char* operation) {
  if (status != CUDA_SUCCESS) {
    const char* name = nullptr;
    cuGetErrorName(status, &name);
    throw std::runtime_error(std::string(operation) + ": " + (name ? name : "CUDA error"));
  }
}
template <class Interface, class Object>
Interface* interface(Object* object, const char* name) {
  auto* result = Argus::interface_cast<Interface>(object);
  if (!result) throw std::runtime_error(std::string("Argus interface unavailable: ") + name);
  return result;
}

// Preserve the calling thread's context: lifecycle, capture worker, and compute run on different
// threads. Never make the Argus primary context permanently current on an HSDK execution lane.
class ContextScope {
 public:
  explicit ContextScope(CUcontext context) {
    check_cuda(cuCtxPushCurrent(context), "push context");
  }
  ~ContextScope() {
    CUcontext previous{};
    cuCtxPopCurrent(&previous);
  }
};

EGLDisplay create_display() {
  auto query = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));
  auto platform = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
      eglGetProcAddress("eglGetPlatformDisplayEXT"));
  if (query && platform) {
    EGLDeviceEXT devices[8]{};
    EGLint count{};
    if (query(8, devices, &count)) {
      for (int i = 0; i < count; ++i) {
        auto display = platform(EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
        if (display != EGL_NO_DISPLAY && eglInitialize(display, nullptr, nullptr)) return display;
      }
    }
  }
  auto display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (display != EGL_NO_DISPLAY && eglInitialize(display, nullptr, nullptr)) return display;
  throw std::runtime_error("cannot initialize a headless or default NVIDIA EGL display");
}

struct Runtime {
  EGLDisplay display{EGL_NO_DISPLAY};
  CUcontext context{};
  bool retained{};
  Argus::UniqueObj<Argus::CameraProvider> provider;
  std::mutex configure_mutex;
  void open() {
    check_cuda(cuInit(0), "cuInit");
    check_cuda(cuDevicePrimaryCtxRetain(&context, 0), "retain primary CUDA context");
    retained = true;
    display = create_display();
    provider.reset(Argus::CameraProvider::create());
    interface<Argus::ICameraProvider>(provider.get(), "CameraProvider (check nvargus-daemon)");
  }
  ~Runtime() {
    provider.reset();
    if (display != EGL_NO_DISPLAY) eglTerminate(display);
    if (retained) cuDevicePrimaryCtxRelease(0);
  }
};

std::shared_ptr<Runtime> runtime() {
  // Argus requires one CameraProvider per process, even with separate capture sessions.
  // Also serialize last-owner destruction against creation of a replacement provider. Recursive
  // locking covers cleanup if allocating the shared owner or initializing Argus throws here.
  static std::recursive_mutex mutex;
  static std::weak_ptr<Runtime> current;
  std::lock_guard lock(mutex);
  auto result = current.lock();
  if (!result) {
    result = std::shared_ptr<Runtime>(new Runtime, [](Runtime* value) {
      std::lock_guard lock(mutex);
      delete value;
    });
    result->open();
    current = result;
  }
  return result;
}

struct Slot {
  NvBufSurface* surface{};
  bool mapped{};
  CUgraphicsResource resource{};
  CUeglFrame egl{};
  Argus::UniqueObj<Argus::Buffer> buffer;
  std::size_t index{};
  // Destroyed only after capture is stopped and all borrowed frames returned.
  ~Slot() {
    buffer.reset();
    if (resource) cuGraphicsUnregisterResource(resource);
    if (mapped) NvBufSurfaceUnMapEglImage(surface, 0);
    if (surface) NvBufSurfaceDestroy(surface);
  }
};

class NativeCaptureService final : public CaptureService {
 public:
  explicit NativeCaptureService(ArgusCaptureConfig config) : config_(std::move(config)) {}
  ~NativeCaptureService() override {
    if (!runtime_) return;
    try {
      ContextScope context(runtime_->context);
      try {
        stop();
      } catch (const std::exception& e) {
        std::fprintf(stderr, "Argus shutdown: %s\n", e.what());
        // Session destruction is the last cancellation boundary after a native stop failure.
        session_.reset();
        i_session_ = nullptr;
      }
      request_.reset();
      slots_.clear();
      stream_.reset();
      session_.reset();
    } catch (const std::exception& e) {
      std::fprintf(stderr, "Argus resource cleanup: %s\n", e.what());
    }
  }

  void open() {
    runtime_ = runtime();
    std::lock_guard lock(runtime_->configure_mutex);
    ContextScope context(runtime_->context);
    auto* provider = interface<Argus::ICameraProvider>(runtime_->provider.get(), "CameraProvider");
    std::vector<Argus::CameraDevice*> devices;
    check(provider->getCameraDevices(&devices), "enumerate cameras");
    if (config_.camera_index >= devices.size())
      throw std::invalid_argument("camera index out of range");
    auto* device = devices[config_.camera_index];
    auto* properties = interface<Argus::ICameraProperties>(device, "CameraProperties");
    std::vector<Argus::SensorMode*> modes;
    check(properties->getAllSensorModes(&modes), "enumerate sensor modes");
    if (config_.sensor_mode >= modes.size())
      throw std::invalid_argument("sensor mode out of range");
    auto* mode = modes[config_.sensor_mode];
    auto* sensor = interface<Argus::ISensorMode>(mode, "SensorMode");
    const auto size = sensor->getResolution();
    if (config_.width > size.width() || config_.height > size.height()) {
      throw std::invalid_argument("output resolution exceeds selected sensor mode");
    }
    const auto duration = static_cast<std::uint64_t>(std::llround(1e9 / config_.fps));
    const auto range = sensor->getFrameDurationRange();
    if (duration < range.min() || duration > range.max()) {
      throw std::invalid_argument("requested frame rate is outside the selected sensor mode range");
    }
    session_.reset(provider->createCaptureSession(device));
    i_session_ = interface<Argus::ICaptureSession>(session_.get(), "CaptureSession");
    Argus::UniqueObj<Argus::OutputStreamSettings> settings(
        i_session_->createOutputStreamSettings(Argus::STREAM_TYPE_BUFFER));
    auto* stream_settings =
        interface<Argus::IBufferOutputStreamSettings>(settings.get(), "BufferStreamSettings");
    check(stream_settings->setBufferType(Argus::BUFFER_TYPE_EGL_IMAGE),
          "set EGL image buffer type");
    check(stream_settings->setSyncType(Argus::SYNC_TYPE_NONE), "set synchronous buffer ownership");
    stream_settings->setMetadataEnable(true);
    stream_.reset(i_session_->createOutputStream(settings.get()));
    i_stream_ = interface<Argus::IBufferOutputStream>(stream_.get(), "BufferOutputStream");
    Argus::UniqueObj<Argus::BufferSettings> buffer_settings(i_stream_->createBufferSettings());
    auto* image_settings =
        interface<Argus::IEGLImageBufferSettings>(buffer_settings.get(), "EGLImageBufferSettings");
    check(image_settings->setEGLDisplay(runtime_->display), "set buffer EGL display");

    for (std::uint32_t index = 0; index < config_.buffer_count; ++index) {
      auto slot = std::make_unique<Slot>();
      slot->index = index;
      NvBufSurfaceAllocateParams allocation{};
      allocation.params.gpuId = config_.cuda_device;
      allocation.params.width = config_.width;
      allocation.params.height = config_.height;
      allocation.params.colorFormat = config_.pixel_format == ArgusPixelFormat::kNv12
                                          ? NVBUF_COLOR_FORMAT_NV12
                                          : NVBUF_COLOR_FORMAT_YUV420;
      allocation.params.layout = NVBUF_LAYOUT_BLOCK_LINEAR;
      allocation.params.memType = NVBUF_MEM_SURFACE_ARRAY;
      allocation.memtag = NvBufSurfaceTag_CAMERA;
      if (NvBufSurfaceAllocate(&slot->surface, 1, &allocation) != 0) {
        throw std::runtime_error(
            "NvBufSurfaceAllocate failed for requested output format/geometry");
      }
      slot->surface->numFilled = 1;
      if (NvBufSurfaceMapEglImage(slot->surface, 0) != 0) {
        throw std::runtime_error("NvBufSurfaceMapEglImage failed");
      }
      slot->mapped = true;
      auto image = slot->surface->surfaceList[0].mappedAddr.eglImage;
      check_cuda(cuGraphicsEGLRegisterImage(&slot->resource, image,
                                            CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY),
                 "register capture EGL image with CUDA");
      check_cuda(cuGraphicsResourceGetMappedEglFrame(&slot->egl, slot->resource, 0, 0),
                 "map capture EGL image");
      const auto expected_planes = config_.pixel_format == ArgusPixelFormat::kNv12 ? 2U : 3U;
      // CUDA's semiplanar enum names describe component order, not byte order:
      // YVU420_SEMIPLANAR has NV12's UV byte order; YUV420_SEMIPLANAR has VU byte order.
      // Planar I420 still uses YUV420_PLANAR because its separate planes are ordered Y, U, V.
      const auto expected_color = config_.pixel_format == ArgusPixelFormat::kNv12
                                      ? CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR
                                      : CU_EGL_COLOR_FORMAT_YUV420_PLANAR;
      const bool layout_matches = slot->egl.planeCount == expected_planes &&
                                  slot->egl.cuFormat == CU_AD_FORMAT_UNSIGNED_INT8 &&
                                  slot->egl.width == config_.width &&
                                  slot->egl.height == config_.height;
      if (!layout_matches || slot->egl.eglColorFormat != expected_color) {
        const auto& surface = slot->surface->surfaceList[0];
        std::ostringstream message;
        message << (layout_matches
                        ? "CUDA EGL image color order differs from requested NV12/I420"
                        : "CUDA EGL image does not match requested 8-bit YUV420 layout")
                << ": requested="
                << (config_.pixel_format == ArgusPixelFormat::kNv12 ? "NV12" : "I420")
                << ", NvBufSurface.colorFormat=" << static_cast<unsigned>(surface.colorFormat)
                << ", requestedNvBufFormat="
                << static_cast<unsigned>(allocation.params.colorFormat)
                << ", eglColorFormat=" << static_cast<unsigned>(slot->egl.eglColorFormat)
                << ", expectedEglColorFormat=" << static_cast<unsigned>(expected_color)
                << ", cuFormat=" << static_cast<unsigned>(slot->egl.cuFormat)
                << ", frameType=" << static_cast<unsigned>(slot->egl.frameType)
                << ", planes=" << slot->egl.planeCount << ", expectedPlanes=" << expected_planes
                << ", size=" << slot->egl.width << 'x' << slot->egl.height
                << ", requestedSize=" << config_.width << 'x' << config_.height;
        throw std::runtime_error(message.str());
      }
      check(image_settings->setEGLImage(image), "set capture EGL image");
      slot->buffer.reset(i_stream_->createBuffer(buffer_settings.get()));
      interface<Argus::IBuffer>(slot->buffer.get(), "Buffer")->setClientData(slot.get());
      slots_.push_back(std::move(slot));
      check(i_stream_->releaseBuffer(slots_.back()->buffer.get()), "queue initial capture buffer");
    }

    request_.reset(i_session_->createRequest(Argus::CAPTURE_INTENT_VIDEO_RECORD));
    auto* request = interface<Argus::IRequest>(request_.get(), "Request");
    check(request->enableOutputStream(stream_.get()), "enable camera output");
    auto* source = interface<Argus::ISourceSettings>(request_.get(), "SourceSettings");
    check(source->setSensorMode(mode), "select sensor mode");
    check(source->setFrameDurationRange(Argus::Range<std::uint64_t>(duration, duration)),
          "set frame rate");
  }

  void start() override {
    if (running_) return;
    check(i_session_->repeat(request_.get()), "start repeated capture");
    running_ = true;
    seen_ = false;
    sequence_epoch_ = 0;
  }

  void stop() override {
    if (!running_ || !i_session_) return;
    i_session_->stopRepeat();
    check(i_session_->cancelRequests(), "cancel pending captures");
    check(i_session_->waitForIdle(static_cast<std::uint64_t>(config_.timeout_ms) * 1000000),
          "wait for capture session to become idle");
    // Recycle completed frames left in the stream, so a restart cannot emit the previous epoch.
    for (;;) {
      Argus::Status status{};
      auto* buffer = i_stream_->acquireBuffer(0, &status);
      if (!buffer && (status == Argus::STATUS_TIMEOUT || status == Argus::STATUS_OK)) break;
      check(status, "drain stopped capture stream");
      if (!buffer) throw std::runtime_error("null buffer while draining capture");
      check(i_stream_->releaseBuffer(buffer), "release stopped capture buffer");
    }
    running_ = false;
  }

  std::optional<CapturedFrame> acquire(std::uint32_t timeout_ms) override {
    Argus::Status status{};
    auto* buffer =
        i_stream_->acquireBuffer(static_cast<std::uint64_t>(timeout_ms) * 1000000, &status);
    if (!buffer && status == Argus::STATUS_TIMEOUT) return std::nullopt;
    check(status, "acquire camera frame");
    if (!buffer) throw std::runtime_error("Argus returned a null capture buffer");
    try {
      const auto* view = interface<Argus::IBuffer>(buffer, "Buffer");
      const auto* slot = static_cast<const Slot*>(view->getClientData());
      if (!slot || slot->index >= slots_.size() || slots_[slot->index].get() != slot) {
        throw std::runtime_error("Argus returned an unknown capture buffer");
      }
      auto* capture =
          interface<const Argus::ICaptureMetadata>(view->getMetadata(), "CaptureMetadata");
      CapturedFrame frame;
      frame.slot = slot->index;
      auto& data = frame.metadata;
      const auto id = capture->getCaptureId();
      // Extend ordinary 32-bit wrap, preserving genuine regressions for SequenceTracker.
      if (seen_ && previous_id_ > 0xC0000000U && id < 0x40000000U) sequence_epoch_ += 1ULL << 32;
      previous_id_ = id;
      seen_ = true;
      data.sequence = sequence_epoch_ + id;
      data.camera_index = config_.camera_index;
      data.sensor_mode = config_.sensor_mode;
      data.source_index = capture->getSourceIndex();
      data.sensor_timestamp_ns = capture->getSensorTimestamp();
      if (auto* tsc =
              Argus::interface_cast<const Argus::Ext::ISensorTimestampTsc>(view->getMetadata())) {
        data.sensor_sof_timestamp_ns = tsc->getSensorSofTimestampTsc();
        if (data.sensor_sof_timestamp_ns) data.timestamp_clock = ArgusTimestampClock_TEGRA_TSC_NS;
      }
      if (!data.sensor_timestamp_ns && !data.sensor_sof_timestamp_ns) {
        throw std::runtime_error("capture metadata has no sensor timestamp");
      }
      if (config_.integration_start_offset_ns &&
          data.timestamp_clock != ArgusTimestampClock_TEGRA_TSC_NS) {
        throw std::runtime_error(
            "calibrated capture time requires the Argus SensorTimestampTsc extension");
      }
      data.exposure_time_ns = capture->getSensorExposureTime();
      data.frame_duration_ns = capture->getFrameDuration();
      data.analog_gain = capture->getSensorAnalogGain();
      data.isp_digital_gain = capture->getIspDigitalGain();
      data.ae_locked = capture->getAeLocked();
      data.ae_state = capture->getAeState().getName();
      data.awb_state = capture->getAwbState().getName();
      data.width = config_.width;
      data.height = config_.height;
      return frame;
    } catch (...) {
      i_stream_->releaseBuffer(buffer);
      throw;
    }
  }

  void release(std::size_t slot) override {
    check(i_stream_->releaseBuffer(slots_.at(slot)->buffer.get()), "release camera frame");
  }

  void copy_frame(std::size_t slot_index, const holoscan::TensorOutputWriteGuard& writer,
                  holoscan::MemoryKind placement) override {
    ContextScope context(runtime_->context);
    auto& slot = *slots_.at(slot_index);
    const bool host =
        placement == holoscan::MemoryKind::kHost || placement == holoscan::MemoryKind::kPinnedHost;
    void* destination = writer.data();
    if (!destination || writer.byte_size() < validate_config(config_)) {
      throw std::runtime_error("HSDK tensor write span is smaller than the capture image");
    }
    const auto stream = writer.stream();
    if (!host && !stream) throw std::runtime_error("HSDK GPU tensor writer has no producer stream");
    // Honor any allocation reuse waits already ordered by the HSDK write guard. Using an unrelated
    // CUDA stream here could overwrite a pooled tensor before its previous readers finish.
    const CUstream copy_stream = stream ? stream->get() : nullptr;
    struct CopyBarrier {
      CUstream stream;
      bool pending{};
      ~CopyBarrier() {
        if (pending) cuStreamSynchronize(stream);
      }
    } barrier{copy_stream};
    if (!host) {
      unsigned int memory_type{};
      int device{};
      check_cuda(cuPointerGetAttribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                       reinterpret_cast<CUdeviceptr>(destination)),
                 "query tensor memory");
      check_cuda(cuPointerGetAttribute(&device, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                       reinterpret_cast<CUdeviceptr>(destination)),
                 "query tensor device");
      if (memory_type != CU_MEMORYTYPE_DEVICE || device != config_.cuda_device) {
        throw std::runtime_error("HSDK tensor placement differs from the capture GPU");
      }
    }
    std::size_t offset = 0;
    for (unsigned int plane = 0; plane < slot.egl.planeCount; ++plane) {
      const std::size_t width = plane == 0 || config_.pixel_format == ArgusPixelFormat::kNv12
                                    ? config_.width
                                    : config_.width / 2;
      const std::size_t height = plane == 0 ? config_.height : config_.height / 2;
      CUDA_MEMCPY2D copy{};
      if (slot.egl.frameType == CU_EGL_FRAME_TYPE_ARRAY) {
        copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.srcArray = slot.egl.frame.pArray[plane];
      } else if (slot.egl.frameType == CU_EGL_FRAME_TYPE_PITCH) {
        copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.srcDevice = reinterpret_cast<CUdeviceptr>(slot.egl.frame.pPitch[plane]);
        copy.srcPitch = slot.surface->surfaceList[0].planeParams.pitch[plane];
      } else {
        throw std::runtime_error("unsupported CUDA EGL frame storage");
      }
      copy.dstMemoryType = host ? CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE;
      copy.dstHost = host ? static_cast<std::uint8_t*>(destination) + offset : nullptr;
      copy.dstDevice = host ? 0 : reinterpret_cast<CUdeviceptr>(destination) + offset;
      copy.dstPitch = width;
      copy.WidthInBytes = width;
      copy.Height = height;
      if (host) {
        // Device-to-host copies complete before returning, including pageable destinations.
        check_cuda(cuMemcpy2D(&copy), "copy capture plane to host HSDK tensor");
      } else {
        barrier.pending = true;
        check_cuda(cuMemcpy2DAsync(&copy, copy_stream), "copy capture plane to GPU HSDK tensor");
      }
      offset += width * height;
    }
    if (barrier.pending) {
      // A device-to-device copy API can return before the transfer completes. Explicitly wait on
      // the producer stream before SYNC_TYPE_NONE hands the image back to the ISP for overwrite.
      check_cuda(cuStreamSynchronize(copy_stream), "complete capture copy before buffer release");
      barrier.pending = false;
    }
  }

  std::optional<std::int64_t> timestamp_now_ns() noexcept override {
#if defined(__aarch64__)
    std::uint64_t ticks{}, frequency{};
    asm volatile("mrs %0, cntvct_el0" : "=r"(ticks));
    asm volatile("mrs %0, cntfrq_el0" : "=r"(frequency));
    if (frequency == 0) return std::nullopt;
    const auto ns = static_cast<unsigned __int128>(ticks) * 1000000000 / frequency;
    if (ns > static_cast<unsigned __int128>(std::numeric_limits<std::int64_t>::max()))
      return std::nullopt;
    return static_cast<std::int64_t>(ns);
#else
    return std::nullopt;
#endif
  }

 private:
  ArgusCaptureConfig config_;
  std::shared_ptr<Runtime> runtime_;
  Argus::UniqueObj<Argus::CaptureSession> session_;
  Argus::UniqueObj<Argus::OutputStream> stream_;
  Argus::UniqueObj<Argus::Request> request_;
  Argus::ICaptureSession* i_session_{};
  Argus::IBufferOutputStream* i_stream_{};
  std::vector<std::unique_ptr<Slot>> slots_;
  bool running_{};
  bool seen_{};
  std::uint32_t previous_id_{};
  std::uint64_t sequence_epoch_{};
};
}  // namespace

std::unique_ptr<CaptureService> make_service(const ArgusCaptureConfig& config) {
  auto service = std::make_unique<NativeCaptureService>(config);
  service->open();
  return service;
}
}  // namespace holoscan::holoscan_camera::argus
