// SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sipl_capture_op/sipl_capture_op.hpp"

#include <dlpack/dlpack.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>

#include "sipl_capture_op/hsb_frame_metadata.hpp"
#include "sipl_capture_op/sipl_capture_service.hpp"
#include "sipl_capture_op/tsc_to_ns.hpp"

namespace holoscan::holoscan_camera {

namespace {

// ---------------------------------------------------------------------------
// ARM Generic Timer helpers
// ---------------------------------------------------------------------------

// Read the ARM Generic Timer counter frequency register.  On Tegra/IGX Orin
// this register always reads 31,250,000 Hz and is accessible from EL0.
std::uint64_t read_arm_cntfrq() noexcept {
#if defined(__aarch64__)
  std::uint64_t cntfrq{};
  asm volatile("mrs %0, cntfrq_el0" : "=r"(cntfrq));
  return cntfrq > 0U ? cntfrq : 31'250'000ULL;
#else
  return 31'250'000ULL;
#endif
}

// File-scope constant — computed once at load time.
const std::uint64_t kArmCntfrq = read_arm_cntfrq();

// Read current ARM virtual timer counter as nanoseconds (same domain as CLOCK_MONOTONIC).
std::optional<std::int64_t> arm_monotonic_now() noexcept {
#if defined(__aarch64__)
  std::uint64_t tsc{};
  asm volatile("mrs %0, cntvct_el0" : "=r"(tsc));
  if (tsc == 0U) {
    return std::nullopt;
  }
  return tsc_to_ns(tsc, kArmCntfrq);
#else
  return std::nullopt;
#endif
}

// ---------------------------------------------------------------------------
// Frame guard
// ---------------------------------------------------------------------------

// RAII guard: releases the SIPL frame on destruction unless handoff_pending_output() was called.
class AcquiredFrameGuard {
 public:
  AcquiredFrameGuard(SIPLCaptureService& service, SIPLCaptureService::AcquiredFrame& frame)
      : service_(service), frame_(frame) {}

  ~AcquiredFrameGuard() {
    if (active_) {
      service_.release_acquired_frame(frame_);
    }
  }

  // Transfer SIPL buffer ownership to the pending-release map so it stays alive until the
  // downstream consumer frees the Tensor.  After this call the SIPL buffer must NOT be
  // released by this guard.
  void handoff_pending_output() {
    if (!active_) {
      return;
    }
    service_.release_raw_buffer_if_unused(frame_.buffer, frame_.buffer_raw);
    service_.register_pending_output(frame_.cuda_ptr, frame_.buffer);
    active_ = false;
  }

  // Release the SIPL buffer early (before compute() returns) so the capture pipeline can reuse
  // it immediately once the copy into the pool slot is complete.
  void release() {
    if (active_) {
      service_.release_acquired_frame(frame_);
      active_ = false;
    }
  }

 private:
  SIPLCaptureService& service_;
  SIPLCaptureService::AcquiredFrame& frame_;
  bool active_{ true };
};

// ---------------------------------------------------------------------------
// Sensor metadata conversion
// ---------------------------------------------------------------------------

// Converts the SIPL ImageMetaData struct (+ optional HSB board metadata) to
// SIPLFrameMetadataT (FlatBuffers native table).
// Called from compute() after reading both sources from the frame buffer.
SIPLFrameMetadataT make_sensor_metadata(const nvsipl::INvSIPLClient::ImageMetaData& m,
    const std::optional<HsbFrameMetadata>& hsb,
    const nvsipl::IspStatsInfo* isp_stats,
    std::int32_t width = 0,
    std::int32_t height = 0,
    std::uint32_t nv12_uv_stride = 0,
    std::uint64_t nv12_uv_offset = 0) {
  SIPLFrameMetadataT out{};
  out.frame_capture_start_tsc = m.frameCaptureStartTSC;

  // SIPL defines DEVBLK_CDI_MAX_EXPOSURES = 8; clamp defensively.
  const auto n = static_cast<std::uint8_t>(
      std::min(static_cast<std::uint32_t>(m.numExposures), 8U));
  out.num_exposures = n;

  // Exposure times and gains.
  out.exp_time_valid = m.sensorExpInfo.expTimeValid != 0;
  out.gain_valid     = m.sensorExpInfo.gainValid != 0;
  out.exposure_time.assign(m.sensorExpInfo.exposureTime,
                           m.sensorExpInfo.exposureTime + n);
  out.sensor_gain.assign(m.sensorExpInfo.sensorGain,
                         m.sensorExpInfo.sensorGain + n);

  // White-balance gains: one SIPLWbGain per exposure (R, Gr, Gb, B).
  out.wb_valid = m.sensorWBInfo.wbValid != 0;
  out.wb_gain.reserve(n);
  for (std::uint8_t i = 0; i < n; ++i) {
    const auto& g = m.sensorWBInfo.wbGain[i].value;
    out.wb_gain.push_back(SIPLWbGain(g[0], g[1], g[2], g[3]));
  }

  // Temperature.
  out.temp_valid = m.sensorTempInfo.tempValid != 0;
  const auto nt = std::min(static_cast<std::uint32_t>(m.sensorTempInfo.numTemperatures), 4U);
  out.sensor_temp_celsius.assign(m.sensorTempInfo.sensorTempCelsius,
                                 m.sensorTempInfo.sensorTempCelsius + nt);

  // Error flag and sensor-embedded timestamp.
  out.error_flag            = m.errorFlag;
  out.frame_timestamp_valid = m.frameTimestampInfo.frameTimestampValid != 0;
  out.frame_timestamp       = m.frameTimestampInfo.frameTimestamp;

  // ISP statistics blob (NV12 mode only; null pointer in RAW10 mode).
  if (isp_stats) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(isp_stats);
    out.isp_stats_blob.assign(bytes, bytes + sizeof(nvsipl::IspStatsInfo));
  }

  // NV12 UV plane geometry (zero in RAW10 mode; non-zero values from NvSci attributes).
  out.nv12_uv_stride = nv12_uv_stride;
  out.nv12_uv_offset = nv12_uv_offset;

  // Frame pixel dimensions (from SIPL buffer plane attributes).
  out.width  = width;
  out.height = height;

  // HSB board metadata (optional — only present on Hololink camera modules).
  if (hsb) {
    out.hsb_valid         = true;
    out.hsb_flags         = hsb->flags;
    out.hsb_psn           = hsb->psn;
    out.hsb_crc           = hsb->crc;
    out.hsb_frame_number  = hsb->frame_number;
    out.hsb_timestamp_s   = hsb->timestamp_s;
    out.hsb_timestamp_ns  = hsb->timestamp_ns;
    out.hsb_bytes_written = hsb->bytes_written;
    out.hsb_metadata_s    = hsb->metadata_s;
    out.hsb_metadata_ns   = hsb->metadata_ns;
  }

  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Setup / contract
// ---------------------------------------------------------------------------

SIPLCaptureOp::SIPLCaptureOp(
    std::shared_ptr<SIPLCaptureService> service,
    std::uint32_t camera_index,
    holoscan::MemoryKind memory_kind)
    : service_(std::move(service)), camera_index_(camera_index), memory_kind_(memory_kind) {
  if (!service_) {
    throw std::invalid_argument("SIPLCaptureOp: service must not be null");
  }
  if (memory_kind_ != holoscan::MemoryKind::kHost &&
      memory_kind_ != holoscan::MemoryKind::kPinnedHost &&
      memory_kind_ != holoscan::MemoryKind::kCudaDevice) {
    // kUnknown is a diagnostic value the schema never admits for a published tensor. kCudaManaged
    // is host-writable and would appear to work, but a captured frame that migrates on first
    // device touch trades a copy this operator can see for a fault it cannot, so it is refused
    // until some consumer asks for it and can say why.
    throw std::invalid_argument(
        "SIPLCaptureOp: memory_kind must be kHost, kPinnedHost, or kCudaDevice");
  }
}

void SIPLCaptureOp::setup(holoscan::OperatorSpec& spec) {
  static constexpr DLDataType kUInt8{ static_cast<uint8_t>(kDLUInt), 8, 1 };
  static constexpr std::size_t kMaxFrameBytes = 64ULL * 1024ULL * 1024ULL;
  spec.output(frame, "frame")
      .max_emits_per_compute(1U)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = {
              .memory_kind = memory_kind_,
              .dtype = kUInt8,
          },
          .bounds = holoscan::TensorBounds{
              .max_rank = 1U,
              .max_byte_span = kMaxFrameBytes,
          },
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
  spec.output(sensor_data, "sensor_data").max_emits_per_compute(1U);
  spec.notification_source(frame_ready_, "frame-ready").capacity(4U).sender_reference_capacity(1U);

  spec.lifecycle()
      .stage(holoscan::LifecycleStage::kConfigure, &SIPLCaptureOp::on_configure)
      .stage(holoscan::LifecycleStage::kDiscover,  &SIPLCaptureOp::on_discover)
      .stage(holoscan::LifecycleStage::kAllocate,  &SIPLCaptureOp::on_allocate)
      .stage(holoscan::LifecycleStage::kArm,       &SIPLCaptureOp::on_arm)
      .stage(holoscan::LifecycleStage::kStart,     &SIPLCaptureOp::on_start)
      .stage(holoscan::LifecycleStage::kStop,      &SIPLCaptureOp::on_stop)
      .stage(holoscan::LifecycleStage::kRelease,   &SIPLCaptureOp::on_release);
}

holoscan::Contract SIPLCaptureOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnNotified{ .event = frame_ready_ });
  return result;
}

// ---------------------------------------------------------------------------
// Lifecycle hooks
// ---------------------------------------------------------------------------

LifecycleStatus SIPLCaptureOp::on_configure(LifecycleContext& /*ctx*/) noexcept {
  return guards_.run<LifecycleStage::kConfigure>([this]() noexcept -> LifecycleStatus {
    timeout_us_ = service_->default_timeout_us();
    frame_count_ = 0;
    // Reset the clock discipline so it re-primes in the next lifecycle generation.
    clock_discipline_.bind(holoscan::ClockId{});
    return LifecycleStatus::kOk;
  });
}

LifecycleStatus SIPLCaptureOp::on_discover(LifecycleContext& /*ctx*/) noexcept {
  return guards_.run<LifecycleStage::kDiscover>([this]() noexcept -> LifecycleStatus {
    std::uint32_t count = 0;
    try {
      count = service_->camera_count();
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("SIPLCaptureOp: camera_count failed: {}", e.what());
      return LifecycleStatus::kFatalFailure;
    }
    if (camera_index_ >= count) {
      HOLOSCAN_LOG_ERROR("SIPLCaptureOp: camera_index {} out of range ({} cameras)",
          camera_index_, count);
      return LifecycleStatus::kFatalFailure;
    }
    try {
      const auto& info = service_->get_camera_info()[camera_index_];
      embedded_top_lines_ = info.embedded_top_lines;
      embedded_bottom_lines_ = info.embedded_bottom_lines;
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("SIPLCaptureOp: get_camera_info failed: {}", e.what());
      return LifecycleStatus::kFatalFailure;
    }
    return LifecycleStatus::kOk;
  });
}

LifecycleStatus SIPLCaptureOp::on_allocate(LifecycleContext& /*ctx*/) noexcept {
  // NvSci buffer allocation happens lazily inside start_buffers() (called from
  // add_operator_ref()), so there is nothing to do here for individual operators.
  // Returning kSkipped lets the guard record that we are allocation-ready without
  // claiming ownership of anything to release.
  return guards_.run<LifecycleStage::kAllocate>(
      []() noexcept -> LifecycleStatus { return LifecycleStatus::kSkipped; });
}

LifecycleStatus SIPLCaptureOp::on_arm(LifecycleContext& ctx) noexcept {
  return guards_.run<LifecycleStage::kArm>([this, &ctx]() noexcept -> LifecycleStatus {
    // kArm owns only the notification sender. Everything on_stop tears down (callback,
    // add_operator_ref) belongs in on_start: Core skips kArm on a kStart-requested restart and
    // replays only kStop → kStart, so anything acquired here that kStop releases would not be
    // re-acquired on restart.
    auto sender = ctx.notification_sender(frame_ready_);
    if (!sender) {
      HOLOSCAN_LOG_ERROR("SIPLCaptureOp: failed to get notification sender: {}",
          sender.error().message);
      return LifecycleStatus::kFatalFailure;
    }
    sender_ = std::move(*sender);
    return LifecycleStatus::kOk;
  });
}

LifecycleStatus SIPLCaptureOp::on_start(LifecycleContext& /*ctx*/) noexcept {
  return guards_.run<LifecycleStage::kStart>([this]() noexcept -> LifecycleStatus {
    // sequence_tracker_ belongs here: SIPL resets the pipeline frame sequence on every start,
    // and kStart is the minimal suffix Core replays on a kStop+kStart restart (kArm is skipped).
    // Resetting here guarantees the tracker is fresh after every restart without relying on kArm.
    sequence_tracker_.reset();
    try {
      service_->set_frame_ready_callback(camera_index_, [this]() {
        if (auto result = sender_.post(); !result) {
          const auto code = result.error().code;
          if (code != holoscan::ErrorCode::kBackpressured
              && code != holoscan::ErrorCode::kNotReady
              && code != holoscan::ErrorCode::kClosed
              && code != holoscan::ErrorCode::kStaleEndpoint) {
            HOLOSCAN_LOG_WARN("SIPLCaptureOp: notification post failed: {}", result.error().message);
          }
        }
      });
      service_->add_operator_ref();
    } catch (const std::exception& e) {
      service_->set_frame_ready_callback(camera_index_, {});
      HOLOSCAN_LOG_ERROR("SIPLCaptureOp: start failed: {}", e.what());
      return LifecycleStatus::kFatalFailure;
    }
    return LifecycleStatus::kOk;
  });
}

LifecycleStatus SIPLCaptureOp::on_stop(LifecycleContext& /*ctx*/) noexcept {
  return guards_.run<LifecycleStage::kStop>([this]() noexcept -> LifecycleStatus {
    service_->set_frame_ready_callback(camera_index_, {});
    try {
      service_->remove_operator_ref();
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("SIPLCaptureOp: remove_operator_ref failed: {}", e.what());
      return LifecycleStatus::kFatalFailure;
    }
    return LifecycleStatus::kOk;
  });
}

LifecycleStatus SIPLCaptureOp::on_release(LifecycleContext& /*ctx*/) noexcept {
  // Not routed through guards_.run<kRelease>: that gates on kAllocate having reported kOk, but
  // NvSci allocation lives in the shared, ref-counted SIPLCaptureService, so on_allocate always
  // reports kSkipped and guards_.allocated() is never true -- gating on it here withheld this
  // body on every release, not just the early-bring-up-failure case the guard exists for.
  // sender_ and clock_discipline_ are the kArm- and compute()-scoped state actually owned here,
  // and both lines are idempotent, so running them unconditionally is correct whether or not
  // kArm/kStart ever succeeded.
  sender_ = {};
  // The measured offset between frameCaptureStartTSC and the reference clock is valid for
  // exactly one SIPL pipeline session. Clear it so the next generation re-primes rather than
  // projecting through a stale offset from a previous session.
  clock_discipline_.bind(holoscan::ClockId{});
  return LifecycleStatus::kOk;
}

// ---------------------------------------------------------------------------
// Clock settlement
// ---------------------------------------------------------------------------

void SIPLCaptureOp::ensure_clock_settled(holoscan::ExecutionContext& context) noexcept {
  if (clock_discipline_.settled()) {
    return;
  }
  const holoscan::ClockRef reference = context.clock();
  if (!reference.id().valid()) {
    return;
  }
  // SIPL frameCaptureStartTSC ticks and cntvct_el0 share the same ARM Generic Timer hardware on
  // Tegra/IGX. After scaling to nanoseconds (via kArmCntfrq), they are in the same epoch as
  // CLOCK_MONOTONIC. ClockDiscipline::prime() measures any residual offset.
  clock_discipline_.bind(reference);
  (void)clock_discipline_.prime(
      []() noexcept -> std::optional<std::int64_t> { return arm_monotonic_now(); },
      [&reference]() noexcept -> std::optional<std::int64_t> {
        const auto now = reference.now();
        if (!now) {
          return std::nullopt;
        }
        return now->timestamp_ns;
      });
}

// ---------------------------------------------------------------------------
// compute()
// ---------------------------------------------------------------------------

holoscan::expected<void, holoscan::Error> SIPLCaptureOp::compute(
    holoscan::ExecutionContext& context) {
  // Lazy clock settlement: the first compute() is the earliest point the reference clock is
  // reachable (LifecycleContext::resolve_clock is not wired in this runtime version).
  ensure_clock_settled(context);

  // Acquire one frame from the service (blocks until a frame is available or times out).
  SIPLCaptureService::AcquiredFrame acquired;
  try {
    acquired = service_->acquire_frame(camera_index_, timeout_us_);
  } catch (const std::exception& e) {
    return holoscan::make_unexpected(
        holoscan::Error{ holoscan::ErrorCode::kFailure, e.what() });
  }
  if (acquired.status != SIPLCaptureService::AcquireStatus::kOk) {
    return holoscan::make_unexpected(
        holoscan::Error{ holoscan::ErrorCode::kFailure, "SIPL acquire_frame reported failure" });
  }

  AcquiredFrameGuard frame_guard(*service_, acquired);

  auto* nvm_buf = dynamic_cast<nvsipl::INvSIPLClient::INvSIPLNvMBuffer*>(acquired.buffer);
  const auto& img_meta = nvm_buf->GetImageData();

  // Prefer the sensor-reported frame sequence number for gap detection; fall back to a local
  // counter when the driver does not populate the embedded sequence field.
  const std::uint64_t local_seq = ++frame_count_;
  const std::uint64_t seq =
      img_meta.frameSeqNumInfo.frameSeqNumValid
          ? img_meta.frameSeqNumInfo.frameSequenceNumber
          : local_seq;
  const auto obs = sequence_tracker_.observe(seq);

  // Build EmitOptions: capture time projected from the SIPL start-of-frame TSC, frame identity,
  // and degraded flag for any dropped frames SIPL or the operator observed.
  //
  // frameCaptureStartTSC, not frameCaptureTSC (end-of-frame): image.fbs requires
  // header.capture_timestamp_ns to be "normatively the start of integration", since a consumer
  // fusing against an IMU reconstructs the image-centred instant as capture_timestamp_ns plus
  // half of exposure_time_ns -- an arithmetic that only holds if this is the start.
  holoscan::EmitOptions options{};
  options.frame_id = seq;
  if (img_meta.frameCaptureStartTSC != 0U) {
    options.capture_time =
        clock_discipline_.project(tsc_to_ns(img_meta.frameCaptureStartTSC, kArmCntfrq));
  }
  if (obs.gap > 0U || img_meta.wasFrameDropped) {
    options.flags |= holoscan::SampleFlags::kDegraded;
  }
  if (img_meta.errorFlag) {
    options.flags |= holoscan::SampleFlags::kInvalid;
  }

  // Detect NV12 (ISP output) vs RAW Bayer (direct CSI output).  Any other format is an error.
  const bool is_nv12 = (acquired.plane_count >= 2U) &&
                       (acquired.plane_color_format[0] == NvSciColor_Y8) &&
                       (acquired.plane_color_format[1] == NvSciColor_V8U8);
  const bool is_raw10 =
      (acquired.plane_color_format[0] == NvSciColor_X2Rc10Rb10Ra10_Bayer10RGGB) ||
      (acquired.plane_color_format[0] == NvSciColor_X2Rc10Rb10Ra10_Bayer10BGGR) ||
      (acquired.plane_color_format[0] == NvSciColor_X2Rc10Rb10Ra10_Bayer10GRBG) ||
      (acquired.plane_color_format[0] == NvSciColor_X2Rc10Rb10Ra10_Bayer10GBRG);

  if (!is_nv12 && !is_raw10) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        "SIPLCaptureOp: unsupported buffer color format"});
  }

  // RAW10's acquired.plane_height[0] is the full CSI capture height, including SIPL's embedded
  // top/bottom lines (register dumps and blanking rows, not Bayer samples -- see
  // embedded_top_lines_/embedded_bottom_lines_, read once in on_discover()). Publishing them as
  // pixel data would break roi_offset_y's own contract below: a consumer is told row 0 of the
  // published tensor is a real sample at full-sensor row roi_offset_y, so row 0 has to actually
  // be one. NV12 is the ISP's own processed output and carries no such rows.
  const std::int64_t raw10_active_height = is_raw10
      ? static_cast<std::int64_t>(acquired.plane_height[0]) -
            static_cast<std::int64_t>(embedded_top_lines_) -
            static_cast<std::int64_t>(embedded_bottom_lines_)
      : 0;
  if (is_raw10 && raw10_active_height <= 0) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        "SIPLCaptureOp: embedded_top_lines_ + embedded_bottom_lines_ exceeds captured height"});
  }

  // For NV12 the consumer needs both Y and UV planes; wrap the full buffer.
  // For RAW10 wrap only the active rows (pitch × active height); the embedded rows are skipped
  // at the copy source below rather than published.
  const std::int64_t tensor_bytes = is_nv12
      ? static_cast<std::int64_t>(acquired.buffer_size)
      : static_cast<std::int64_t>(acquired.plane_pitch[0]) * raw10_active_height;

  // Read all metadata that requires the SIPL buffer's CPU mapping BEFORE the tensor copy,
  // so the buffer can be released as soon as the copy is done.

  // RAW10: get the buffer's CPU pointer once; reuse it for both the frame copy and HSB metadata.
  // This avoids the CUDA DeviceToHost path entirely for RAW output and keeps the buffer-hold
  // window as short as possible (pure CPU memcpy, no CUDA stream synchronisation overhead).
  const void* raw_cpu_ptr = nullptr;
  if (is_raw10 && acquired.buf_obj != nullptr) {
    NvSciBufObjGetConstCpuPtr(acquired.buf_obj, &raw_cpu_ptr);
  }

  // HSB board metadata lives at the end of the raw image buffer.
  std::optional<HsbFrameMetadata> hsb_meta;
  if (is_raw10 && raw_cpu_ptr != nullptr) {
    const std::size_t pixel_bytes = static_cast<std::size_t>(acquired.plane_pitch[0])
                                    * acquired.plane_height[0];
    hsb_meta = parse_hsb_metadata(
        static_cast<const std::uint8_t*>(raw_cpu_ptr), acquired.buffer_size, pixel_bytes);
  } else if (is_nv12 && acquired.buffer_raw != nullptr) {
    // NV12: buf_obj is the ISP output; get the raw (ICP) buffer's NvSciBufObj via the
    // NvMBuffer interface, query its pitch/height, then map the CPU pointer.
    auto* nvm_raw =
        dynamic_cast<nvsipl::INvSIPLClient::INvSIPLNvMBuffer*>(acquired.buffer_raw);
    if (nvm_raw) {
      NvSciBufObj buf_obj_raw = nvm_raw->GetNvSciBufImage();
      if (buf_obj_raw) {
        NvSciBufAttrList attr_list_raw = nullptr;
        if (NvSciBufObjGetAttrList(buf_obj_raw, &attr_list_raw) == NvSciError_Success) {
          NvSciBufAttrKeyValuePair attrs[] = {
              { NvSciBufImageAttrKey_PlanePitch,  nullptr, 0 },
              { NvSciBufImageAttrKey_PlaneHeight, nullptr, 0 },
              { NvSciBufImageAttrKey_Size,        nullptr, 0 },
          };
          if (NvSciBufAttrListGetAttrs(attr_list_raw, attrs, 3) == NvSciError_Success) {
            const auto pitch      = *static_cast<const uint32_t*>(attrs[0].value);
            const auto height     = *static_cast<const uint32_t*>(attrs[1].value);
            const auto buf_size   = *static_cast<const std::uint64_t*>(attrs[2].value);
            const void* raw_cpu = nullptr;
            if (NvSciBufObjGetConstCpuPtr(buf_obj_raw, &raw_cpu) == NvSciError_Success
                && raw_cpu != nullptr) {
              const std::size_t pixel_bytes = static_cast<std::size_t>(pitch) * height;
              hsb_meta = parse_hsb_metadata(
                  static_cast<const std::uint8_t*>(raw_cpu),
                  buf_size,
                  pixel_bytes);
            }
          }
        }
      }
    }
  }

  // ISP statistics must be read before the buffer is released.
  const nvsipl::IspStatsInfo* isp_stats_ptr = nullptr;
  nvsipl::IspStatsInfo isp_stats_copy{};
  if (is_nv12) {
    auto* isp_if = service_->isp_stats(camera_index_);
    if (isp_if && nvm_buf) {
      const auto* info = isp_if->GetIspStatsInfo(nvm_buf);
      if (info) {
        isp_stats_copy = *info;
        isp_stats_ptr = &isp_stats_copy;
      }
    }
  }

  // Capture descriptor fields from acquired.* before the buffer is released.
  // acquired.plane_pitch/width/height/offset/color_format are pointers into SIPL buffer
  // attributes and expire when the buffer is returned to the capture pipeline.
  const auto desc_width        = static_cast<std::int32_t>(acquired.plane_width[0]);
  const auto desc_height       = is_raw10
      ? static_cast<std::int32_t>(raw10_active_height)
      : static_cast<std::int32_t>(acquired.plane_height[0]);
  const auto desc_y_pitch      = acquired.plane_pitch[0];
  const auto nv12_uv_pitch     = is_nv12 ? acquired.plane_pitch[1]  : 0U;
  const auto nv12_uv_offset    = is_nv12 ? acquired.plane_offset[1] : 0ULL;

  // NV12 (ISP output) is exactly what ImageEncoding_NV12 describes: one byte per luma/chroma
  // sample, addressed by plane_layouts below. RAW10 is NvSci's X2Rc10Rb10Ra10 packing -- three
  // 10-bit samples packed LSB-first into each 4-byte dword -- which no BAYER_* encoding describes:
  // the schema defines a Bayer sample's width as coming from the Tensor element type, i.e. one
  // Tensor element is one sample, and a packed dword is not byte-addressable per pixel. Declaring
  // BAYER_RGGB here would assert a container the bytes do not have and mislead any consumer that
  // demosaics from the descriptor alone. CUSTOM is the honest choice: it states no organization,
  // and the Bayer phase itself is already available out-of-band from
  // SIPLCaptureService::CameraInfo::bayer_format, which every consumer of this rig already reads
  // to configure capture.
  const holoscan::schema::ImageEncoding desc_encoding =
      is_nv12 ? holoscan::schema::ImageEncoding_NV12 : holoscan::schema::ImageEncoding_CUSTOM;

  // Single-exposure sensor values; zero when the sensor does not report them.
  const std::int64_t desc_exposure_ns =
      (img_meta.sensorExpInfo.expTimeValid != 0 && img_meta.numExposures >= 1U)
          ? static_cast<std::int64_t>(img_meta.sensorExpInfo.exposureTime[0] * 1e9)
          : 0;
  const float desc_gain =
      (img_meta.sensorExpInfo.gainValid != 0 && img_meta.numExposures >= 1U)
          ? img_meta.sensorExpInfo.sensorGain[0]
          : 0.0f;

  // Copy the SIPL buffer into a pool-allocated tensor at the placement the compiled plan froze.
  static constexpr DLDataType kUInt8{ static_cast<uint8_t>(kDLUInt), 8, 1 };
  const std::array<int64_t, 1> loan_shape{ tensor_bytes };

  auto loan = frame.allocate_tensor(
      holoscan::TensorLoanRequest{ .shape = loan_shape, .dtype = kUInt8 });
  if (!loan) {
    const auto code = loan.error().code;
    if (code == holoscan::ErrorCode::kResourceExhausted) {
      HOLOSCAN_LOG_WARN("SIPLCaptureOp: tensor pool exhausted; dropping frame");
      return {};
    }
    return holoscan::make_unexpected(std::move(loan).error());
  }

  // RAW10 copies starting past the embedded top rows, matching the active-only tensor_bytes
  // computed above; raw_cpu_ptr itself stays unshifted, since HSB metadata parsing above reads
  // from the start of the full buffer, which is a separate boundary from SIPL's embedded lines.
  const void* raw10_copy_src = (raw_cpu_ptr != nullptr)
      ? static_cast<const std::uint8_t*>(raw_cpu_ptr) +
            static_cast<std::size_t>(acquired.plane_pitch[0]) * embedded_top_lines_
      : nullptr;
  const bool src_is_host = raw10_copy_src != nullptr;
  const void* copy_src = src_is_host ? raw10_copy_src : acquired.cuda_ptr;

  // Branch on the placement the compiled plan actually froze (read back from the loan) rather
  // than on memory_kind_ directly, so the declared and written placements cannot drift.
  // write() is for a device-resident allocation; write_host() is the only entry point admitted
  // for a host-accessible one.
  const holoscan::MemoryKind placement = loan->memory_kind();
  const bool dest_is_host = placement == holoscan::MemoryKind::kHost ||
                             placement == holoscan::MemoryKind::kPinnedHost;

  auto writer = dest_is_host ? loan->write_host() : loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }

  // RAW10 source is a host-mapped NvSciBuf CPU pointer; NV12 source is an ISP device frame.
  // The destination is whichever placement was frozen above, so all four source/destination
  // combinations are possible (e.g. RAW10 on a kHost graph is HostToHost, not HostToDevice).
  cudaMemcpyKind copy_kind;
  if (src_is_host) {
    copy_kind = dest_is_host ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice;
  } else {
    copy_kind = dest_is_host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
  }
  const cudaError_t copy_err =
      cudaMemcpy(writer->data(), copy_src, static_cast<std::size_t>(tensor_bytes), copy_kind);
  if (copy_err != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, "SIPLCaptureOp: cudaMemcpy failed"});
  }

  // Copy complete; release the SIPL buffer so the capture pipeline can reuse the slot.
  frame_guard.release();

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }

  holoscan::schema::ImageT descriptor;
  descriptor.width            = desc_width;
  descriptor.height           = desc_height;
  descriptor.encoding         = desc_encoding;
  // NV12's 8-bit samples need no precision statement (zero means "equals the container width").
  // CUSTOM states no organization at all, so there is no container to report RAW10's 10 bits of
  // precision within either -- the packed layout is documented out-of-band (see
  // tools/decode_sipl_frame.py), not through this field.
  descriptor.significant_bits = 0U;
  // The published tensor starts at full-sensor row embedded_top_lines_ (RAW10 only; the ISP
  // strips embedded rows before NV12 ever reaches this operator), and row 0 really is a sample
  // at that row now that the copy above skips the embedded rows instead of publishing them.
  descriptor.roi_offset_y     = is_raw10 ? static_cast<std::int32_t>(embedded_top_lines_) : 0;
  descriptor.exposure_time_ns = desc_exposure_ns;
  descriptor.gain             = desc_gain;
  // NV12 is 2-plane; provide explicit per-plane layout so consumers can locate the UV plane.
  // RAW10 is single-plane; leave plane_layouts empty and defer to the tensor strides.
  if (is_nv12) {
    // Chroma is subsampled by two vertically; image.fbs rounds an odd extent up so the last luma
    // row keeps a chroma row of its own, hence (desc_height + 1) / 2 rather than desc_height / 2.
    const auto nv12_uv_rows = (static_cast<std::uint64_t>(desc_height) + 1U) / 2U;
    descriptor.plane_layouts = {
        holoscan::schema::ImagePlaneLayout(
            0, desc_y_pitch,
            static_cast<std::uint64_t>(desc_y_pitch) * desc_height),
        holoscan::schema::ImagePlaneLayout(
            nv12_uv_offset, nv12_uv_pitch,
            static_cast<std::uint64_t>(nv12_uv_pitch) * nv12_uv_rows),
    };
  }
  // Frame identity: sequence number, capture timestamp, and coordinate frame name.
  descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
  descriptor.header->frame_id = "sipl_camera_" + std::to_string(camera_index_);
  if (options.frame_id.has_value()) {
    descriptor.header->device_sequence = *options.frame_id;
  }
  if (options.capture_time) {
    descriptor.header->capture_timestamp_ns = options.capture_time->timestamp_ns;
  }
  if (auto fr = frame.emit_tensor(std::move(*loan), descriptor, options); !fr) {
    HOLOSCAN_LOG_ERROR("SIPLCaptureOp: frame.emit_tensor failed: {}", fr.error().message);
    return fr;
  }

  // After the frame: sensor_data is correlated to this frame's identity, so it must not be
  // published when the frame itself was not.
  if (auto sr = sensor_data.emit(make_sensor_metadata(img_meta, hsb_meta, isp_stats_ptr,
          desc_width, desc_height, nv12_uv_pitch, nv12_uv_offset), options); !sr) {
    HOLOSCAN_LOG_WARN("SIPLCaptureOp: failed to emit sensor_data: {}", sr.error().message);
  }
  return {};
}

}  // namespace holoscan::holoscan_camera
