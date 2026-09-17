// SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Port of hololink/src/hololink/operators/sipl_capture/sipl_capture_service.hpp
// Key changes vs the hololink original:
//   - No dependency on hololink::core or HSB headers.
//   - PixelFormat / BayerFormat are plain enums defined here.
//   - buffer_release_callback returns void (not nvidia::gxf::Expected<void>).
//   - cuda_device_ parameter added (hololink hardcoded device 0).

#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <INvSIPLISPStatCustomInterface.hpp>
#include <NvSIPLCamera.hpp>

#include "sipl_capture_op/sipl_compat.hpp"

namespace holoscan::holoscan_camera {

/**
 * @brief Manages a single NvSIPL camera session and per-camera capture resources.
 *
 * One SIPLCaptureService covers an entire camera rig (one or more cameras sharing the same SIPL
 * pipeline). SIPLCaptureOp instances (one per camera) hold a shared_ptr to this service and call
 * add_operator_ref() / remove_operator_ref() to coordinate startup and teardown.
 *
 * No dependency on HSB / hololink headers. SIPL internally uses HSB core through the
 * statically-linked UDDF drivers; that dependency does not surface here.
 */
class SIPLCaptureService : public std::enable_shared_from_this<SIPLCaptureService> {
 public:
  // Values mirror hololink::csi::PixelFormat (RAW_10=1, RAW_12=2) so users of
  // both projects can cast between the two.  RAW_8 is omitted: SIPL sensors
  // only ever produce RAW10 or RAW12 and exposing an unreachable enumerator
  // would be misleading.
  enum class PixelFormat : std::uint8_t { kRaw10 = 1, kRaw12 = 2 };
  // Values mirror hololink::csi::BayerFormat which in turn matches the NPP
  // NppiBayerGridPosition ordering (see nppdefs.html).
  enum class BayerFormat : std::uint8_t { kBggr = 0, kRggb = 1, kGbrg = 2, kGrbg = 3 };

  struct CameraInfo {
    std::string output_name;
    std::uint32_t offset{};                 // byte offset past embedded top lines
    std::uint32_t embedded_top_lines{};     // number of embedded rows before active pixels
    std::uint32_t embedded_bottom_lines{};  // number of embedded rows after active pixels
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t bytes_per_line{};
    PixelFormat pixel_format{ PixelFormat::kRaw10 };
    BayerFormat bayer_format{ BayerFormat::kRggb };
  };

  // Parameter order matches hololink SIPLCaptureService for easy reference.
  // @param cuda_device  CUDA device index used when importing NvSci buffers.
  //   On single-GPU systems (Jetson iGPU) this is always 0.  On multi-GPU
  //   systems (IGX dGPU) pass the index of the GPU that will consume the
  //   frames so the buffer mapping targets the right device.  hololink
  //   hardcodes 0; this parameter was added to support multi-GPU pipelines.
  explicit SIPLCaptureService(
      const std::string& camera_config,
      const std::string& json_config,
      bool raw_output,
      std::uint32_t capture_queue_depth = 4,
      const std::string& nito_base_path = "/var/nvidia/nvcam/settings/sipl",
      std::uint32_t timeout_us = 1'000'000,
      std::int32_t cuda_device = 0);

  ~SIPLCaptureService();

  // Non-copyable, non-movable.
  SIPLCaptureService(const SIPLCaptureService&) = delete;
  SIPLCaptureService& operator=(const SIPLCaptureService&) = delete;

  static void list_available_configs(const std::string& json_config = "");

  // Trigger lazy NvSIPL init on the first call — non-const by design.
  [[nodiscard]] const std::vector<CameraInfo>& get_camera_info();
  [[nodiscard]] std::uint32_t camera_count();

  // Called by each SIPLCaptureOp::on_arm(); allocates buffers + starts streaming on first call.
  void add_operator_ref();
  // Called by each SIPLCaptureOp::on_stop(); tears down when the last ref is released.
  void remove_operator_ref();

  // Register (or clear, by passing {}) the per-operator notification callback.  The callback is
  // invoked by the acquire thread each time a new frame is ready.
  void set_frame_ready_callback(std::uint32_t camera_index, std::function<void()> callback);

  enum class AcquireStatus : std::uint8_t { kOk, kTimeout, kError };

  // Frame borrowed from SIPL until release_acquired_frame() or register_pending_output().
  // plane_* pointers refer to NvSci attribute storage valid for the lifetime of this struct.
  struct AcquiredFrame {
    AcquiredFrame() = default;
    AcquiredFrame(const AcquiredFrame&) = delete;
    AcquiredFrame& operator=(const AcquiredFrame&) = delete;
    AcquiredFrame(AcquiredFrame&&) = default;
    AcquiredFrame& operator=(AcquiredFrame&&) = default;

    AcquireStatus status{ AcquireStatus::kError };

    nvsipl::INvSIPLClient::INvSIPLBuffer* buffer{};
    nvsipl::INvSIPLClient::INvSIPLBuffer* buffer_raw{};
    nvsipl::INvSIPLClient::INvSIPLBuffer* buffer_isp{};
    NvSciBufObj buf_obj{};
    void* cuda_ptr{};
    std::uint64_t buffer_size{};
    std::uint32_t plane_count{};
    const std::uint32_t* plane_pitch{};
    const std::uint32_t* plane_width{};
    const std::uint32_t* plane_height{};
    const std::uint64_t* plane_offset{};
    const NvSciBufAttrValColorFmt* plane_color_format{};
  };

  AcquiredFrame acquire_frame(std::uint32_t camera_index, std::uint32_t timeout_us);

  // Release SIPL buffers when processing fails before handoff_pending_output().
  // After handoff_pending_output() the buffer is owned by the DLPack tensor deleter
  // (buffer_release_callback); calling this afterwards would double-release.
  void release_acquired_frame(AcquiredFrame& frame);

  const std::string& output_name(std::uint32_t camera_index) const;

  // Returns the ISP statistics interface for the given camera, or nullptr in
  // RAW10 mode (ISP pipeline not active) or before init_cameras() has run.
  [[nodiscard]] nvsipl::INvSIPLISPStatCustomInterface* isp_stats(
      std::uint32_t camera_index) const;
  [[nodiscard]] std::uint32_t default_timeout_us() const { return timeout_us_; }
  [[nodiscard]] bool raw_output() const { return raw_output_; }
  // cuda_device is not present in the hololink version (hardcoded to 0 there).
  [[nodiscard]] std::int32_t cuda_device() const { return cuda_device_; }

  void release_raw_buffer_if_unused(nvsipl::INvSIPLClient::INvSIPLBuffer* buffer,
                                    nvsipl::INvSIPLClient::INvSIPLBuffer* buffer_raw);
  void register_pending_output(void* cuda_ptr, nvsipl::INvSIPLClient::INvSIPLBuffer* buffer);

  // DLManagedTensor deleter: looks up the service from the static map and releases
  // the SIPL buffer that backs the tensor's CUDA memory.
  static void buffer_release_callback(void* cuda_ptr) noexcept;

 private:
  struct PerCameraState {
    std::string output_name_;
    nvsipl::INvSIPLISPStatCustomInterface* isp_stats_{};
    nvsipl::NvSIPLPipelineQueues queues_;
    std::vector<NvSciBufObj> sci_bufs_icp_;
    std::vector<NvSciBufObj> sci_bufs_isp0_;
    NvSciSyncObj sci_sync_isp0_{};
    // One per camera, not shared: NvSciSyncCpuWaitContext wraps an OS-level wait primitive
    // meant for one owning thread. acquire_frame() is called independently by each camera's
    // SIPLCaptureOp::compute() thread, so a single service-wide context would be waited on
    // concurrently by both -- undefined as far as that API's contract goes.
    NvSciSyncCpuWaitContext cpu_wait_context_{};

    std::thread acquire_thread_;
    std::unique_ptr<std::atomic<bool>> stop_thread_{ std::make_unique<std::atomic<bool>>(false) };
    std::unique_ptr<std::mutex> buffer_mutex_{ std::make_unique<std::mutex>() };
    std::unique_ptr<std::condition_variable> buffer_available_{
        std::make_unique<std::condition_variable>() };
    nvsipl::INvSIPLClient::INvSIPLBuffer* buffer_raw_{};
    nvsipl::INvSIPLClient::INvSIPLBuffer* buffer_isp_{};
    std::function<void()> frame_ready_callback_;
    std::unique_ptr<std::mutex> frame_ready_mutex_{ std::make_unique<std::mutex>() };
  };

  struct CudaBufferMapping {
    cudaExternalMemory_t mem_;
    void* ptr_;
  };

  void init_cameras();
  void init_nvsipl();
  void init_nvsci();
  void fill_camera_info();
  void allocate_buffers(std::uint32_t camera_index,
                        nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type,
                        std::vector<NvSciBufObj>& bufs);
  void register_buffers(std::uint32_t camera_index,
                        nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type,
                        const std::vector<NvSciBufObj>& bufs);
  void free_buffers(std::vector<NvSciBufObj>& bufs);
  void allocate_sync(std::uint32_t camera_index,
                     nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type,
                     NvSciSyncObj& sync);
  void register_autocontrol(std::uint32_t camera_index);
  bool load_nito_file(std::string name, std::vector<std::uint8_t>& nito);
  void start_buffers();
  void stop_buffers();
  void teardown_initialized_state();
  void teardown_buffer_allocations();
  void forfeit_pending_outputs();
  void ensure_streaming_started();
  void on_pending_output_released(void* cuda_ptr) noexcept;
  void acquire_buffer_thread_func(PerCameraState* camera_state);
  void* map_buffer_to_cuda(NvSciBufObj buf_obj, std::uint64_t size);

  std::string camera_config_;
  std::string json_config_;
  bool raw_output_;
  std::uint32_t capture_queue_depth_;
  std::string nito_base_path_;
  std::uint32_t timeout_us_;

  std::unique_ptr<nvsipl::INvSIPLCameraQuery> sipl_query_;
  sipl_compat::SystemConfig sipl_config_;
  std::unique_ptr<nvsipl::INvSIPLCamera> sipl_camera_;

  NvSciBufModule sci_buf_module_{};
  NvSciSyncModule sci_sync_module_{};

  std::vector<PerCameraState> per_camera_state_;
  std::vector<CameraInfo> camera_info_;

  std::map<NvSciBufObj, CudaBufferMapping> cuda_mappings_;

  bool initialized_{ false };
  bool buffers_started_{ false };
  bool streaming_{ false };

  std::mutex init_mutex_;
  std::mutex ref_mutex_;
  std::uint32_t operator_ref_count_{ 0 };
  std::mutex streaming_mutex_;
  std::mutex cuda_mappings_mutex_;

  std::map<void*, nvsipl::INvSIPLClient::INvSIPLBuffer*> pending_outputs_;
  std::mutex pending_outputs_mutex_;
  std::condition_variable pending_output_released_;

  std::int32_t cuda_device_;

  static std::map<void*, std::weak_ptr<SIPLCaptureService>> pending_release_targets_;
  static std::mutex pending_release_targets_mutex_;
};

}  // namespace holoscan::holoscan_camera
