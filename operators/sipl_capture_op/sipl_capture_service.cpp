// SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Port of hololink/src/hololink/operators/sipl_capture/sipl_capture_service.cpp
// Key changes vs the hololink original:
//   - No dependency on hololink::core or HSB headers.
//   - HSB_LOG_* → HOLOSCAN_LOG_*.
//   - hololink::csi::PixelFormat / BayerFormat → SIPLCaptureService::PixelFormat / BayerFormat.
//   - buffer_release_callback / on_pending_output_released return void (not gxf::Expected<void>).
//   - cuda_device_ used in place of hardcoded device 0.
//   - list_available_configs uses sipl_compat accessors instead of sipl_fmt formatters.

#include "sipl_capture_op/sipl_capture_service.hpp"

#include <cuda.h>
#include <strings.h>

#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>

namespace holoscan::holoscan_camera {

namespace sc = sipl_compat;

std::map<void*, std::weak_ptr<SIPLCaptureService>> SIPLCaptureService::pending_release_targets_;
std::mutex SIPLCaptureService::pending_release_targets_mutex_;

namespace {

struct CloseNvSciBufAttrList {
    void operator()(NvSciBufAttrList* attr_list) const {
        if (attr_list != nullptr) {
            if ((*attr_list) != nullptr) {
                NvSciBufAttrListFree(*attr_list);
            }
            delete attr_list;
        }
    }
};

void print_config_details(const std::string& config_name, const sc::SystemConfig& camera_config) {
    std::cout << config_name << ":\n";
    for (const auto& mod : sc::get_modules(camera_config)) {
        std::cout << "  module: " << sc::get_module_name(mod) << "\n"
                  << "    resolution:          " << sc::get_resolution_width(mod)
                  << "x" << sc::get_resolution_height(mod) << "\n"
                  << "    cfa:                 " << sc::get_cfa(mod) << "\n"
                  << "    inputFormat:         " << static_cast<int>(sc::get_input_format(mod)) << "\n"
                  << "    embeddedTopLines:    " << sc::get_embedded_top_lines(mod) << "\n";
    }
    std::cout << "\n";
}

}  // namespace


SIPLCaptureService::SIPLCaptureService(const std::string& camera_config,
                                       const std::string& json_config,
                                       bool raw_output,
                                       std::uint32_t capture_queue_depth,
                                       const std::string& nito_base_path,
                                       std::uint32_t timeout_us,
                                       std::int32_t cuda_device)
    : camera_config_(camera_config)
    , json_config_(json_config)
    , raw_output_(raw_output)
    , capture_queue_depth_(capture_queue_depth)
    , nito_base_path_(nito_base_path)
    , timeout_us_(timeout_us)
    , cuda_device_(cuda_device) {
}

SIPLCaptureService::~SIPLCaptureService() {
    if (buffers_started_) {
        stop_buffers();
    } else if (initialized_) {
        teardown_initialized_state();
    }
}

void SIPLCaptureService::buffer_release_callback(void* cuda_ptr) noexcept {
    std::shared_ptr<SIPLCaptureService> service;
    {
        std::lock_guard<std::mutex> lock(pending_release_targets_mutex_);
        auto target = pending_release_targets_.find(cuda_ptr);
        if (target != pending_release_targets_.end()) {
            service = target->second.lock();
            pending_release_targets_.erase(target);
        }
    }
    if (!service) {
        HOLOSCAN_LOG_WARN("No live SIPLCaptureService for pending buffer {}", cuda_ptr);
        return;
    }
    service->on_pending_output_released(cuda_ptr);
}

void SIPLCaptureService::on_pending_output_released(void* cuda_ptr) noexcept {
    std::lock_guard<std::mutex> lock(pending_outputs_mutex_);
    auto pending = pending_outputs_.find(cuda_ptr);
    if (pending == pending_outputs_.end()) {
        HOLOSCAN_LOG_ERROR("Unknown pending buffer {}", cuda_ptr);
        return;
    }
    auto status = pending->second->Release();
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        HOLOSCAN_LOG_ERROR("Failed to release buffer {}", static_cast<void*>(pending->second));
    } else {
        HOLOSCAN_LOG_TRACE("Released buffer {}", static_cast<void*>(pending->second));
    }
    pending_outputs_.erase(pending);
    pending_output_released_.notify_all();
}

void SIPLCaptureService::list_available_configs(const std::string& json_config) {
    auto sipl_query = nvsipl::INvSIPLCameraQuery::GetInstance();
    if (!sipl_query) {
        throw std::runtime_error("Failed to get NvSIPLCameraQuery instance");
    }

    // Always load the UDDF driver database first so transport drivers (e.g. HsbTransport)
    // are registered before the custom JSON overlay is applied.
    if (sipl_query->ParseDatabase() != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error("Failed to parse NvSIPLCameraQuery database");
    }
    if (!json_config.empty()) {
        if (sipl_query->ParseJsonFile(json_config) != nvsipl::NVSIPL_STATUS_OK) {
            throw std::runtime_error("Failed to parse NvSIPLCameraQuery JSON config");
        }
    }

    // Get available configuration names.
    const auto& config_names = sipl_query->GetCameraConfigNames();
    if (config_names.empty()) {
        throw std::runtime_error("No camera configurations available");
    }

    // Detect and warn about duplicate config names (e.g. if multiple
    // driver libraries use the same config names).
    std::set<std::string> unique_configs(config_names.begin(), config_names.end());
    if (config_names.size() != unique_configs.size()) {
        HOLOSCAN_LOG_WARN("Duplicate camera config names found in database");
    }

    // Print out the config details.
    std::cout << unique_configs.size() << " Available Camera Configurations:\n\n";
    for (const auto& config_name : unique_configs) {
        sc::SystemConfig camera_config;
        auto status = sc::get_system_config(*sipl_query, config_name, camera_config);
        if (status != nvsipl::NVSIPL_STATUS_OK) {
            throw std::runtime_error("Failed to get camera system config");
        }
        print_config_details(config_name, camera_config);
    }
}

void SIPLCaptureService::init_cameras() {
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (initialized_) {
        return;
    }
    try {
        init_nvsipl();
        init_nvsci();
        fill_camera_info();
        initialized_ = true;
    } catch (...) {
        teardown_initialized_state();
        throw;
    }
}

void SIPLCaptureService::init_nvsipl() {
    sipl_query_ = nvsipl::INvSIPLCameraQuery::GetInstance();
    if (!sipl_query_) {
        throw std::runtime_error("Failed to get NvSIPLCameraQuery instance");
    }

    // Always load the UDDF driver database first so transport drivers (e.g. HsbTransport)
    // are registered before the custom JSON overlay is applied.
    if (sipl_query_->ParseDatabase() != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error("Failed to parse NvSIPLCameraQuery database");
    }
    if (!json_config_.empty()) {
        if (sipl_query_->ParseJsonFile(json_config_) != nvsipl::NVSIPL_STATUS_OK) {
            throw std::runtime_error("Failed to parse NvSIPLCameraQuery JSON config");
        }
    }

    // Select the first available configuration whose first module has a non-empty platformConfig.
    // This picks up the platform identifier (e.g. "VB1940") directly from the parsed SIPL data.
    if (camera_config_.empty()) {
        for (const auto& name : sipl_query_->GetCameraConfigNames()) {
            sc::SystemConfig cfg;
            if (sc::get_system_config(*sipl_query_, name, cfg) != nvsipl::NVSIPL_STATUS_OK) {
                continue;
            }
            const auto& mods = sc::get_modules(cfg);
            if (!mods.empty() && !mods[0].platformConfig.empty()) {
                camera_config_ = name;
                break;
            }
        }
        if (camera_config_.empty()) {
            throw std::runtime_error(
                "No camera configuration with platformConfig found in database or JSON");
        }
    }

    // Resolve: exact match first, then case-insensitive fallback.
    auto status = sc::get_system_config(*sipl_query_, camera_config_, sipl_config_);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        for (const auto& name : sipl_query_->GetCameraConfigNames()) {
            if (strcasecmp(name.c_str(), camera_config_.c_str()) == 0) {
                camera_config_ = name;
                status = sc::get_system_config(*sipl_query_, camera_config_, sipl_config_);
                break;
            }
        }
    }
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error("Camera config '" + camera_config_ + "' not found");
    }

    // Open the SIPL instance.
    sipl_camera_ = nvsipl::INvSIPLCamera::GetInstance();
    if (!sipl_camera_) {
        throw std::runtime_error("Failed to get NvSIPLCamera instance");
    }

    // Set the platform config.
    status = sipl_camera_->SetPlatformCfg(sipl_config_);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error(
            "Failed to set NvSIPLCamera platform config (SIPLStatus=" +
            std::to_string(static_cast<int>(status)) +
            "). If status=2 (NOT_SUPPORTED), Camera HAL is not compiled into libnvsipl.so.");
    }

    // Allocate the per-camera state.
    per_camera_state_.resize(sc::get_modules(sipl_config_).size());

    // Set the pipeline configs to capture either RAW or ISP-processed images.
    nvsipl::NvSIPLPipelineConfiguration sipl_pipeline_config = {
        .captureOutputRequested = true,
        .isp0OutputRequested = !raw_output_,
        .isp1OutputRequested = false,
        .isp2OutputRequested = false,
        .disableSubframe = true
    };
    for (uint32_t camera_index = 0; camera_index < per_camera_state_.size(); ++camera_index) {
        auto& camera_state = per_camera_state_[camera_index];
        status = sipl_camera_->SetPipelineCfg(camera_index, sipl_pipeline_config, camera_state.queues_);
        if (status != nvsipl::NVSIPL_STATUS_OK) {
            throw std::runtime_error("Failed to set NvSIPLCamera pipeline config");
        }

        if (!raw_output_) {
            // Use the pipeline interface provider to get the ISP stats interface.
            nvsipl::IInterfaceProvider* interface_provider;
            status = sipl_camera_->GetPipelineInterfaceProvider(camera_index, interface_provider);
            if (status != nvsipl::NVSIPL_STATUS_OK || !interface_provider) {
                throw std::runtime_error(
                    "Failed to get NvSIPLCamera pipeline interface provider");
            }
            camera_state.isp_stats_ =
                static_cast<nvsipl::INvSIPLISPStatCustomInterface*>(
                    interface_provider->GetInterface(
                        nvsipl::INvSIPLISPStatCustomInterface::getClassInterfaceID()));
            if (!camera_state.isp_stats_) {
                throw std::runtime_error("Failed to get INvSIPLISPStatCustomInterface");
            }
        }

        camera_state.output_name_ =
            sc::get_module_name(sc::get_modules(sipl_config_)[camera_index]) + "_" + std::to_string(camera_index);
    }

    // Initialize SIPL
    status = sipl_camera_->Init();
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error("Failed to initialize NvSIPLCamera");
    }
}

void SIPLCaptureService::init_nvsci() {
    NvSciError err;

    err = NvSciSyncModuleOpen(&sci_sync_module_);
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to initialize NvSciSyncModule");
    }

    err = NvSciBufModuleOpen(&sci_buf_module_);
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to initialize NvSciBufModule");
    }

    // One context per camera, not one shared: see the comment on PerCameraState::
    // cpu_wait_context_ in the header for why. per_camera_state_ is already sized by
    // init_nvsipl(), which runs before init_nvsci() in init_cameras().
    for (auto& camera_state : per_camera_state_) {
        err = NvSciSyncCpuWaitContextAlloc(sci_sync_module_, &camera_state.cpu_wait_context_);
        if (err != NvSciError_Success) {
            throw std::runtime_error("Failed to allocate NvSciCpuWaitContext");
        }
    }
}

void SIPLCaptureService::fill_camera_info() {
    const auto& modules = sc::get_modules(sipl_config_);
    camera_info_.resize(modules.size());
    for (uint32_t camera_index = 0; camera_index < camera_info_.size(); ++camera_index) {
        // Get and reconcile the buffer attributes for the given camera in order to
        // determine the buffer pitch.
        std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> attr_list;
        attr_list.reset(new NvSciBufAttrList());
        auto err = NvSciBufAttrListCreate(sci_buf_module_, attr_list.get());
        if (err != NvSciError_Success) {
            throw std::runtime_error("Failed to create NvSciBufAttrList");
        }

        const auto& module = modules[camera_index];
        auto status = sipl_camera_->GetImageAttributes(
            sc::get_sensor_id(module),
            nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ICP,
            *(attr_list.get()));
        if (status != nvsipl::NVSIPL_STATUS_OK) {
            throw std::runtime_error("Failed to get image attributes (" +
                                     std::to_string(static_cast<int>(status)) + ")");
        }

        std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> reconciled_attr_list;
        std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> conflict_attr_list;
        reconciled_attr_list.reset(new NvSciBufAttrList());
        conflict_attr_list.reset(new NvSciBufAttrList());
        err = NvSciBufAttrListReconcile(attr_list.get(),
            1U,
            reconciled_attr_list.get(),
            conflict_attr_list.get());
        if (err != NvSciError_Success) {
            throw std::runtime_error("Failed to reconcile NvSciBuf attributes");
        }

        NvSciBufAttrKeyValuePair img_attrs[] = { { NvSciBufImageAttrKey_PlanePitch, NULL, 0 } };
        err = NvSciBufAttrListGetAttrs(*reconciled_attr_list, img_attrs, 1);
        if (err != NvSciError_Success) {
            throw std::runtime_error("Failed to get buffer attributes");
        }
        const std::uint32_t plane_pitch = *(static_cast<const std::uint32_t*>(img_attrs[0].value));

        // Fill in the camera information.
        auto& info = camera_info_[camera_index];
        info.output_name = per_camera_state_[camera_index].output_name_;
        info.offset = 0;
        info.width = sc::get_resolution_width(module);
        info.height = sc::get_resolution_height(module);
        info.bytes_per_line = plane_pitch;
        // Embedded lines are a property of the CSI capture itself, not of RAW10 vs. RAW12, so
        // both cases below read the same value; only the bits-per-pixel divisor in offset differs.
        info.embedded_top_lines = sc::get_embedded_top_lines(module);
        info.embedded_bottom_lines = sc::get_embedded_bottom_lines(module);

        // Pixel format
        switch (sc::get_input_format(module)) {
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10:
#if !SIPL_V2 || defined(NV_EMBEDDED_L4T)
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10TP:
#endif
            info.pixel_format = PixelFormat::kRaw10;
            info.offset = info.bytes_per_line * info.embedded_top_lines;
            break;
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12:
#if SIPL_V2 && defined(NV_EMBEDDED_L4T)
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12TP:
#endif
            info.pixel_format = PixelFormat::kRaw12;
            info.offset = info.bytes_per_line * info.embedded_top_lines;
            break;
        default:
            throw std::runtime_error("Unsupported input format");
        }

        // Bayer order
        switch (sc::get_cfa(module)) {
        case NVSIPL_PIXEL_ORDER_RGGB:
            info.bayer_format = BayerFormat::kRggb;
            break;
        case NVSIPL_PIXEL_ORDER_BGGR:
            info.bayer_format = BayerFormat::kBggr;
            break;
        case NVSIPL_PIXEL_ORDER_GRBG:
            info.bayer_format = BayerFormat::kGrbg;
            break;
        case NVSIPL_PIXEL_ORDER_GBRG:
            info.bayer_format = BayerFormat::kGbrg;
            break;
        default:
            throw std::runtime_error("Unsupported color filter array");
        }
    }
}

void SIPLCaptureService::allocate_buffers(
    std::uint32_t camera_index,
    nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type,
    std::vector<NvSciBufObj>& bufs) {

    // Create the requested attribute list.
    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> attr_list;
    attr_list.reset(new NvSciBufAttrList());
    NvSciError err = NvSciBufAttrListCreate(sci_buf_module_, attr_list.get());
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to create NvSciBufAttrList");
    }

    // Common attributes (to ICP and ISP outputs).
    constexpr NvSciBufType buf_type = NvSciBufType_Image;
    constexpr NvSciBufAttrValAccessPerm access_perm = NvSciBufAccessPerm_Readonly;

    // GpuId is required for ISP output (NV12) so the buffer can be mapped into CUDA.
    // In RAW10 mode (ICP only) the operator uses NvSciBufObjGetConstCpuPtr and never
    // maps the buffer into CUDA, so we skip the GPU attribute setup entirely.
    NvSciRmGpuId gpu_id{};
    if (!raw_output_) {
        CUresult cu_init = cuInit(0);
        if (cu_init != CUDA_SUCCESS) {
            const char* err_str = nullptr;
            cuGetErrorString(cu_init, &err_str);
            throw std::runtime_error(
                std::string("cuInit failed: ") + (err_str ? err_str : "unknown"));
        }
        CUuuid uuid;
        const CUresult cu_err = cuDeviceGetUuid_v2(&uuid, cuda_device_);
        if (cu_err != CUDA_SUCCESS) {
            const char* err_str = nullptr;
            cuGetErrorString(cu_err, &err_str);
            throw std::runtime_error(
                std::string("cuDeviceGetUuid_v2 failed: ") + (err_str ? err_str : "unknown"));
        }
        memcpy(&gpu_id.bytes, uuid.bytes, sizeof(uuid.bytes));
    }

    // Require CPU read access for ISP output.
    constexpr bool is_cpu_access_req = true;
    constexpr bool is_cpu_cache_enabled = true;

    // Require pitch-linear NV12 for ISP output.
    constexpr NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
    constexpr NvSciBufSurfType surf_type = NvSciSurfType_YUV;
    constexpr NvSciBufSurfSampleType surf_sample_type = NvSciSurfSampleType_420;
    constexpr NvSciBufSurfBPC surf_bpc = NvSciSurfBPC_8;
    constexpr NvSciBufSurfMemLayout surf_mem_layout = NvSciSurfMemLayout_SemiPlanar;
    constexpr NvSciBufSurfComponentOrder surf_comp_order = NvSciSurfComponentOrder_YUV;
    constexpr NvSciBufAttrValColorStd surf_color_std[] = { NvSciColorStd_REC709_ER };

    NvSciBufAttrKeyValuePair attr_kvp[] = {
        { NvSciBufGeneralAttrKey_Types,          &buf_type,            sizeof(buf_type) },
        { NvSciBufGeneralAttrKey_RequiredPerm,   &access_perm,         sizeof(access_perm) },
        { NvSciBufGeneralAttrKey_GpuId,          &gpu_id,              sizeof(gpu_id) },
        // ISP-specific attributes:
        { NvSciBufGeneralAttrKey_NeedCpuAccess,  &is_cpu_access_req,   sizeof(is_cpu_access_req) },
        { NvSciBufGeneralAttrKey_EnableCpuCache, &is_cpu_cache_enabled, sizeof(is_cpu_cache_enabled) },
        { NvSciBufImageAttrKey_Layout,           &layout,              sizeof(layout) },
        { NvSciBufImageAttrKey_SurfType,         &surf_type,           sizeof(surf_type) },
        { NvSciBufImageAttrKey_SurfBPC,          &surf_bpc,            sizeof(surf_bpc) },
        { NvSciBufImageAttrKey_SurfMemLayout,    &surf_mem_layout,     sizeof(surf_mem_layout) },
        { NvSciBufImageAttrKey_SurfSampleType,   &surf_sample_type,    sizeof(surf_sample_type) },
        { NvSciBufImageAttrKey_SurfComponentOrder, &surf_comp_order,   sizeof(surf_comp_order) },
        { NvSciBufImageAttrKey_SurfColorStd,     &surf_color_std,      sizeof(surf_color_std) },
    };

    const std::size_t num_attrs =
        (output_type == nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ICP)
            ? 2U   // ICP (RAW) buffers: Types + RequiredPerm only; GpuId not needed (CPU path)
            : sizeof(attr_kvp) / sizeof(attr_kvp[0]);
    err = NvSciBufAttrListSetAttrs(*(attr_list.get()), attr_kvp, num_attrs);
    if (err != NvSciError_Success) {
        throw std::runtime_error(
            "Failed to set NvSciBufAttrList values (NvSciError=" + std::to_string(err) + ")");
    }

    // Get the attributes provided by the camera.
    const auto sensor_id = sc::get_sensor_id(sc::get_modules(sipl_config_)[camera_index]);
    auto status = sipl_camera_->GetImageAttributes(sensor_id, output_type, *(attr_list.get()));
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error("Failed to get image attributes (" +
                                 std::to_string(static_cast<int>(status)) + ")");
    }

    // Reconcile the attributes.
    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> reconciled_attr_list;
    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> conflict_attr_list;
    reconciled_attr_list.reset(new NvSciBufAttrList());
    conflict_attr_list.reset(new NvSciBufAttrList());
    err = NvSciBufAttrListReconcile(attr_list.get(),
        1U,
        reconciled_attr_list.get(),
        conflict_attr_list.get());
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to reconcile NvSciBuf attributes");
    }

    // Allocate the buffers.
    for (std::size_t i = 0; i < capture_queue_depth_; ++i) {
        NvSciBufObj buf_obj;
        err = NvSciBufObjAlloc(*(reconciled_attr_list.get()), &buf_obj);
        if (err != NvSciError_Success || !buf_obj) {
            throw std::runtime_error("Failed to allocate NvSciBufObj");
        }
        bufs.push_back(buf_obj);
    }

    HOLOSCAN_LOG_DEBUG("Allocated {} buffers for output type {}",
        bufs.size(), static_cast<std::uint32_t>(output_type));
}

void SIPLCaptureService::register_buffers(
    std::uint32_t camera_index,
    nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type,
    const std::vector<NvSciBufObj>& bufs) {

    const auto sensor_id = sc::get_sensor_id(sc::get_modules(sipl_config_)[camera_index]);
    auto status = sipl_camera_->RegisterImages(sensor_id, output_type, bufs);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error("Failed to register images with NvSIPLCamera");
    }

    HOLOSCAN_LOG_DEBUG("Registered {} buffers for output type {}",
        bufs.size(), static_cast<std::uint32_t>(output_type));
}

void SIPLCaptureService::free_buffers(std::vector<NvSciBufObj>& bufs) {
    for (auto buf : bufs) {
        NvSciBufObjFree(buf);
    }
    bufs.clear();
}

void SIPLCaptureService::allocate_sync(
    std::uint32_t camera_index,
    nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type,
    NvSciSyncObj& sync) {

    // Helper for cleaning up attribute lists
    struct CloseNvSciSyncAttrList {
        void operator()(NvSciSyncAttrList* attr_list) const {
            if (attr_list != nullptr) {
                if ((*attr_list) != nullptr) {
                    NvSciSyncAttrListFree(*attr_list);
                }
                delete attr_list;
            }
        }
    };

    // Create the CPU waiter attribute list.
    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> waiter_attr_list;
    waiter_attr_list.reset(new NvSciSyncAttrList());
    auto err = NvSciSyncAttrListCreate(sci_sync_module_, waiter_attr_list.get());
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to create waiter NvSciSyncAttrList");
    }

    NvSciSyncAttrKeyValuePair kv[2];
    memset(kv, 0, sizeof(kv));

    bool cpu_signaler_waiter = true;
    kv[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    kv[0].value = &cpu_signaler_waiter;
    kv[0].len = sizeof(cpu_signaler_waiter);

    NvSciSyncAccessPerm cpu_perm = NvSciSyncAccessPerm_WaitOnly;
    kv[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    kv[1].value = &cpu_perm;
    kv[1].len = sizeof(cpu_perm);

    err = NvSciSyncAttrListSetAttrs(*(waiter_attr_list.get()), kv, 2);
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to set NvSciSyncAttrList values");
    }

    // Get the camera signaler attribute list.
    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> signaler_attr_list;
    signaler_attr_list.reset(new NvSciSyncAttrList());
    err = NvSciSyncAttrListCreate(sci_sync_module_, signaler_attr_list.get());
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to create signaler NvSciSyncAttrList");
    }

    const auto sensor_id = sc::get_sensor_id(sc::get_modules(sipl_config_)[camera_index]);
    auto status = sipl_camera_->FillNvSciSyncAttrList(
        sensor_id, output_type, *(signaler_attr_list.get()), nvsipl::SIPL_SIGNALER);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error("Failure in FillNvSciSyncAttrList");
    }

    // Reconcile the attributes.
    NvSciSyncAttrList unreconciled_list[2] = {
        *(waiter_attr_list.get()), *(signaler_attr_list.get())
    };
    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> reconciled_attr_list;
    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> conflict_attr_list;
    reconciled_attr_list.reset(new NvSciSyncAttrList());
    conflict_attr_list.reset(new NvSciSyncAttrList());

    err = NvSciSyncAttrListReconcile(unreconciled_list, 2, reconciled_attr_list.get(), conflict_attr_list.get());
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to reconcile NvSciSync attributes");
    }

    // Allocate the sync object.
    err = NvSciSyncObjAlloc(*(reconciled_attr_list.get()), &sync);
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to allocate NvSciSyncObj");
    }

    // Register the sync object.
    status = sipl_camera_->RegisterNvSciSyncObj(sensor_id, output_type, nvsipl::NVSIPL_EOFSYNCOBJ, sync);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error("Failed to register NvSciSyncObj");
    }

    HOLOSCAN_LOG_DEBUG("Created and registered sync object for output type {}",
        static_cast<std::uint32_t>(output_type));
}

void SIPLCaptureService::register_autocontrol(std::uint32_t camera_index) {
    const auto& module = sc::get_modules(sipl_config_)[camera_index];
    std::vector<std::uint8_t> blob;
    if (!load_nito_file(std::string(sc::get_module_name(module)), blob)) {
        throw std::runtime_error("Failed to load NITO file for autocontrol");
    }

    const auto sensor_id = sc::get_sensor_id(module);
    auto status = sipl_camera_->RegisterAutoControlPlugin(sensor_id, nvsipl::NV_PLUGIN, nullptr, blob);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error(
            "Failed to register autocontrol plugin (camera_index=" + std::to_string(camera_index) +
            ", sensor_id=" + std::to_string(sensor_id) +
            ", status=" + std::to_string(static_cast<int>(status)) + ")");
    }
}

const std::vector<SIPLCaptureService::CameraInfo>& SIPLCaptureService::get_camera_info() {
    init_cameras();
    return camera_info_;
}

std::uint32_t SIPLCaptureService::camera_count() {
    if (!initialized_) {
        init_cameras();
    }
    return static_cast<std::uint32_t>(per_camera_state_.size());
}

void SIPLCaptureService::add_operator_ref() {
    std::lock_guard<std::mutex> lock(ref_mutex_);
    if (operator_ref_count_ == 0) {
        start_buffers();
        ensure_streaming_started();
        // Increment AFTER start/ensure so a thrown exception leaves the count at 0
        // and the subsequent remove_operator_ref() is a safe no-op.
        operator_ref_count_ = 1;
    } else {
        ++operator_ref_count_;
    }
}

void SIPLCaptureService::remove_operator_ref() {
    std::lock_guard<std::mutex> lock(ref_mutex_);
    if (operator_ref_count_ == 0) {
        return;
    }
    if (--operator_ref_count_ == 0) {
        stop_buffers();
    }
}

void SIPLCaptureService::set_frame_ready_callback(
    std::uint32_t camera_index, std::function<void()> callback) {
    auto& state = per_camera_state_[camera_index];
    std::lock_guard<std::mutex> lock(*state.frame_ready_mutex_);
    state.frame_ready_callback_ = std::move(callback);
}


void SIPLCaptureService::start_buffers() {
    if (buffers_started_) {
        return;
    }
    try {
        init_cameras();

        // Allocate buffers for every camera first.
        for (uint32_t camera_index = 0; camera_index < per_camera_state_.size(); ++camera_index) {
            auto& camera_state = per_camera_state_[camera_index];

            // Allocate RAW capture buffers.
            allocate_buffers(camera_index, nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ICP,
                             camera_state.sci_bufs_icp_);

            // Allocate ISP0 output buffers.
            if (!raw_output_) {
                allocate_buffers(camera_index, nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ISP0,
                                 camera_state.sci_bufs_isp0_);
            }
        }

        // Register every camera's buffers with SIPL.
        for (uint32_t camera_index = 0; camera_index < per_camera_state_.size(); ++camera_index) {
            auto& camera_state = per_camera_state_[camera_index];

            register_buffers(camera_index, nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ICP,
                             camera_state.sci_bufs_icp_);

            if (!raw_output_) {
                register_buffers(camera_index, nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ISP0,
                                 camera_state.sci_bufs_isp0_);
            }
        }

        // Allocate sync objects and register autocontrol.
        if (!raw_output_) {
            for (uint32_t camera_index = 0; camera_index < per_camera_state_.size(); ++camera_index) {
                auto& camera_state = per_camera_state_[camera_index];

                allocate_sync(camera_index, nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ISP0,
                              camera_state.sci_sync_isp0_);

                // A synced sensor group shares one autocontrol instance, so only the primary
                // sensor of the group registers it.
                if (sc::is_sensor_group_secondary(sc::get_modules(sipl_config_)[camera_index])) {
                    HOLOSCAN_LOG_DEBUG(
                        "Skipping autocontrol for camera {}: secondary of a sensor group",
                        camera_index);
                    continue;
                }
                register_autocontrol(camera_index);
            }
        }
        buffers_started_ = true;
    } catch (...) {
        teardown_initialized_state();
        throw;
    }
}

void SIPLCaptureService::stop_buffers() {
    if (!buffers_started_) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(streaming_mutex_);
        if (streaming_) {
            for (auto& camera_state : per_camera_state_) {
                camera_state.stop_thread_->store(true);
            }
            for (auto& camera_state : per_camera_state_) {
                if (camera_state.acquire_thread_.joinable()) {
                    camera_state.acquire_thread_.join();
                }
            }
            sipl_camera_->Stop();
            streaming_ = false;
        }
    }

    // Wait for any pending buffers to be released.
    std::uint32_t wait_limit = 30;
    std::unique_lock<std::mutex> pending_buffer_lock(pending_outputs_mutex_);
    while (!pending_outputs_.empty()) {
        auto status = pending_output_released_.wait_for(pending_buffer_lock,
            std::chrono::milliseconds(100));
        if (status == std::cv_status::timeout && (--wait_limit == 0)) {
            HOLOSCAN_LOG_ERROR("Failed to wait for pending buffers to be released.");
            break;
        }
    }
    if (!pending_outputs_.empty()) {
        HOLOSCAN_LOG_ERROR("{} pending SIPL buffers still in use; skipping CUDA and NvSci buffer cleanup",
            pending_outputs_.size());
        pending_buffer_lock.unlock();
        forfeit_pending_outputs();
        return;
    }
    pending_buffer_lock.unlock();

    {
        std::lock_guard<std::mutex> lock(cuda_mappings_mutex_);
        for (auto& [buf_obj, mapping] : cuda_mappings_) {
            cudaFree(mapping.ptr_);
            cudaDestroyExternalMemory(mapping.mem_);
        }
        cuda_mappings_.clear();
    }

    // Cleanup per-camera state.
    for (auto& camera_state : per_camera_state_) {
        // Free sync objects.
        if (camera_state.sci_sync_isp0_) {
            NvSciSyncObjFree(camera_state.sci_sync_isp0_);
            camera_state.sci_sync_isp0_ = nullptr;
        }
        if (camera_state.cpu_wait_context_) {
            NvSciSyncCpuWaitContextFree(camera_state.cpu_wait_context_);
            camera_state.cpu_wait_context_ = nullptr;
        }

        // Free buffers.
        free_buffers(camera_state.sci_bufs_icp_);
        free_buffers(camera_state.sci_bufs_isp0_);
    }
    per_camera_state_.clear();

    // Cleanup NvSciSync/Buf modules.
    if (sci_sync_module_) {
        NvSciSyncModuleClose(sci_sync_module_);
        sci_sync_module_ = nullptr;
    }
    if (sci_buf_module_) {
        NvSciBufModuleClose(sci_buf_module_);
        sci_buf_module_ = nullptr;
    }

    // Cleanup NvSIPL state.
    sipl_camera_.reset();
    sipl_query_.reset();
    camera_info_.clear();
    initialized_ = false;
    buffers_started_ = false;
}

void SIPLCaptureService::teardown_buffer_allocations() {
    for (auto& camera_state : per_camera_state_) {
        if (camera_state.sci_sync_isp0_) {
            NvSciSyncObjFree(camera_state.sci_sync_isp0_);
            camera_state.sci_sync_isp0_ = nullptr;
        }
        if (camera_state.cpu_wait_context_) {
            NvSciSyncCpuWaitContextFree(camera_state.cpu_wait_context_);
            camera_state.cpu_wait_context_ = nullptr;
        }
        free_buffers(camera_state.sci_bufs_icp_);
        free_buffers(camera_state.sci_bufs_isp0_);
    }
}

void SIPLCaptureService::forfeit_pending_outputs() {
    {
        std::lock_guard<std::mutex> lock(pending_outputs_mutex_);
        for (auto& [ptr, buffer] : pending_outputs_) {
            buffer->Release();
        }
        pending_outputs_.clear();
        pending_output_released_.notify_all();
    }
    std::lock_guard<std::mutex> lock(pending_release_targets_mutex_);
    for (auto it = pending_release_targets_.begin(); it != pending_release_targets_.end();) {
        auto service = it->second.lock();
        if (!service || service.get() == this) {
            it = pending_release_targets_.erase(it);
        } else {
            ++it;
        }
    }
}

void SIPLCaptureService::teardown_initialized_state() {
    teardown_buffer_allocations();
    per_camera_state_.clear();
    sipl_camera_.reset();
    sipl_query_.reset();
    camera_info_.clear();

    if (sci_sync_module_) {
        NvSciSyncModuleClose(sci_sync_module_);
        sci_sync_module_ = nullptr;
    }
    if (sci_buf_module_) {
        NvSciBufModuleClose(sci_buf_module_);
        sci_buf_module_ = nullptr;
    }
    initialized_ = false;
    buffers_started_ = false;
}

void SIPLCaptureService::ensure_streaming_started() {
    std::lock_guard<std::mutex> lock(streaming_mutex_);
    if (streaming_) {
        return;
    }

    HOLOSCAN_LOG_DEBUG("Starting streaming");

    // Create a thread to continuously acquire buffers from SIPL and offer them to the
    // compute() thread as needed. This is done because SIPL does not gracefully handle
    // a consumer that is slower than the camera frame rate and will lead to streaming
    // failures if it runs out of buffers. This means that the compute() method may miss
    // frames and the last known frame is always used.
    for (auto& camera_state : per_camera_state_) {
        camera_state.stop_thread_->store(false);
        if (camera_state.acquire_thread_.joinable()) {
            camera_state.acquire_thread_.join();
        }
        camera_state.acquire_thread_ = std::thread(
            &SIPLCaptureService::acquire_buffer_thread_func, this, &camera_state);
    }

    auto status = sipl_camera_->Start();
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        for (auto& camera_state : per_camera_state_) {
            camera_state.stop_thread_->store(true);
            if (camera_state.acquire_thread_.joinable()) {
                camera_state.acquire_thread_.join();
            }
        }
        throw std::runtime_error("Failed to start streaming");
    }
    streaming_ = true;
}

const std::string& SIPLCaptureService::output_name(std::uint32_t camera_index) const {
    return per_camera_state_.at(camera_index).output_name_;
}

nvsipl::INvSIPLISPStatCustomInterface* SIPLCaptureService::isp_stats(
    std::uint32_t camera_index) const {
  if (camera_index >= per_camera_state_.size()) {
    return nullptr;
  }
  return per_camera_state_[camera_index].isp_stats_;
}



void* SIPLCaptureService::map_buffer_to_cuda(NvSciBufObj buf_obj, std::uint64_t size) {
    std::lock_guard<std::mutex> lock(cuda_mappings_mutex_);
    if (cuda_mappings_.find(buf_obj) == cuda_mappings_.end()) {
        CudaBufferMapping mapping;

        cudaExternalMemoryHandleDesc ext_mem_handle_desc{};
        ext_mem_handle_desc.type = cudaExternalMemoryHandleTypeNvSciBuf;
        ext_mem_handle_desc.handle.nvSciBufObject = buf_obj;
        ext_mem_handle_desc.size = size;
        auto cuda_err = cudaImportExternalMemory(&mapping.mem_, &ext_mem_handle_desc);
        if (cuda_err != cudaSuccess) {
            throw std::runtime_error("Failed to import NvSciBuf into CUDA");
        }

        cudaExternalMemoryBufferDesc buffer_desc{};
        buffer_desc.offset = 0;
        buffer_desc.size = size;
        cuda_err = cudaExternalMemoryGetMappedBuffer(&mapping.ptr_, mapping.mem_, &buffer_desc);
        if (cuda_err != cudaSuccess) {
            cudaDestroyExternalMemory(mapping.mem_);
            throw std::runtime_error("Failed to map NvSciBuf into CUDA");
        }

        cuda_mappings_[buf_obj] = mapping;

        HOLOSCAN_LOG_DEBUG("Mapped buffer ({}) into CUDA (ptr={})",
            static_cast<void*>(buf_obj), mapping.ptr_);
    }
    return cuda_mappings_[buf_obj].ptr_;
}

SIPLCaptureService::AcquiredFrame SIPLCaptureService::acquire_frame(
    std::uint32_t camera_index, std::uint32_t timeout_us) {

    ensure_streaming_started();

    auto& camera_state = per_camera_state_.at(camera_index);

    std::unique_lock<std::mutex> buffer_state_lock(*camera_state.buffer_mutex_);
    while (camera_state.buffer_raw_ == nullptr) {
        HOLOSCAN_LOG_TRACE("Waiting for buffer for {} (timeout={}us)...",
            camera_state.output_name_, timeout_us);
        auto status = camera_state.buffer_available_->wait_for(buffer_state_lock,
            std::chrono::microseconds(timeout_us));
        if (status == std::cv_status::timeout) {
            throw std::runtime_error("Timed out waiting for frame from " +
                                     camera_state.output_name_);
        }
    }

    AcquiredFrame frame;
    frame.status = AcquireStatus::kOk;
    frame.buffer_raw = camera_state.buffer_raw_;
    frame.buffer_isp = camera_state.buffer_isp_;
    camera_state.buffer_raw_ = nullptr;
    camera_state.buffer_isp_ = nullptr;
    buffer_state_lock.unlock();

    frame.buffer = frame.buffer_isp ? frame.buffer_isp : frame.buffer_raw;

    struct AcquiredFrameCleanup {
        SIPLCaptureService* service;
        AcquiredFrame* frame;
        bool active = true;
        ~AcquiredFrameCleanup() {
            if (active && frame && frame->buffer) {
                service->release_acquired_frame(*frame);
            }
        }
        void dismiss() noexcept { active = false; }
    } frame_cleanup{ this, &frame };

    auto nvm_buffer = dynamic_cast<nvsipl::INvSIPLClient::INvSIPLNvMBuffer*>(frame.buffer);
    if (!nvm_buffer) {
        throw std::runtime_error("Failed to get INvSIPLNvMBuffer");
    }

    // Get and wait for the EOF fence (if there is one).
    NvSciSyncFence fence = NvSciSyncFenceInitializer;
    auto status = nvm_buffer->GetEOFNvSciSyncFence(&fence);
    if (status == nvsipl::NVSIPL_STATUS_OK) {
        auto err = NvSciSyncFenceWait(&fence, camera_state.cpu_wait_context_, timeout_us);
        if (err != NvSciError_Success) {
            NvSciSyncFenceClear(&fence);
            throw std::runtime_error("Failed to wait for EOF fence");
        }
    }
    NvSciSyncFenceClear(&fence);

    // Get the NvSci buffer object
    frame.buf_obj = nvm_buffer->GetNvSciBufImage();
    if (!frame.buf_obj) {
        throw std::runtime_error("Failed to get NvSciBufObj");
    }

    // Get the attributes of the buffer.
    NvSciBufAttrList buf_attr_list;
    auto err = NvSciBufObjGetAttrList(frame.buf_obj, &buf_attr_list);
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to get buffer attribute list");
    }

    // Get the specific attributes that are needed to determine how to
    // allocate and read pixels to an output buffer.
    NvSciBufAttrKeyValuePair img_attrs[] = {
        { NvSciBufImageAttrKey_Size,               NULL, 0 },
        { NvSciBufImageAttrKey_Layout,             NULL, 0 },
        { NvSciBufImageAttrKey_PlaneCount,         NULL, 0 },
        { NvSciBufImageAttrKey_PlanePitch,         NULL, 0 },
        { NvSciBufImageAttrKey_PlaneWidth,         NULL, 0 },
        { NvSciBufImageAttrKey_PlaneHeight,        NULL, 0 },
        { NvSciBufImageAttrKey_PlaneBitsPerPixel,  NULL, 0 },
        { NvSciBufImageAttrKey_PlaneAlignedHeight, NULL, 0 },
        { NvSciBufImageAttrKey_PlaneAlignedSize,   NULL, 0 },
        { NvSciBufImageAttrKey_PlaneChannelCount,  NULL, 0 },
        { NvSciBufImageAttrKey_PlaneOffset,        NULL, 0 },
        { NvSciBufImageAttrKey_PlaneColorFormat,   NULL, 0 },
        { NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency, NULL, 0 },
    };

    err = NvSciBufAttrListGetAttrs(buf_attr_list, img_attrs,
                                   sizeof(img_attrs) / sizeof(img_attrs[0]));
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to get buffer attributes");
    }

    frame.buffer_size       = *(static_cast<const std::uint64_t*>(img_attrs[0].value));
    frame.plane_count       = *(static_cast<const std::uint32_t*>(img_attrs[2].value));
    frame.plane_pitch       = static_cast<const std::uint32_t*>(img_attrs[3].value);
    frame.plane_width       = static_cast<const std::uint32_t*>(img_attrs[4].value);
    frame.plane_height      = static_cast<const std::uint32_t*>(img_attrs[5].value);
    frame.plane_offset      = static_cast<const std::uint64_t*>(img_attrs[10].value);
    frame.plane_color_format = static_cast<const NvSciBufAttrValColorFmt*>(img_attrs[11].value);

    // In RAW10 mode the operator reads via NvSciBufObjGetConstCpuPtr; no CUDA mapping needed.
    if (!raw_output_) {
        frame.cuda_ptr = map_buffer_to_cuda(frame.buf_obj, frame.buffer_size);
    }

    frame_cleanup.dismiss();
    return frame;
}

void SIPLCaptureService::release_raw_buffer_if_unused(
    nvsipl::INvSIPLClient::INvSIPLBuffer* buffer,
    nvsipl::INvSIPLClient::INvSIPLBuffer* buffer_raw) {
    if (buffer_raw && buffer != buffer_raw) {
        buffer_raw->Release();
    }
}

void SIPLCaptureService::release_acquired_frame(AcquiredFrame& frame) {
    if (!frame.buffer) {
        return;
    }
    release_raw_buffer_if_unused(frame.buffer, frame.buffer_raw);
    frame.buffer->Release();
    frame.buffer = nullptr;
    frame.buffer_raw = nullptr;
    frame.buffer_isp = nullptr;
}

void SIPLCaptureService::register_pending_output(
    void* cuda_ptr, nvsipl::INvSIPLClient::INvSIPLBuffer* buffer) {
    {
        std::lock_guard<std::mutex> lock(pending_outputs_mutex_);
        pending_outputs_[cuda_ptr] = buffer;
        HOLOSCAN_LOG_TRACE("Registered pending output buffer {} ({} total pending)",
            static_cast<void*>(buffer), pending_outputs_.size());
    }
    std::lock_guard<std::mutex> lock(pending_release_targets_mutex_);
    pending_release_targets_[cuda_ptr] = weak_from_this();
}

bool SIPLCaptureService::load_nito_file(std::string module_name, std::vector<std::uint8_t>& nito) {
    std::string module_name_lower;
    module_name_lower.reserve(module_name.size());
    for (auto c : module_name) {
        module_name_lower.push_back(static_cast<char>(std::tolower(c)));
    }

    std::string paths[] = {
        nito_base_path_ + "/" + module_name + ".nito",
        nito_base_path_ + "/" + module_name_lower + ".nito",
    };

    FILE* fp = nullptr;
    HOLOSCAN_LOG_DEBUG("Opening NITO file for module \"{}\"", module_name);
    for (const auto& path : paths) {
        fp = fopen(path.c_str(), "rb");
        if (fp) {
            HOLOSCAN_LOG_DEBUG("  Opened file: \"{}\"", path);
            break;
        }
        HOLOSCAN_LOG_DEBUG("  File not found: \"{}\"", path);
    }

    if (!fp) {
        HOLOSCAN_LOG_ERROR("Unable to open NITO file for module \"{}\"", module_name);
        return false;
    }

    // Check file size.
    fseek(fp, 0, SEEK_END);
    const auto fsize = static_cast<std::int64_t>(ftell(fp));
    rewind(fp);
    if (fsize <= 0) {
        HOLOSCAN_LOG_ERROR("NITO file for module \"{}\" is of invalid size", module_name);
        fclose(fp);
        return false;
    }

    // Read the file.
    nito.resize(static_cast<std::size_t>(fsize));
    const auto result = static_cast<std::int64_t>(fread(nito.data(), 1, static_cast<std::size_t>(fsize), fp));
    if (result != fsize) {
        HOLOSCAN_LOG_ERROR("Failed to read NITO file for module \"{}\"", module_name);
        nito.clear();
        fclose(fp);
        return false;
    }

    fclose(fp);

    HOLOSCAN_LOG_DEBUG("Loaded NITO file for module \"{}\" ({} bytes)", module_name, nito.size());

    return true;
}

void SIPLCaptureService::acquire_buffer_thread_func(PerCameraState* state) {
    HOLOSCAN_LOG_DEBUG("Starting acquire buffer thread for camera: {}", state->output_name_);

    while (!state->stop_thread_->load()) {
        // Use a reasonably short timeout here so that we don't block too
        // long when thread termination is requested.
        constexpr std::uint32_t timeout = 100'000;  // 100ms

        // Wait for a new RAW frame.
        nvsipl::INvSIPLClient::INvSIPLBuffer* buffer_raw = nullptr;
        auto status = state->queues_.captureCompletionQueue->Get(buffer_raw, timeout);
        if (status != nvsipl::NVSIPL_STATUS_OK) {
            if (status == nvsipl::NVSIPL_STATUS_TIMED_OUT) {
                // Timeout is expected, continue loop to check stop condition.
                HOLOSCAN_LOG_TRACE("Timeout getting RAW buffer for {}", state->output_name_);
                continue;
            }
            HOLOSCAN_LOG_ERROR("Failed to get RAW buffer for {} (status={})",
                state->output_name_, static_cast<int>(status));
            continue;
        }

        // Wait for the corresponding ISP frame.
        nvsipl::INvSIPLClient::INvSIPLBuffer* buffer_isp = nullptr;
        if (!raw_output_) {
            status = state->queues_.isp0CompletionQueue->Get(buffer_isp, timeout);
            if (status != nvsipl::NVSIPL_STATUS_OK) {
                if (buffer_raw) {
                    buffer_raw->Release();
                    buffer_raw = nullptr;
                }
                if (status == nvsipl::NVSIPL_STATUS_TIMED_OUT) {
                    // Timeout is expected, continue loop to check stop condition.
                    HOLOSCAN_LOG_WARN("Timeout getting ISP buffer for {}", state->output_name_);
                    continue;
                }
                HOLOSCAN_LOG_ERROR("Failed to get ISP buffer for {} (status={})",
                    state->output_name_, static_cast<int>(status));
                continue;
            }
        }

        // Notify compute thread of the new buffer.
        {
            std::lock_guard<std::mutex> lock(*state->buffer_mutex_);
            // Release any unused buffers.
            if (state->buffer_raw_) {
                state->buffer_raw_->Release();
            }
            if (state->buffer_isp_) {
                state->buffer_isp_->Release();
            }
            state->buffer_raw_ = buffer_raw;
            state->buffer_isp_ = buffer_isp;
            state->buffer_available_->notify_all();
        }
        // Wake the operator's scheduler via the registered notification sender.
        {
            std::lock_guard<std::mutex> cb_lock(*state->frame_ready_mutex_);
            if (state->frame_ready_callback_) {
                state->frame_ready_callback_();
            }
        }
    }

    // Release any unused buffers.
    std::lock_guard<std::mutex> lock(*state->buffer_mutex_);
    if (state->buffer_raw_) {
        state->buffer_raw_->Release();
        state->buffer_raw_ = nullptr;
    }
    if (state->buffer_isp_) {
        state->buffer_isp_->Release();
        state->buffer_isp_ = nullptr;
    }

    HOLOSCAN_LOG_DEBUG("Stopped acquire buffer thread for camera: {}", state->output_name_);
}

}  // namespace holoscan::holoscan_camera
