// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Port of hololink/src/hololink/operators/sipl_capture/sipl_compat.hpp
// Namespace changed to holoscan::holoscan_camera::sipl_compat.
// No functional changes to the compatibility logic.

#pragma once

#include <stdexcept>
#include <string>

#include <NvSIPLCamera.hpp>
#include <NvSIPLCameraQuery.hpp>

// Convenience macro for version checks in #if directives.
// Use (NVSIPL_API_MAJOR_VERSION + 0) so an empty -DNVSIPL_API_MAJOR_VERSION= from
// the build does not produce "#if  >= 2" (invalid); undefined names become 0 in #if.
#if defined(NVSIPL_API_MAJOR_VERSION) && (NVSIPL_API_MAJOR_VERSION + 0 >= 2)
#define SIPL_V2 1
#else
#define SIPL_V2 0
#endif

namespace holoscan::holoscan_camera::sipl_compat {

// ============================================================
// Type aliases
// ============================================================

#if SIPL_V2

using SystemConfig = nvsipl::sensorconfig::SensorSystemConfig;
using ModuleConfig = nvsipl::sensorconfig::ModuleConfig;
using ModuleTypeVar = nvsipl::sensorconfig::ModuleType;
using CoEModuleType = nvsipl::sensorconfig::CoEModule;
using CoESensorConfig = nvsipl::sensorconfig::CoECameraSensorConfig;
using CoETransport = nvsipl::sensorconfig::CoETransportConfig;
using TransportConfig = nvsipl::sensorconfig::TransportConfig;
using MacAddress = nvsipl::sensorconfig::MacAddress;
using VCInfo = nvsipl::sensorconfig::CommonSensorConfig::VirtualChannelInfo;

#else  // v1

using SystemConfig = nvsipl::CameraSystemConfig;
using ModuleConfig = nvsipl::CameraConfig;
using ModuleTypeVar = nvsipl::CameraType;
using CoEModuleType = nvsipl::CoECamera;
using CoESensorConfig = nvsipl::CoESensor;
using CoETransport = nvsipl::CoETransSettings;
using TransportConfig = nvsipl::TransportConfig;
using MacAddress = nvsipl::MacAddress;
using VCInfo = nvsipl::SensorInfo::VirtualChannelInfo;

#endif

// ============================================================
// Internal helpers (v2 only)
// ============================================================

#if SIPL_V2

inline const VCInfo& get_first_vcInfo(const nvsipl::sensorconfig::CoECameraSensorConfig& sensor_config) {
    if (sensor_config.vcInfoList.empty()) {
        throw std::runtime_error("No virtual channels found in sensor");
    }
    return sensor_config.vcInfoList[0];
}

inline const nvsipl::sensorconfig::CoECameraSensorConfig& get_first_coe_sensor(const ModuleConfig& mod) {
    const auto& coe = std::get<nvsipl::sensorconfig::CoEModule>(mod.moduleType);
    if (coe.sensorConfigs.empty()) {
        throw std::runtime_error("No sensors found in CoE module");
    }
    return std::get<nvsipl::sensorconfig::CoECameraSensorConfig>(coe.sensorConfigs[0]);
}

#endif

// ============================================================
// SystemConfig accessors
// ============================================================

inline auto& get_modules(SystemConfig& config) {
#if SIPL_V2
    return config.modules;
#else
    return config.cameras;
#endif
}

inline const auto& get_modules(const SystemConfig& config) {
#if SIPL_V2
    return config.modules;
#else
    return config.cameras;
#endif
}

// ============================================================
// ModuleConfig accessors
// ============================================================

inline const std::string& get_module_name(const ModuleConfig& mod) {
    return mod.name;
}

inline bool is_coe_module(const ModuleConfig& mod) {
#if SIPL_V2
    return std::holds_alternative<nvsipl::sensorconfig::CoEModule>(mod.moduleType);
#else
    return std::holds_alternative<nvsipl::CoECamera>(mod.cameratype);
#endif
}

// ============================================================
// Sensor info accessors
// ============================================================

inline uint32_t get_sensor_id(const ModuleConfig& mod) {
#if SIPL_V2
    return get_first_coe_sensor(mod).id;
#else
    return mod.sensorInfo.id;
#endif
}

inline uint32_t get_resolution_width(const ModuleConfig& mod) {
#if SIPL_V2
    return get_first_vcInfo(get_first_coe_sensor(mod)).resolution.width;
#else
    return mod.sensorInfo.vcInfo.resolution.width;
#endif
}

inline uint32_t get_resolution_height(const ModuleConfig& mod) {
#if SIPL_V2
    return get_first_vcInfo(get_first_coe_sensor(mod)).resolution.height;
#else
    return mod.sensorInfo.vcInfo.resolution.height;
#endif
}

inline NvSiplCapInputFormatType get_input_format(const ModuleConfig& mod) {
#if SIPL_V2
    return get_first_vcInfo(get_first_coe_sensor(mod)).inputFormat;
#else
    return mod.sensorInfo.vcInfo.inputFormat;
#endif
}

inline uint32_t get_cfa(const ModuleConfig& mod) {
#if SIPL_V2
    return get_first_vcInfo(get_first_coe_sensor(mod)).cfa;
#else
    return mod.sensorInfo.vcInfo.cfa;
#endif
}

inline uint32_t get_embedded_top_lines(const ModuleConfig& mod) {
#if SIPL_V2
    return get_first_vcInfo(get_first_coe_sensor(mod)).embeddedTopLines;
#else
    return mod.sensorInfo.vcInfo.embeddedTopLines;
#endif
}

inline uint32_t get_embedded_bottom_lines(const ModuleConfig& mod) {
#if SIPL_V2
    return get_first_vcInfo(get_first_coe_sensor(mod)).embeddedBottomLines;
#else
    return mod.sensorInfo.vcInfo.embeddedBottomLines;
#endif
}

inline bool is_sensor_group_secondary(const ModuleConfig& mod) {
#if SIPL_V2
    if (!is_coe_module(mod)) {
        return false;
    }
    const auto& sensor = get_first_coe_sensor(mod);
    return (sensor.sensorGroup > 0U) && (sensor.deviceIndex != 0U);
#else
    (void)mod;
    return false;
#endif
}

// ============================================================
// Query API wrapper
// ============================================================

inline nvsipl::SIPLStatus get_system_config(
    nvsipl::INvSIPLCameraQuery& query,
    const std::string& name,
    SystemConfig& config) {
#if SIPL_V2
    return query.GetSensorSystemConfig(name, config);
#else
    return query.GetCameraSystemConfig(name, config);
#endif
}

}  // namespace holoscan::holoscan_camera::sipl_compat
