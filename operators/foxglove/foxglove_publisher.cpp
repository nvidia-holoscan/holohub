/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "foxglove_publisher.hpp"

#include <algorithm>
#include <any>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <foxglove/foxglove.hpp>
#include <fmt/format.h>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/metadata.hpp>

namespace holoscan::ops {
namespace {

constexpr const char* kDefaultTimestampMetadataKeys =
    "acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns";

foxglove::messages::Timestamp timestamp_from_ns(uint64_t timestamp_ns) {
  return foxglove::messages::Timestamp{
      static_cast<uint32_t>(timestamp_ns / 1'000'000'000ULL),
      static_cast<uint32_t>(timestamp_ns % 1'000'000'000ULL)};
}

void throw_on_foxglove_error(foxglove::FoxgloveError error, const std::string& action) {
  if (error != foxglove::FoxgloveError::Ok) {
    throw std::runtime_error(fmt::format("{} failed: {}", action, foxglove::strerror(error)));
  }
}

template <typename ChannelMap>
void close_channels(ChannelMap& channels, std::string_view channel_type) {
  for (auto& [topic, channel] : channels) {
    if constexpr (std::is_same_v<decltype(channel.close()), void>) {
      channel.close();
    } else {
      const auto error = channel.close();
      if (error != foxglove::FoxgloveError::Ok) {
        HOLOSCAN_LOG_WARN("Failed to close Foxglove {} channel '{}' cleanly: {}",
                          channel_type,
                          topic,
                          foxglove::strerror(error));
      }
    }
  }
}

void throw_on_cuda_error(cudaError_t error, const std::string& action) {
  if (error != cudaSuccess) {
    throw std::runtime_error(fmt::format("{} failed: {}", action, cudaGetErrorString(error)));
  }
}

void wait_for_copy(cudaStream_t stream) {
  cudaEvent_t event = nullptr;
  throw_on_cuda_error(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                      "cudaEventCreateWithFlags");
  try {
    throw_on_cuda_error(cudaEventRecord(event, stream), "cudaEventRecord");
    throw_on_cuda_error(cudaEventSynchronize(event), "cudaEventSynchronize");
  } catch (...) {
    cudaEventDestroy(event);
    throw;
  }
  throw_on_cuda_error(cudaEventDestroy(event), "cudaEventDestroy");
}

std::vector<std::byte> copy_host_or_device(const void* data,
                                           size_t size,
                                           bool device_memory,
                                           const std::shared_ptr<Allocator>& allocator,
                                           PinnedHostBufferPool* pinned_pool,
                                           cudaStream_t stream = cudaStreamDefault) {
  std::vector<std::byte> out(size);
  if (size == 0) {
    return out;
  }
  if (data == nullptr) {
    throw std::runtime_error("Cannot copy null tensor/video buffer pointer");
  }
  if (device_memory) {
    std::byte* pinned = nullptr;
    nvidia::byte* allocator_ptr = nullptr;
    if (allocator) {
      allocator_ptr = allocator->allocate(size, MemoryStorageType::kHost);
      pinned = reinterpret_cast<std::byte*>(allocator_ptr);
    }
    if (pinned == nullptr) {
      if (pinned_pool == nullptr) {
        throw std::runtime_error("Pinned host buffer pool is not available");
      }
      pinned = pinned_pool->acquire(size);
    }

    try {
      throw_on_cuda_error(cudaMemcpyAsync(pinned, data, size, cudaMemcpyDeviceToHost, stream),
                          "cudaMemcpyAsyncDeviceToHost");
      wait_for_copy(stream);
      std::memcpy(out.data(), pinned, size);
    } catch (...) {
      if (allocator && allocator_ptr != nullptr) {
        allocator->free(allocator_ptr);
      }
      throw;
    }
    if (allocator && allocator_ptr != nullptr) {
      allocator->free(allocator_ptr);
    }
  } else {
    std::memcpy(out.data(), data, size);
  }
  return out;
}

bool is_device_tensor(const Tensor& tensor) {
  const auto device = tensor.device();
  return device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
}

std::vector<std::byte> copy_tensor_bytes(const Tensor& tensor,
                                         const std::shared_ptr<Allocator>& allocator,
                                         PinnedHostBufferPool* pinned_pool,
                                         cudaStream_t stream = cudaStreamDefault) {
  return copy_host_or_device(
      tensor.data(), tensor.nbytes(), is_device_tensor(tensor), allocator, pinned_pool, stream);
}

template <typename T>
std::vector<float> numeric_bytes_to_float(const std::vector<std::byte>& bytes) {
  if (bytes.size() % sizeof(T) != 0) {
    throw std::runtime_error("Tensor byte size is not aligned to dtype size");
  }
  const auto count = bytes.size() / sizeof(T);
  std::vector<float> out(count);
  for (size_t i = 0; i < count; ++i) {
    T value{};
    std::memcpy(&value, bytes.data() + i * sizeof(T), sizeof(T));
    out[i] = static_cast<float>(value);
  }
  return out;
}

template <>
std::vector<float> numeric_bytes_to_float<__half>(const std::vector<std::byte>& bytes) {
  if (bytes.size() % sizeof(__half) != 0) {
    throw std::runtime_error("Tensor byte size is not aligned to float16 dtype size");
  }
  const auto count = bytes.size() / sizeof(__half);
  std::vector<float> out(count);
  for (size_t i = 0; i < count; ++i) {
    __half value{};
    std::memcpy(&value, bytes.data() + i * sizeof(__half), sizeof(__half));
    out[i] = __half2float(value);
  }
  return out;
}

std::vector<float> tensor_to_float_vector(const Tensor& tensor,
                                          const std::shared_ptr<Allocator>& allocator,
                                          PinnedHostBufferPool* pinned_pool,
                                          cudaStream_t stream = cudaStreamDefault) {
  const auto dtype = tensor.dtype();
  if (dtype.lanes != 1) {
    throw std::runtime_error("Only single-lane tensors are supported for detection adaptation");
  }

  const auto bytes = copy_tensor_bytes(tensor, allocator, pinned_pool, stream);
  if (dtype.code == kDLFloat && dtype.bits == 16) {
    return numeric_bytes_to_float<__half>(bytes);
  }
  if (dtype.code == kDLFloat && dtype.bits == 32) {
    return numeric_bytes_to_float<float>(bytes);
  }
  if (dtype.code == kDLFloat && dtype.bits == 64) {
    return numeric_bytes_to_float<double>(bytes);
  }
  if (dtype.code == kDLInt && dtype.bits == 8) {
    return numeric_bytes_to_float<int8_t>(bytes);
  }
  if (dtype.code == kDLInt && dtype.bits == 16) {
    return numeric_bytes_to_float<int16_t>(bytes);
  }
  if (dtype.code == kDLInt && dtype.bits == 32) {
    return numeric_bytes_to_float<int32_t>(bytes);
  }
  if (dtype.code == kDLInt && dtype.bits == 64) {
    return numeric_bytes_to_float<int64_t>(bytes);
  }
  if (dtype.code == kDLUInt && dtype.bits == 8) {
    return numeric_bytes_to_float<uint8_t>(bytes);
  }
  if (dtype.code == kDLUInt && dtype.bits == 16) {
    return numeric_bytes_to_float<uint16_t>(bytes);
  }
  if (dtype.code == kDLUInt && dtype.bits == 32) {
    return numeric_bytes_to_float<uint32_t>(bytes);
  }
  if (dtype.code == kDLUInt && dtype.bits == 64) {
    return numeric_bytes_to_float<uint64_t>(bytes);
  }
  throw std::runtime_error(fmt::format("Unsupported tensor dtype code={} bits={}",
                                       static_cast<int>(dtype.code),
                                       static_cast<int>(dtype.bits)));
}

const std::shared_ptr<Tensor>& tensor_from_map(const TensorMap& tensors,
                                               const std::string& name,
                                               const char* purpose) {
  if (!name.empty()) {
    const auto iter = tensors.find(name);
    if (iter != tensors.end() && iter->second) {
      return iter->second;
    }
    throw std::runtime_error(fmt::format("Tensor '{}' for {} was not found", name, purpose));
  }
  if (tensors.empty() || !tensors.begin()->second) {
    throw std::runtime_error(fmt::format("No tensors were received for {}", purpose));
  }
  return tensors.begin()->second;
}

std::shared_ptr<Tensor> optional_tensor_from_map(const TensorMap& tensors,
                                                 const std::string& name) {
  if (name.empty()) {
    return nullptr;
  }
  const auto iter = tensors.find(name);
  return iter == tensors.end() ? nullptr : iter->second;
}

std::shared_ptr<Tensor> tensor_from_entity(gxf::Entity& entity,
                                           const std::string& name,
                                           const char* purpose) {
  if (!name.empty()) {
    auto tensor = entity.get<Tensor>(name.c_str());
    if (tensor) {
      return tensor;
    }
    throw std::runtime_error(
        fmt::format("{} input entity has no tensor named '{}'", purpose, name));
  }

  auto tensor = entity.get<Tensor>();
  if (tensor) {
    return tensor;
  }
  throw std::runtime_error(fmt::format("{} input entity contains no Tensor", purpose));
}

std::string json_escape(std::string_view text) {
  std::string escaped;
  escaped.reserve(text.size());
  for (const unsigned char ch : text) {
    switch (ch) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (ch < 0x20) {
          escaped += fmt::format("\\u{:04x}", static_cast<unsigned int>(ch));
        } else {
          escaped.push_back(static_cast<char>(ch));
        }
        break;
    }
  }
  return escaped;
}

std::string path_response_json(const std::string& path) {
  return fmt::format(R"({{"path":"{}"}})", json_escape(path));
}

std::string trim_copy(std::string text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

std::vector<std::string> parse_label_map(const std::string& label_map) {
  std::vector<std::string> labels;
  std::stringstream stream(label_map);
  std::string item;
  while (std::getline(stream, item, ',')) {
    labels.push_back(trim_copy(item));
  }
  return labels;
}

std::vector<std::string> parse_metadata_keys(const std::string& metadata_keys) {
  auto keys = parse_label_map(metadata_keys);
  if (keys.empty()) {
    keys = parse_label_map(kDefaultTimestampMetadataKeys);
  }
  return keys;
}

std::unordered_set<std::string> parse_name_set(const std::string& names) {
  auto items = parse_label_map(names);
  return {items.begin(), items.end()};
}

bool parameter_requested(const std::string& name, const std::vector<std::string_view>& requested) {
  if (requested.empty()) {
    return true;
  }
  return std::find(requested.begin(), requested.end(), std::string_view{name}) != requested.end();
}

template <typename T>
std::optional<T> holoscan_parameter_value(ParameterWrapper& wrapper) {
  auto* param = static_cast<Parameter<T>*>(wrapper.storage_ptr());
  if (param == nullptr) {
    return std::nullopt;
  }
  if (param->try_get().has_value()) {
    return param->try_get().value();
  }
  if (param->has_default_value()) {
    return param->default_value();
  }
  return std::nullopt;
}

template <typename T>
std::optional<foxglove::Parameter> integer_parameter(const std::string& name,
                                                     ParameterWrapper& wrapper) {
  if (const auto value = holoscan_parameter_value<T>(wrapper)) {
    return foxglove::Parameter(name, static_cast<int64_t>(value.value()));
  }
  return std::nullopt;
}

std::optional<foxglove::Parameter> foxglove_parameter_from_wrapper(
    const std::string& name,
    ParameterWrapper& wrapper) {
  if (wrapper.storage_ptr() == nullptr) {
    return std::nullopt;
  }
  const auto& type = wrapper.type();
  if (type == typeid(bool)) {
    if (const auto value = holoscan_parameter_value<bool>(wrapper)) {
      return foxglove::Parameter(name, value.value());
    }
  } else if (type == typeid(std::string)) {
    if (const auto value = holoscan_parameter_value<std::string>(wrapper)) {
      return foxglove::Parameter(name, value.value());
    }
  } else if (type == typeid(float)) {
    if (const auto value = holoscan_parameter_value<float>(wrapper)) {
      return foxglove::Parameter(name, static_cast<double>(value.value()));
    }
  } else if (type == typeid(double)) {
    if (const auto value = holoscan_parameter_value<double>(wrapper)) {
      return foxglove::Parameter(name, value.value());
    }
  } else if (type == typeid(int8_t)) {
    return integer_parameter<int8_t>(name, wrapper);
  } else if (type == typeid(uint8_t)) {
    return integer_parameter<uint8_t>(name, wrapper);
  } else if (type == typeid(int16_t)) {
    return integer_parameter<int16_t>(name, wrapper);
  } else if (type == typeid(uint16_t)) {
    return integer_parameter<uint16_t>(name, wrapper);
  } else if (type == typeid(int32_t)) {
    return integer_parameter<int32_t>(name, wrapper);
  } else if (type == typeid(uint32_t)) {
    return integer_parameter<uint32_t>(name, wrapper);
  } else if (type == typeid(int64_t)) {
    return integer_parameter<int64_t>(name, wrapper);
  } else if (type == typeid(uint64_t)) {
    if (const auto value = holoscan_parameter_value<uint64_t>(wrapper)) {
      if (value.value() <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return foxglove::Parameter(name, static_cast<int64_t>(value.value()));
      }
    }
  }
  return std::nullopt;
}

template <typename Value>
std::optional<double> parameter_value_as_double(const Value& value) {
  if (const auto* double_value = std::get_if<double>(&value)) {
    return *double_value;
  }
  if (const auto* int_value = std::get_if<int64_t>(&value)) {
    return static_cast<double>(*int_value);
  }
  return std::nullopt;
}

template <typename Value>
std::optional<int64_t> parameter_value_as_int64(const Value& value) {
  if (const auto* int_value = std::get_if<int64_t>(&value)) {
    return *int_value;
  }
  return std::nullopt;
}

template <typename T>
bool set_integer_parameter(ParameterWrapper& wrapper, int64_t value) {
  if constexpr (std::is_unsigned_v<T>) {
    if (value < 0 ||
        static_cast<uint64_t>(value) > static_cast<uint64_t>(std::numeric_limits<T>::max())) {
      return false;
    }
  } else if (value < static_cast<int64_t>(std::numeric_limits<T>::min()) ||
             value > static_cast<int64_t>(std::numeric_limits<T>::max())) {
    return false;
  }
  auto* param = static_cast<Parameter<T>*>(wrapper.storage_ptr());
  if (param == nullptr) {
    return false;
  }
  param->try_get() = static_cast<T>(value);
  return true;
}

template <typename Value>
bool set_holoscan_parameter_from_value(ParameterWrapper& wrapper, const Value& value) {
  if (wrapper.storage_ptr() == nullptr || std::holds_alternative<std::monostate>(value)) {
    return false;
  }
  const auto& type = wrapper.type();
  if (type == typeid(bool)) {
    if (const auto* bool_value = std::get_if<bool>(&value)) {
      static_cast<Parameter<bool>*>(wrapper.storage_ptr())->try_get() = *bool_value;
      return true;
    }
  }
  if (type == typeid(std::string)) {
    if (const auto* string_value = std::get_if<std::string>(&value)) {
      static_cast<Parameter<std::string>*>(wrapper.storage_ptr())->try_get() = *string_value;
      return true;
    }
  }
  if (type == typeid(float)) {
    if (const auto double_value = parameter_value_as_double(value)) {
      static_cast<Parameter<float>*>(wrapper.storage_ptr())->try_get() =
          static_cast<float>(double_value.value());
      return true;
    }
  }
  if (type == typeid(double)) {
    if (const auto double_value = parameter_value_as_double(value)) {
      static_cast<Parameter<double>*>(wrapper.storage_ptr())->try_get() = double_value.value();
      return true;
    }
  }
  if (const auto int_value = parameter_value_as_int64(value)) {
    if (type == typeid(int8_t)) {
      return set_integer_parameter<int8_t>(wrapper, int_value.value());
    }
    if (type == typeid(uint8_t)) {
      return set_integer_parameter<uint8_t>(wrapper, int_value.value());
    }
    if (type == typeid(int16_t)) {
      return set_integer_parameter<int16_t>(wrapper, int_value.value());
    }
    if (type == typeid(uint16_t)) {
      return set_integer_parameter<uint16_t>(wrapper, int_value.value());
    }
    if (type == typeid(int32_t)) {
      return set_integer_parameter<int32_t>(wrapper, int_value.value());
    }
    if (type == typeid(uint32_t)) {
      return set_integer_parameter<uint32_t>(wrapper, int_value.value());
    }
    if (type == typeid(int64_t)) {
      return set_integer_parameter<int64_t>(wrapper, int_value.value());
    }
    if (type == typeid(uint64_t)) {
      return set_integer_parameter<uint64_t>(wrapper, int_value.value());
    }
  }
  return false;
}

template <typename T>
std::optional<uint64_t> any_to_uint64(const std::any& value) {
  try {
    const auto converted = std::any_cast<T>(value);
    if constexpr (std::is_signed_v<T>) {
      if (converted < 0) {
        return std::nullopt;
      }
    }
    return static_cast<uint64_t>(converted);
  } catch (const std::bad_any_cast&) {
    return std::nullopt;
  }
}

std::optional<uint64_t> metadata_uint64(const MetadataDictionary& metadata,
                                        const std::string& key) {
  if (!metadata.has_key(key)) {
    return std::nullopt;
  }
  const auto object = metadata.get(key);
  if (!object) {
    return std::nullopt;
  }
  const auto value = object->value();
  if (auto converted = any_to_uint64<uint64_t>(value)) {
    return converted;
  }
  if (auto converted = any_to_uint64<int64_t>(value)) {
    return converted;
  }
  if (auto converted = any_to_uint64<uint32_t>(value)) {
    return converted;
  }
  if (auto converted = any_to_uint64<int32_t>(value)) {
    return converted;
  }
  return std::nullopt;
}

std::optional<std::string> metadata_string(const MetadataDictionary& metadata,
                                           const std::string& key) {
  if (!metadata.has_key(key)) {
    return std::nullopt;
  }
  const auto object = metadata.get(key);
  if (!object) {
    return std::nullopt;
  }
  const auto value = object->value();
  try {
    return std::any_cast<std::string>(value);
  } catch (const std::bad_any_cast&) {
  }
  if (auto converted = any_to_uint64<uint64_t>(value)) {
    return fmt::format("{}", converted.value());
  }
  if (auto converted = any_to_uint64<int64_t>(value)) {
    return fmt::format("{}", converted.value());
  }
  if (auto converted = any_to_uint64<uint32_t>(value)) {
    return fmt::format("{}", converted.value());
  }
  if (auto converted = any_to_uint64<int32_t>(value)) {
    return fmt::format("{}", converted.value());
  }
  try {
    return fmt::format("{:.6f}", std::any_cast<double>(value));
  } catch (const std::bad_any_cast&) {
  }
  try {
    return std::any_cast<bool>(value) ? "true" : "false";
  } catch (const std::bad_any_cast&) {
  }
  return std::nullopt;
}

uint64_t timestamp_from_input_metadata(Operator& op,
                                       InputContext& op_input,
                                       const char* input_port,
                                       const std::string& timestamp_metadata_keys) {
  if (op.is_metadata_enabled()) {
    const auto meta = op.metadata();
    if (meta) {
      for (const auto& key : parse_metadata_keys(timestamp_metadata_keys)) {
        if (const auto timestamp = metadata_uint64(*meta, key)) {
          if (timestamp.value() > 0) {
            return timestamp.value();
          }
        }
      }
    }
  }

  if (const auto acquisition_timestamp = op_input.get_acquisition_timestamp(input_port)) {
    if (acquisition_timestamp.value() > 0) {
      return static_cast<uint64_t>(acquisition_timestamp.value());
    }
  }

  return now_epoch_ns();
}

uint64_t timestamp_from_input_metadata_at(Operator& op,
                                          InputContext& op_input,
                                          const char* input_port,
                                          const std::string& timestamp_metadata_keys,
                                          size_t index) {
  if (op.is_metadata_enabled()) {
    const auto meta = op.metadata();
    if (meta) {
      for (const auto& key : parse_metadata_keys(timestamp_metadata_keys)) {
        if (const auto timestamp = metadata_uint64(*meta, key)) {
          if (timestamp.value() > 0) {
            return timestamp.value();
          }
        }
      }
    }
  }

  const auto timestamps = op_input.get_acquisition_timestamps(input_port);
  if (index < timestamps.size() && timestamps[index].has_value() && timestamps[index].value() > 0) {
    return static_cast<uint64_t>(timestamps[index].value());
  }
  return timestamp_from_input_metadata(op, op_input, input_port, timestamp_metadata_keys);
}

std::vector<FoxgloveKeyValue> metadata_key_values(Operator& op,
                                                  uint64_t timestamp_ns,
                                                  const std::string& topic = "/metadata") {
  std::vector<FoxgloveKeyValue> values;
  if (!op.is_metadata_enabled()) {
    return values;
  }
  const auto meta = op.metadata();
  if (!meta) {
    return values;
  }
  for (const auto& key : {"frame_index", "sequence_id"}) {
    if (const auto value = metadata_string(*meta, key)) {
      FoxgloveKeyValue key_value;
      key_value.topic = topic;
      key_value.key = key;
      key_value.value = value.value();
      key_value.timestamp_ns = timestamp_ns;
      values.push_back(std::move(key_value));
    }
  }
  return values;
}

std::array<double, 4> quaternion_from_rotation_matrix(const std::array<double, 9>& matrix) {
  const double trace = matrix[0] + matrix[4] + matrix[8];
  std::array<double, 4> quaternion{};
  if (trace > 0.0) {
    const double scale = std::sqrt(trace + 1.0) * 2.0;
    quaternion[3] = 0.25 * scale;
    quaternion[0] = (matrix[7] - matrix[5]) / scale;
    quaternion[1] = (matrix[2] - matrix[6]) / scale;
    quaternion[2] = (matrix[3] - matrix[1]) / scale;
  } else if (matrix[0] > matrix[4] && matrix[0] > matrix[8]) {
    const double scale = std::sqrt(1.0 + matrix[0] - matrix[4] - matrix[8]) * 2.0;
    quaternion[3] = (matrix[7] - matrix[5]) / scale;
    quaternion[0] = 0.25 * scale;
    quaternion[1] = (matrix[1] + matrix[3]) / scale;
    quaternion[2] = (matrix[2] + matrix[6]) / scale;
  } else if (matrix[4] > matrix[8]) {
    const double scale = std::sqrt(1.0 + matrix[4] - matrix[0] - matrix[8]) * 2.0;
    quaternion[3] = (matrix[2] - matrix[6]) / scale;
    quaternion[0] = (matrix[1] + matrix[3]) / scale;
    quaternion[1] = 0.25 * scale;
    quaternion[2] = (matrix[5] + matrix[7]) / scale;
  } else {
    const double scale = std::sqrt(1.0 + matrix[8] - matrix[0] - matrix[4]) * 2.0;
    quaternion[3] = (matrix[3] - matrix[1]) / scale;
    quaternion[0] = (matrix[2] + matrix[6]) / scale;
    quaternion[1] = (matrix[5] + matrix[7]) / scale;
    quaternion[2] = 0.25 * scale;
  }
  return quaternion;
}

FoxgloveFrameTransform transform_from_values(const std::vector<float>& values,
                                             const std::string& format) {
  FoxgloveFrameTransform transform;
  if (format == "matrix4x4") {
    if (values.size() < 16) {
      throw std::runtime_error("matrix4x4 pose tensor must contain at least 16 values");
    }
    transform.translation = {values[3], values[7], values[11]};
    const std::array<double, 9> rotation_matrix{values[0],
                                                values[1],
                                                values[2],
                                                values[4],
                                                values[5],
                                                values[6],
                                                values[8],
                                                values[9],
                                                values[10]};
    transform.rotation = quaternion_from_rotation_matrix(rotation_matrix);
    return transform;
  }
  if (format == "xyz_quat" || format == "xyz_xyzw") {
    if (values.size() < 7) {
      throw std::runtime_error("xyz_quat pose tensor must contain at least 7 values");
    }
    transform.translation = {values[0], values[1], values[2]};
    transform.rotation = {values[3], values[4], values[5], values[6]};
    return transform;
  }
  throw std::runtime_error(
      fmt::format("Unsupported pose format '{}'; expected matrix4x4 or xyz_quat", format));
}

std::string label_for_id(int64_t class_id, const std::vector<std::string>& labels) {
  if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size() &&
      !labels[static_cast<size_t>(class_id)].empty()) {
    return labels[static_cast<size_t>(class_id)];
  }
  return fmt::format("class_{}", class_id);
}

double scale_coordinate(double value, uint32_t extent, bool normalized) {
  return normalized ? value * static_cast<double>(extent) : value;
}

FoxgloveBox2D make_detection_box(double a,
                                  double b,
                                  double c,
                                  double d,
                                  const std::string& box_format,
                                  bool normalized,
                                  uint32_t image_width,
                                  uint32_t image_height,
                                  bool clamp_to_image,
                                  const std::string& label,
                                  double confidence) {
  if (normalized && (image_width == 0 || image_height == 0)) {
    throw std::runtime_error(
        "image_width and image_height are required when normalized_coordinates is true");
  }

  const auto sx = [image_width, normalized](double value) {
    return scale_coordinate(value, image_width, normalized);
  };
  const auto sy = [image_height, normalized](double value) {
    return scale_coordinate(value, image_height, normalized);
  };

  double x = 0.0;
  double y = 0.0;
  double width = 0.0;
  double height = 0.0;
  if (box_format == "xyxy") {
    const double x1 = sx(a);
    const double y1 = sy(b);
    const double x2 = sx(c);
    const double y2 = sy(d);
    x = std::min(x1, x2);
    y = std::min(y1, y2);
    width = std::abs(x2 - x1);
    height = std::abs(y2 - y1);
  } else if (box_format == "xywh") {
    x = sx(a);
    y = sy(b);
    width = sx(c);
    height = sy(d);
  } else {
    throw std::runtime_error(
        fmt::format("Unsupported box_format '{}'; expected xyxy or xywh", box_format));
  }

  if (clamp_to_image && image_width > 0 && image_height > 0) {
    const double max_x = static_cast<double>(image_width);
    const double max_y = static_cast<double>(image_height);
    const double x2 = std::clamp(x + width, 0.0, max_x);
    const double y2 = std::clamp(y + height, 0.0, max_y);
    x = std::clamp(x, 0.0, max_x);
    y = std::clamp(y, 0.0, max_y);
    width = std::max(0.0, x2 - x);
    height = std::max(0.0, y2 - y);
  }

  FoxgloveBox2D box;
  box.x = x;
  box.y = y;
  box.width = width;
  box.height = height;
  box.label = label;
  box.confidence = confidence;
  return box;
}

std::pair<size_t, size_t> matrix_shape(const Tensor& tensor, size_t fallback_columns) {
  const auto shape = tensor.shape();
  if (shape.empty()) {
    return {0, 0};
  }
  if (shape.size() == 1) {
    return {1, static_cast<size_t>(shape[0])};
  }
  const auto columns = static_cast<size_t>(shape.back());
  const auto rows = static_cast<size_t>(tensor.size()) / std::max<size_t>(columns, 1);
  return {rows, columns == 0 ? fallback_columns : columns};
}

std::string gxf_video_encoding(nvidia::gxf::VideoFormat format) {
  switch (format) {
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB:
      return "rgb8";
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA:
      return "rgba8";
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRA:
      return "bgra8";
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY:
      return "mono8";
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY16:
      return "mono16";
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32F:
      return "32FC1";
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12:
    case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_ER:
      return "nv12";
    default:
      return "";
  }
}

std::string normalize_topic(std::string topic, const char* fallback) {
  if (topic.empty()) {
    topic = fallback;
  }
  if (!topic.empty() && topic.front() != '/') {
    topic.insert(topic.begin(), '/');
  }
  return topic;
}

foxglove::McapCompression mcap_compression_from_string(const std::string& compression) {
  if (compression == "zstd") {
    return foxglove::McapCompression::Zstd;
  }
  if (compression == "lz4") {
    return foxglove::McapCompression::Lz4;
  }
  if (compression == "none") {
    return foxglove::McapCompression::None;
  }
  throw std::runtime_error(
      fmt::format("Unsupported MCAP compression '{}'; expected zstd, lz4, or none", compression));
}

template <typename ChannelMap>
bool should_log_channel(const std::string& topic,
                        const ChannelMap& channels,
                        bool drop_when_unsubscribed,
                        bool has_mcap_sink) {
  if (!drop_when_unsubscribed || has_mcap_sink) {
    return true;
  }
  const auto iter = channels.find(topic);
  return iter != channels.end() && iter->second.hasSinks();
}

void log_resolved_topic_once(bool& logged,
                             const std::string& operator_name,
                             const std::string& topic) {
  if (logged) {
    return;
  }
  logged = true;
  HOLOSCAN_LOG_INFO("{} publishing Foxglove topic '{}'", operator_name, topic);
}

}  // namespace

PinnedHostBufferPool::~PinnedHostBufferPool() {
  clear();
}

size_t PinnedHostBufferPool::size_class(size_t size) {
  size_t capacity = 1;
  while (capacity < size) {
    capacity <<= 1;
  }
  return capacity;
}

std::byte* PinnedHostBufferPool::acquire(size_t size) {
  const auto capacity = size_class(size);
  for (auto& buffer : buffers_) {
    if (buffer.capacity >= capacity) {
      return buffer.data;
    }
  }

  void* ptr = nullptr;
  throw_on_cuda_error(cudaHostAlloc(&ptr, capacity, cudaHostAllocDefault), "cudaHostAlloc");
  buffers_.push_back(Buffer{static_cast<std::byte*>(ptr), capacity});
  return buffers_.back().data;
}

void PinnedHostBufferPool::clear() {
  for (auto& buffer : buffers_) {
    if (buffer.data != nullptr) {
      const auto status = cudaFreeHost(buffer.data);
      if (status != cudaSuccess) {
        HOLOSCAN_LOG_WARN("cudaFreeHost failed while releasing Foxglove pinned buffer: {}",
                          cudaGetErrorString(status));
      }
    }
  }
  buffers_.clear();
}

std::vector<std::byte> to_byte_vector(const uint8_t* data, size_t size) {
  std::vector<std::byte> out(size);
  if (size > 0) {
    std::memcpy(out.data(), data, size);
  }
  return out;
}

uint64_t now_epoch_ns() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

uint64_t resolve_timestamp_ns(uint64_t timestamp_ns) {
  return timestamp_ns == 0 ? now_epoch_ns() : timestamp_ns;
}

uint32_t infer_image_step(uint32_t width, const std::string& encoding) {
  auto typed_encoding_step = [width, &encoding](
                                 const std::string& prefix,
                                 uint32_t bytes_per_element) -> std::optional<uint32_t> {
    if (encoding.rfind(prefix, 0) != 0) {
      return std::nullopt;
    }
    const auto suffix = encoding.substr(prefix.size());
    if (suffix.empty()) {
      return std::nullopt;
    }
    try {
      const auto channels = static_cast<uint64_t>(std::stoul(suffix));
      const auto bytes = static_cast<uint64_t>(width) * bytes_per_element * channels;
      if (bytes > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Image row step exceeds Foxglove RawImage step range");
      }
      return static_cast<uint32_t>(bytes);
    } catch (const std::invalid_argument&) {
      return std::nullopt;
    } catch (const std::out_of_range&) {
      throw std::runtime_error("Image encoding channel count exceeds supported range");
    }
  };

  for (const auto& [prefix, bytes_per_element] :
       std::initializer_list<std::pair<std::string, uint32_t>>{
           {"8UC", 1}, {"8SC", 1}, {"16UC", 2}, {"16SC", 2}, {"32UC", 4},
           {"32SC", 4}, {"32FC", 4}, {"64FC", 8}}) {
    if (const auto step = typed_encoding_step(prefix, bytes_per_element)) {
      return step.value();
    }
  }
  if (encoding == "rgba8" || encoding == "bgra8") {
    return width * 4;
  }
  if (encoding == "rgb8" || encoding == "bgr8") {
    return width * 3;
  }
  if (encoding == "mono16") {
    return width * 2;
  }
  if (encoding == "rgba32f" || encoding == "bgra32f") {
    return width * 16;
  }
  if (encoding == "rgb32f" || encoding == "bgr32f") {
    return width * 12;
  }
  if (encoding == "mono32f") {
    return width * 4;
  }
  return width;
}

namespace {

struct TensorImageLayout {
  uint32_t height = 0;
  uint32_t width = 0;
  uint32_t channels = 1;
  size_t height_dim = 0;
};

uint32_t checked_tensor_dim(int64_t value, const std::string& name) {
  if (value <= 0 || value > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error(fmt::format("Tensor image {} dimension is out of range", name));
  }
  return static_cast<uint32_t>(value);
}

TensorImageLayout tensor_image_layout(const Tensor& tensor, const std::string& context) {
  const auto shape = tensor.shape();
  if (shape.size() < 2 || shape.size() > 4) {
    throw std::runtime_error(fmt::format("{} must have rank 2, 3, or 4", context));
  }

  TensorImageLayout layout;
  if (shape.size() == 4) {
    if (shape[0] != 1) {
      throw std::runtime_error(fmt::format("{} batched tensor input must have batch size 1",
                                           context));
    }
    layout.height = checked_tensor_dim(shape[1], "height");
    layout.width = checked_tensor_dim(shape[2], "width");
    layout.channels = checked_tensor_dim(shape[3], "channel");
    layout.height_dim = 1;
    return layout;
  }

  if (shape.size() == 3 && (shape[2] == 1 || shape[2] == 3 || shape[2] == 4)) {
    layout.height = checked_tensor_dim(shape[0], "height");
    layout.width = checked_tensor_dim(shape[1], "width");
    layout.channels = checked_tensor_dim(shape[2], "channel");
    layout.height_dim = 0;
    return layout;
  }

  layout.height = checked_tensor_dim(shape[shape.size() - 2], "height");
  layout.width = checked_tensor_dim(shape[shape.size() - 1], "width");
  layout.channels = 1;
  layout.height_dim = shape.size() - 2;
  return layout;
}

uint32_t tensor_dtype_bytes(const Tensor& tensor) {
  const auto dtype = tensor.dtype();
  if (dtype.lanes != 1 || dtype.bits % 8 != 0) {
    throw std::runtime_error("Tensor image dtype must use whole-byte scalar elements");
  }
  return dtype.bits / 8;
}

std::string tensor_image_encoding(const Tensor& tensor, uint32_t channels) {
  const auto dtype = tensor.dtype();
  if (dtype.lanes != 1) {
    throw std::runtime_error("Tensor image dtype lanes > 1 require explicit image_encoding");
  }
  if (dtype.code == kDLUInt && dtype.bits == 8) {
    if (channels == 1) {
      return "mono8";
    }
    if (channels == 3) {
      return "rgb8";
    }
    if (channels == 4) {
      return "rgba8";
    }
    return fmt::format("8UC{}", channels);
  }
  if (dtype.code == kDLUInt && dtype.bits == 16) {
    if (channels == 1) {
      return "mono16";
    }
    return fmt::format("16UC{}", channels);
  }
  if (dtype.code == kDLUInt && dtype.bits == 32) {
    return fmt::format("32UC{}", channels);
  }
  if (dtype.code == kDLInt && dtype.bits == 8) {
    return fmt::format("8SC{}", channels);
  }
  if (dtype.code == kDLInt && dtype.bits == 16) {
    return fmt::format("16SC{}", channels);
  }
  if (dtype.code == kDLInt && dtype.bits == 32) {
    return fmt::format("32SC{}", channels);
  }
  if (dtype.code == kDLFloat && dtype.bits == 32) {
    return fmt::format("32FC{}", channels);
  }
  if (dtype.code == kDLFloat && dtype.bits == 64) {
    return fmt::format("64FC{}", channels);
  }
  throw std::runtime_error("Tensor image dtype requires explicit image_encoding");
}

uint32_t tensor_image_step(const Tensor& tensor,
                           const TensorImageLayout& layout,
                           const std::string& encoding,
                           uint32_t width) {
  const auto strides = tensor.strides();
  if (layout.height_dim < strides.size() && strides[layout.height_dim] > 0) {
    const auto stride = strides[layout.height_dim];
    if (stride > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("Tensor image row stride exceeds Foxglove RawImage step range");
    }
    return static_cast<uint32_t>(stride);
  }
  const auto bytes = static_cast<uint64_t>(width) * layout.channels * tensor_dtype_bytes(tensor);
  if (bytes > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Tensor image row step exceeds Foxglove RawImage step range");
  }
  const auto inferred_from_dtype = static_cast<uint32_t>(bytes);
  const auto inferred_from_encoding = infer_image_step(width, encoding);
  return std::max(inferred_from_dtype, inferred_from_encoding);
}

uint32_t video_buffer_image_step(const nvidia::gxf::VideoBufferInfo& info,
                                 uint32_t width,
                                 const std::string& encoding) {
  if (!info.color_planes.empty() && info.color_planes[0].stride > 0) {
    const auto stride = info.color_planes[0].stride;
    if (stride > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("GXF VideoBuffer row stride exceeds Foxglove RawImage step range");
    }
    return static_cast<uint32_t>(stride);
  }
  return infer_image_step(width, encoding);
}

}  // namespace

void FoxglovePublisherOp::setup(OperatorSpec& spec) {
  spec.input<std::vector<std::shared_ptr<FoxgloveBatch>>>("messages", IOSpec::kAnySize);
  spec.input<gxf::Entity>("image", IOSpec::kAnySize);
  spec.input<TensorMap>("tensors", IOSpec::kAnySize);
  spec.input<std::shared_ptr<FoxgloveImageAnnotations>>("annotations", IOSpec::kAnySize);
  spec.input<std::shared_ptr<FoxglovePointCloud>>("point_cloud", IOSpec::kAnySize);
  spec.input<std::shared_ptr<FoxgloveKeyValue>>("state", IOSpec::kAnySize);

  spec.param(bind_address_,
             "bind_address",
             "Bind address",
             "Address for the Foxglove WebSocket server to bind",
             std::string("127.0.0.1"));
  spec.param(port_, "port", "Port", "Foxglove WebSocket server port", uint16_t{8765});
  spec.param(server_name_,
             "server_name",
             "Server name",
             "Name advertised to Foxglove clients",
             std::string("Holoscan Foxglove"));
  spec.param(publish_server_time_,
             "publish_server_time",
             "Publish server time",
             "Broadcast message time through Foxglove's Time capability",
             true);
  spec.param(drop_when_unsubscribed_,
             "drop_when_unsubscribed",
             "Drop when unsubscribed",
             "Skip channel log calls when no WebSocket client or MCAP sink is subscribed",
             true);
  spec.param(enable_mcap_,
             "enable_mcap",
             "Enable MCAP",
             "Record all published messages to an MCAP file",
             false);
  spec.param(mcap_path_,
             "mcap_path",
             "MCAP path",
             "Output MCAP path when enable_mcap is true",
             std::string("holoscan_foxglove.mcap"));
  spec.param(mcap_compression_,
             "mcap_compression",
             "MCAP compression",
             "MCAP chunk compression: zstd, lz4, or none",
             std::string("zstd"));
  spec.param(timestamp_metadata_keys_,
             "timestamp_metadata_keys",
             "Timestamp metadata keys",
             "Comma-separated metadata keys checked before falling back to acquisition timestamp "
             "or current time",
             std::string(kDefaultTimestampMetadataKeys));
  spec.param(mutable_parameters_,
             "mutable_parameters",
             "Mutable parameters",
             "Comma-separated Foxglove parameter names that may be updated from Studio",
             std::string(""));
  spec.param(image_topic_,
             "image_topic",
             "Image topic",
             "Foxglove topic for direct image and tensor inputs",
             std::string("/image"));
  spec.param(image_frame_id_,
             "image_frame_id",
             "Image frame ID",
             "Frame ID for direct image and tensor inputs",
             std::string("camera"));
  spec.param(image_tensor_name_,
             "image_tensor_name",
             "Image tensor name",
             "Tensor component name for direct image and tensor inputs; empty selects the first "
             "tensor",
             std::string(""));
  spec.param(image_encoding_,
             "image_encoding",
             "Image encoding",
             "Foxglove RawImage encoding for direct image and tensor inputs; empty infers common "
             "formats",
             std::string(""));
  spec.param(image_width_, "image_width", "Image width", "Direct image width override", 0u);
  spec.param(image_height_, "image_height", "Image height", "Direct image height override", 0u);
  spec.param(image_step_, "image_step", "Image step", "Direct image row stride override", 0u);
  spec.param(image_prefer_video_buffer_,
             "image_prefer_video_buffer",
             "Image prefer VideoBuffer",
             "Use a GXF VideoBuffer before Tensor on the direct image port",
             true);
  spec.param(allocator_,
             "allocator",
             "Allocator",
             "Optional allocator used for pinned host staging buffers",
             std::shared_ptr<Allocator>{});
}

void FoxglovePublisherOp::open_mcap_writer(const std::string& path,
                                           const std::string& compression) {
  foxglove::McapWriterOptions writer_options;
  writer_options.context = context_;
  writer_options.path = path;
  writer_options.compression = mcap_compression_from_string(compression);
  writer_options.truncate = true;

  auto writer_result = foxglove::McapWriter::create(writer_options);
  if (!writer_result.has_value()) {
    throw std::runtime_error(
        fmt::format("Failed to create MCAP writer '{}': {}",
                    path,
                    foxglove::strerror(writer_result.error())));
  }
  mcap_writer_.emplace(std::move(writer_result.value()));
  HOLOSCAN_LOG_INFO("Recording Foxglove stream to {}", path);
}

std::string FoxglovePublisherOp::mcap_compression() const {
  std::lock_guard<std::mutex> lock(parameter_mutex_);
  return mcap_compression_.get();
}

std::string FoxglovePublisherOp::mcap_path() const {
  std::lock_guard<std::mutex> lock(parameter_mutex_);
  return mcap_path_.get();
}

void FoxglovePublisherOp::close_mcap_writer() {
  if (!mcap_writer_) {
    return;
  }
  const auto error = mcap_writer_->close();
  if (error != foxglove::FoxgloveError::Ok) {
    HOLOSCAN_LOG_WARN("Failed to close MCAP writer cleanly: {}", foxglove::strerror(error));
  }
  mcap_writer_.reset();
}

std::string FoxglovePublisherOp::snapshot_mcap_path(const std::string& base_path_string) const {
  static std::atomic<uint64_t> snapshot_counter{0};
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  const auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() %
      1000;
  const auto nonce = snapshot_counter.fetch_add(1, std::memory_order_relaxed);
  std::tm local_time{};
#if defined(_WIN32)
  localtime_s(&local_time, &time);
#else
  localtime_r(&time, &local_time);
#endif
  std::ostringstream suffix;
  suffix << std::put_time(&local_time, "%Y%m%d_%H%M%S") << '_' << std::setw(3)
         << std::setfill('0') << milliseconds << '_' << nonce;

  const std::filesystem::path base_path(base_path_string);
  auto stem = base_path.stem().string();
  if (stem.empty()) {
    stem = "holoscan_foxglove";
  }
  auto extension = base_path.extension().string();
  if (extension.empty()) {
    extension = ".mcap";
  }

  const auto snapshot_name = fmt::format("{}_{}{}", stem, suffix.str(), extension);
  return (base_path.parent_path() / snapshot_name).string();
}

std::vector<foxglove::Parameter> FoxglovePublisherOp::foxglove_parameters(
    const std::vector<std::string_view>& names) {
  std::lock_guard<std::mutex> lock(parameter_mutex_);
  std::vector<foxglove::Parameter> parameters;
  if (fragment() == nullptr) {
    return parameters;
  }

  for (const auto& op : fragment()->graph().get_nodes()) {
    if (!op || !op->spec()) {
      continue;
    }
    for (auto& [param_name, wrapper] : op->spec()->params()) {
      const auto name = fmt::format("{}.{}", op->name(), param_name);
      if (!parameter_requested(name, names)) {
        continue;
      }
      if (auto parameter = foxglove_parameter_from_wrapper(name, wrapper)) {
        parameters.push_back(std::move(parameter.value()));
      }
    }
  }
  return parameters;
}

std::vector<foxglove::Parameter> FoxglovePublisherOp::enqueue_foxglove_parameter_updates(
    const std::vector<foxglove::ParameterView>& params) {
  std::vector<PendingParameterUpdate> updates;
  {
    std::lock_guard<std::mutex> lock(parameter_mutex_);
    auto mutable_parameters = parse_name_set(mutable_parameters_.get());
    for (const auto& parameter : params) {
      const auto name = std::string(parameter.name());
      if (mutable_parameters.find(name) == mutable_parameters.end()) {
        HOLOSCAN_LOG_WARN("Ignoring read-only Foxglove parameter update for '{}'", name);
        continue;
      }
      if (!parameter.hasValue()) {
        HOLOSCAN_LOG_WARN("Ignoring empty Foxglove parameter update for '{}'", name);
        continue;
      }

      ParameterValue value;
      if (parameter.is<bool>()) {
        value = parameter.get<bool>();
      } else if (parameter.is<std::string>()) {
        value = parameter.get<std::string>();
      } else if (parameter.is<double>()) {
        value = parameter.get<double>();
      } else if (parameter.is<int64_t>()) {
        value = parameter.get<int64_t>();
      } else {
        HOLOSCAN_LOG_WARN("Ignoring unsupported Foxglove parameter update type for '{}'", name);
        continue;
      }
      updates.push_back({name, std::move(value)});
    }
    pending_parameter_updates_.insert(pending_parameter_updates_.end(),
                                      std::make_move_iterator(updates.begin()),
                                      std::make_move_iterator(updates.end()));
  }
  return foxglove_parameters();
}

void FoxglovePublisherOp::apply_pending_parameter_updates() {
  std::lock_guard<std::mutex> lock(parameter_mutex_);
  if (pending_parameter_updates_.empty()) {
    return;
  }
  auto updates = std::move(pending_parameter_updates_);
  pending_parameter_updates_.clear();

  for (const auto& update : updates) {
    const auto& name = update.name;
    const auto separator = name.find('.');
    if (separator == std::string::npos || fragment() == nullptr) {
      continue;
    }
    const auto op_name = name.substr(0, separator);
    const auto param_name = name.substr(separator + 1);
    auto op = fragment()->graph().find_node(op_name);
    if (!op || !op->spec()) {
      continue;
    }
    auto& params_map = op->spec()->params();
    auto iter = params_map.find(param_name);
    if (iter == params_map.end()) {
      continue;
    }
    if (!set_holoscan_parameter_from_value(iter->second, update.value)) {
      HOLOSCAN_LOG_WARN("Could not update Foxglove parameter '{}'", name);
    }
  }
}

bool FoxglovePublisherOp::drop_when_unsubscribed() const {
  std::lock_guard<std::mutex> lock(parameter_mutex_);
  return drop_when_unsubscribed_.get();
}

bool FoxglovePublisherOp::should_publish_raw_image(const std::string& topic) {
  const auto drop_when_unsubscribed_value = drop_when_unsubscribed();
  std::lock_guard<std::mutex> lock(mcap_mutex_);
  raw_image_channel(topic);
  return should_log_channel(topic,
                            raw_image_channels_,
                            drop_when_unsubscribed_value,
                            mcap_writer_.has_value());
}

void FoxglovePublisherOp::register_services() {
  if (!server_) {
    return;
  }
  service_handlers_.clear();
  service_handlers_.reserve(3);

  auto make_service = [this](const std::string& name, foxglove::ServiceHandler handler) {
    service_handlers_.push_back(std::move(handler));
    foxglove::ServiceSchema schema;
    schema.name = name;
    auto result = foxglove::Service::create(name, schema, service_handlers_.back());
    if (!result.has_value()) {
      throw std::runtime_error(fmt::format("Failed to create Foxglove service '{}': {}",
                                           name,
                                           foxglove::strerror(result.error())));
    }
    throw_on_foxglove_error(server_->addService(std::move(result.value())),
                            fmt::format("Add Foxglove service '{}'", name));
  };

  make_service("start_recording",
               [this]([[maybe_unused]] const foxglove::ServiceRequest& request,
                      foxglove::ServiceResponder&& responder) {
                 const auto path = mcap_path();
                 const auto compression = mcap_compression();
                 std::lock_guard<std::mutex> lock(mcap_mutex_);
                 try {
                   if (!mcap_writer_) {
                     open_mcap_writer(path, compression);
                   }
                   const auto response = path_response_json(path);
                   const auto* data =
                       reinterpret_cast<const std::byte*>(response.data());
                   std::move(responder).respondOk(data, response.size());
                 } catch (const std::exception& exc) {
                   std::move(responder).respondError(exc.what());
                 }
               });

  make_service("stop_recording",
               [this]([[maybe_unused]] const foxglove::ServiceRequest& request,
                      foxglove::ServiceResponder&& responder) {
                 std::lock_guard<std::mutex> lock(mcap_mutex_);
                 close_mcap_writer();
                 constexpr std::string_view response = R"({"recording":false})";
                 std::move(responder).respondOk(
                     reinterpret_cast<const std::byte*>(response.data()), response.size());
               });

  make_service("snapshot_mcap",
               [this]([[maybe_unused]] const foxglove::ServiceRequest& request,
                      foxglove::ServiceResponder&& responder) {
                 const auto path = snapshot_mcap_path(mcap_path());
                 const auto compression = mcap_compression();
                 std::lock_guard<std::mutex> lock(mcap_mutex_);
                 try {
                   close_mcap_writer();
                   open_mcap_writer(path, compression);
                   const auto response = path_response_json(path);
                   const auto* data =
                       reinterpret_cast<const std::byte*>(response.data());
                   std::move(responder).respondOk(data, response.size());
                 } catch (const std::exception& exc) {
                   std::move(responder).respondError(exc.what());
                 }
               });
}

void FoxglovePublisherOp::precreate_channels() {
  raw_image_channel(normalize_topic(image_topic_.get(), "/image"));
  compressed_video_channel("/video/compressed");
  calibration_channel("/camera/calibration");
  annotation_channel("/image/annotations");
  point_cloud_channel("/points");
  frame_transform_channel("/tf");
  key_value_channel("/metadata");
  key_value_channel("/state");
}

FoxgloveImage FoxglovePublisherOp::image_from_entity(gxf::Entity entity, cudaStream_t stream) {
  if (image_prefer_video_buffer_.get()) {
    auto maybe_video_buffer =
        static_cast<nvidia::gxf::Entity&>(entity).get<nvidia::gxf::VideoBuffer>();
    if (maybe_video_buffer) {
      const auto& buffer = *maybe_video_buffer.value().get();
      const auto info = buffer.video_frame_info();
      std::string encoding = image_encoding_.get();
      if (encoding.empty()) {
        encoding = gxf_video_encoding(info.color_format);
      }
      if (encoding.empty()) {
        throw std::runtime_error("Unsupported GXF VideoBuffer format; set image_encoding");
      }

      FoxgloveImage image;
      image.topic = normalize_topic(image_topic_.get(), "/image");
      image.frame_id = image_frame_id_.get();
      image.encoding = encoding;
      image.width = image_width_.get() == 0 ? info.width : image_width_.get();
      image.height = image_height_.get() == 0 ? info.height : image_height_.get();
      image.step = image_step_.get() == 0
                       ? video_buffer_image_step(info, image.width, image.encoding)
                       : image_step_.get();
      image.data = copy_host_or_device(buffer.pointer(),
                                       buffer.size(),
                                       buffer.storage_type() ==
                                           nvidia::gxf::MemoryStorageType::kDevice,
                                       allocator_.get(),
                                       &pinned_host_pool_,
                                       stream);
      return image;
    }
  }

  const auto tensor_name = image_tensor_name_.get();
  auto maybe_tensor = tensor_from_entity(entity, tensor_name, "Direct Foxglove image");
  TensorMap tensors;
  tensors.emplace(tensor_name, maybe_tensor);
  return image_from_tensor_map(tensors, stream);
}

FoxgloveImage FoxglovePublisherOp::image_from_tensor_map(const TensorMap& tensors,
                                                         cudaStream_t stream) {
  const auto& tensor = tensor_from_map(tensors, image_tensor_name_.get(), "direct image input");
  const auto layout = tensor_image_layout(*tensor, "Direct tensor image input");

  FoxgloveImage image;
  image.topic = normalize_topic(image_topic_.get(), "/image");
  image.frame_id = image_frame_id_.get();
  image.width = image_width_.get();
  image.height = image_height_.get();
  if (image.height == 0) {
    image.height = layout.height;
  }
  if (image.width == 0) {
    image.width = layout.width;
  }

  image.encoding = image_encoding_.get();
  if (image.encoding.empty()) {
    image.encoding = tensor_image_encoding(*tensor, layout.channels);
  }
  image.step = image_step_.get() == 0 ? tensor_image_step(*tensor, layout, image.encoding,
                                                          image.width)
                                      : image_step_.get();

  const auto device = tensor->device();
  const bool is_device = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
  image.data = copy_host_or_device(
      tensor->data(), tensor->nbytes(), is_device, allocator_.get(), &pinned_host_pool_, stream);
  return image;
}

void FoxglovePublisherOp::start() {
  foxglove::setLogLevel(foxglove::LogLevel::Warn);
  context_ = foxglove::Context::create();

  foxglove::WebSocketServerOptions options;
  options.context = context_;
  options.host = bind_address_.get();
  options.port = port_.get();
  options.name = server_name_.get();
  options.capabilities = foxglove::WebSocketServerCapabilities::Time |
                         foxglove::WebSocketServerCapabilities::Parameters |
                         foxglove::WebSocketServerCapabilities::Services;
  options.supported_encodings = {"protobuf", "json"};
  options.callbacks.onGetParameters =
      [this]([[maybe_unused]] uint32_t client_id,
             [[maybe_unused]] std::optional<std::string_view> request_id,
             const std::vector<std::string_view>& param_names) {
        try {
          return foxglove_parameters(param_names);
        } catch (const std::exception& error) {
          HOLOSCAN_LOG_ERROR("Failed to handle Foxglove parameter read request: {}",
                             error.what());
        } catch (...) {
          HOLOSCAN_LOG_ERROR("Failed to handle Foxglove parameter read request: unknown error");
        }
        return std::vector<foxglove::Parameter>{};
      };
  options.callbacks.onSetParameters =
      [this]([[maybe_unused]] uint32_t client_id,
             [[maybe_unused]] std::optional<std::string_view> request_id,
             const std::vector<foxglove::ParameterView>& params) {
        try {
          return enqueue_foxglove_parameter_updates(params);
        } catch (const std::exception& error) {
          HOLOSCAN_LOG_ERROR("Failed to handle Foxglove parameter update request: {}",
                             error.what());
        } catch (...) {
          HOLOSCAN_LOG_ERROR("Failed to handle Foxglove parameter update request: unknown error");
        }
        return std::vector<foxglove::Parameter>{};
      };

  auto server_result = foxglove::WebSocketServer::create(std::move(options));
  if (!server_result.has_value()) {
    throw std::runtime_error(
        fmt::format("Failed to start Foxglove WebSocket server: {}",
                    foxglove::strerror(server_result.error())));
  }
  server_.emplace(std::move(server_result.value()));
  register_services();
  precreate_channels();
  HOLOSCAN_LOG_INFO("Foxglove WebSocket server listening on {}:{}",
                    bind_address_.get(),
                    server_->port());

  if (enable_mcap_.get()) {
    const auto path = mcap_path();
    const auto compression = mcap_compression();
    std::lock_guard<std::mutex> lock(mcap_mutex_);
    open_mcap_writer(path, compression);
  }
}

void FoxglovePublisherOp::stop() {
  close_channels(raw_image_channels_, "RawImage");
  close_channels(compressed_video_channels_, "CompressedVideo");
  close_channels(calibration_channels_, "CameraCalibration");
  close_channels(annotation_channels_, "ImageAnnotations");
  close_channels(point_cloud_channels_, "PointCloud");
  close_channels(frame_transform_channels_, "FrameTransform");
  close_channels(key_value_channels_, "KeyValuePair");
  raw_image_channels_.clear();
  compressed_video_channels_.clear();
  calibration_channels_.clear();
  annotation_channels_.clear();
  point_cloud_channels_.clear();
  frame_transform_channels_.clear();
  key_value_channels_.clear();
  pinned_host_pool_.clear();

  {
    std::lock_guard<std::mutex> lock(mcap_mutex_);
    close_mcap_writer();
  }
  if (server_) {
    const auto error = server_->stop();
    if (error != foxglove::FoxgloveError::Ok) {
      HOLOSCAN_LOG_WARN("Failed to stop Foxglove WebSocket server cleanly: {}",
                        foxglove::strerror(error));
    }
    server_.reset();
  }
  service_handlers_.clear();
}

void FoxglovePublisherOp::compute(InputContext& op_input,
                                  [[maybe_unused]] OutputContext& op_output,
                                  [[maybe_unused]] ExecutionContext& context) {
  apply_pending_parameter_updates();
  uint64_t latest_timestamp = 0;

  auto maybe_batches = op_input.receive<std::vector<std::shared_ptr<FoxgloveBatch>>>("messages");
  if (maybe_batches) {
    for (const auto& batch_ptr : maybe_batches.value()) {
      if (!batch_ptr) {
        continue;
      }
      const auto& batch = *batch_ptr;

      for (const auto& image : batch.images) {
        latest_timestamp = std::max(latest_timestamp, publish_image(image));
      }
      for (const auto& video : batch.compressed_videos) {
        latest_timestamp = std::max(latest_timestamp, publish_compressed_video(video));
      }
      for (const auto& calibration : batch.calibrations) {
        latest_timestamp = std::max(latest_timestamp, publish_calibration(calibration));
      }
      for (const auto& annotations : batch.annotations) {
        latest_timestamp = std::max(latest_timestamp, publish_annotations(annotations));
      }
      for (const auto& point_cloud : batch.point_clouds) {
        latest_timestamp = std::max(latest_timestamp, publish_point_cloud(point_cloud));
      }
      for (const auto& transform : batch.frame_transforms) {
        latest_timestamp = std::max(latest_timestamp, publish_frame_transform(transform));
      }
      for (const auto& key_value : batch.key_values) {
        latest_timestamp = std::max(latest_timestamp, publish_key_value(key_value));
      }
    }
  }

  if (auto maybe_images = op_input.receive<std::vector<gxf::Entity>>("image")) {
    const auto topic = normalize_topic(image_topic_.get(), "/image");
    if (should_publish_raw_image(topic)) {
      const auto stream = op_input.receive_cuda_stream("image", false, true);
      size_t index = 0;
      for (auto& entity : maybe_images.value()) {
        const auto timestamp_ns = timestamp_from_input_metadata_at(
            *this, op_input, "image", timestamp_metadata_keys_.get(), index++);
        auto image = image_from_entity(std::move(entity), stream);
        image.timestamp_ns = timestamp_ns;
        latest_timestamp = std::max(latest_timestamp, publish_image(image));
        for (const auto& key_value : metadata_key_values(*this, timestamp_ns)) {
          latest_timestamp = std::max(latest_timestamp, publish_key_value(key_value));
        }
      }
    }
  }

  if (auto maybe_tensors = op_input.receive<std::vector<TensorMap>>("tensors")) {
    const auto topic = normalize_topic(image_topic_.get(), "/image");
    if (should_publish_raw_image(topic)) {
      const auto stream = op_input.receive_cuda_stream("tensors", false, true);
      size_t index = 0;
      for (const auto& tensors : maybe_tensors.value()) {
        const auto timestamp_ns = timestamp_from_input_metadata_at(
            *this, op_input, "tensors", timestamp_metadata_keys_.get(), index++);
        auto image = image_from_tensor_map(tensors, stream);
        image.timestamp_ns = timestamp_ns;
        latest_timestamp = std::max(latest_timestamp, publish_image(image));
        for (const auto& key_value : metadata_key_values(*this, timestamp_ns)) {
          latest_timestamp = std::max(latest_timestamp, publish_key_value(key_value));
        }
      }
    }
  }

  if (auto maybe_annotations =
          op_input.receive<std::vector<std::shared_ptr<FoxgloveImageAnnotations>>>("annotations")) {
    size_t index = 0;
    for (const auto& annotations : maybe_annotations.value()) {
      if (!annotations) {
        continue;
      }
      auto copy = *annotations;
      if (copy.timestamp_ns == 0) {
        copy.timestamp_ns = timestamp_from_input_metadata_at(
            *this, op_input, "annotations", timestamp_metadata_keys_.get(), index);
      }
      ++index;
      latest_timestamp = std::max(latest_timestamp, publish_annotations(copy));
    }
  }

  if (auto maybe_point_clouds =
          op_input.receive<std::vector<std::shared_ptr<FoxglovePointCloud>>>("point_cloud")) {
    size_t index = 0;
    for (const auto& point_cloud : maybe_point_clouds.value()) {
      if (!point_cloud) {
        continue;
      }
      auto copy = *point_cloud;
      if (copy.timestamp_ns == 0) {
        copy.timestamp_ns = timestamp_from_input_metadata_at(
            *this, op_input, "point_cloud", timestamp_metadata_keys_.get(), index);
      }
      ++index;
      latest_timestamp = std::max(latest_timestamp, publish_point_cloud(copy));
    }
  }

  if (auto maybe_state =
          op_input.receive<std::vector<std::shared_ptr<FoxgloveKeyValue>>>("state")) {
    size_t index = 0;
    for (const auto& key_value : maybe_state.value()) {
      if (!key_value) {
        continue;
      }
      auto copy = *key_value;
      if (copy.timestamp_ns == 0) {
        copy.timestamp_ns = timestamp_from_input_metadata_at(
            *this, op_input, "state", timestamp_metadata_keys_.get(), index);
      }
      ++index;
      latest_timestamp = std::max(latest_timestamp, publish_key_value(copy));
    }
  }

  if (publish_server_time_.get() && server_ && latest_timestamp != 0) {
    server_->broadcastTime(latest_timestamp);
  }
}

foxglove::messages::RawImageChannel& FoxglovePublisherOp::raw_image_channel(
    const std::string& topic) {
  auto iter = raw_image_channels_.find(topic);
  if (iter != raw_image_channels_.end()) {
    return iter->second;
  }
  auto result = foxglove::messages::RawImageChannel::create(topic, context_);
  if (!result.has_value()) {
    throw std::runtime_error(fmt::format("Failed to create RawImage channel '{}': {}",
                                         topic,
                                         foxglove::strerror(result.error())));
  }
  auto [created, _] = raw_image_channels_.emplace(topic, std::move(result.value()));
  return created->second;
}

foxglove::messages::CompressedVideoChannel& FoxglovePublisherOp::compressed_video_channel(
    const std::string& topic) {
  auto iter = compressed_video_channels_.find(topic);
  if (iter != compressed_video_channels_.end()) {
    return iter->second;
  }
  auto result = foxglove::messages::CompressedVideoChannel::create(topic, context_);
  if (!result.has_value()) {
    throw std::runtime_error(fmt::format("Failed to create CompressedVideo channel '{}': {}",
                                         topic,
                                         foxglove::strerror(result.error())));
  }
  auto [created, _] = compressed_video_channels_.emplace(topic, std::move(result.value()));
  return created->second;
}

foxglove::messages::CameraCalibrationChannel& FoxglovePublisherOp::calibration_channel(
    const std::string& topic) {
  auto iter = calibration_channels_.find(topic);
  if (iter != calibration_channels_.end()) {
    return iter->second;
  }
  auto result = foxglove::messages::CameraCalibrationChannel::create(topic, context_);
  if (!result.has_value()) {
    throw std::runtime_error(fmt::format("Failed to create CameraCalibration channel '{}': {}",
                                         topic,
                                         foxglove::strerror(result.error())));
  }
  auto [created, _] = calibration_channels_.emplace(topic, std::move(result.value()));
  return created->second;
}

foxglove::messages::ImageAnnotationsChannel& FoxglovePublisherOp::annotation_channel(
    const std::string& topic) {
  auto iter = annotation_channels_.find(topic);
  if (iter != annotation_channels_.end()) {
    return iter->second;
  }
  auto result = foxglove::messages::ImageAnnotationsChannel::create(topic, context_);
  if (!result.has_value()) {
    throw std::runtime_error(fmt::format("Failed to create ImageAnnotations channel '{}': {}",
                                         topic,
                                         foxglove::strerror(result.error())));
  }
  auto [created, _] = annotation_channels_.emplace(topic, std::move(result.value()));
  return created->second;
}

foxglove::messages::PointCloudChannel& FoxglovePublisherOp::point_cloud_channel(
    const std::string& topic) {
  auto iter = point_cloud_channels_.find(topic);
  if (iter != point_cloud_channels_.end()) {
    return iter->second;
  }
  auto result = foxglove::messages::PointCloudChannel::create(topic, context_);
  if (!result.has_value()) {
    throw std::runtime_error(fmt::format("Failed to create PointCloud channel '{}': {}",
                                         topic,
                                         foxglove::strerror(result.error())));
  }
  auto [created, _] = point_cloud_channels_.emplace(topic, std::move(result.value()));
  return created->second;
}

foxglove::messages::FrameTransformChannel& FoxglovePublisherOp::frame_transform_channel(
    const std::string& topic) {
  auto iter = frame_transform_channels_.find(topic);
  if (iter != frame_transform_channels_.end()) {
    return iter->second;
  }
  auto result = foxglove::messages::FrameTransformChannel::create(topic, context_);
  if (!result.has_value()) {
    throw std::runtime_error(fmt::format("Failed to create FrameTransform channel '{}': {}",
                                         topic,
                                         foxglove::strerror(result.error())));
  }
  auto [created, _] = frame_transform_channels_.emplace(topic, std::move(result.value()));
  return created->second;
}

foxglove::messages::KeyValuePairChannel& FoxglovePublisherOp::key_value_channel(
    const std::string& topic) {
  auto iter = key_value_channels_.find(topic);
  if (iter != key_value_channels_.end()) {
    return iter->second;
  }
  auto result = foxglove::messages::KeyValuePairChannel::create(topic, context_);
  if (!result.has_value()) {
    throw std::runtime_error(fmt::format("Failed to create KeyValuePair channel '{}': {}",
                                         topic,
                                         foxglove::strerror(result.error())));
  }
  auto [created, _] = key_value_channels_.emplace(topic, std::move(result.value()));
  return created->second;
}

uint64_t FoxglovePublisherOp::publish_image(const FoxgloveImage& image) {
  const auto log_time_ns = resolve_timestamp_ns(image.timestamp_ns);
  auto topic = normalize_topic(image.topic, "/image");
  const auto drop_when_unsubscribed_value = drop_when_unsubscribed();
  foxglove::messages::RawImage message;
  message.timestamp = timestamp_from_ns(log_time_ns);
  message.frame_id = image.frame_id;
  message.width = image.width;
  message.height = image.height;
  message.encoding = image.encoding;
  message.step = image.step == 0 ? infer_image_step(image.width, image.encoding) : image.step;
  message.data = image.data;
  std::lock_guard<std::mutex> lock(mcap_mutex_);
  auto& channel = raw_image_channel(topic);
  if (!should_log_channel(topic,
                          raw_image_channels_,
                          drop_when_unsubscribed_value,
                          mcap_writer_.has_value())) {
    return log_time_ns;
  }
  throw_on_foxglove_error(channel.log(message, log_time_ns), "RawImage log");
  return log_time_ns;
}

uint64_t FoxglovePublisherOp::publish_compressed_video(const FoxgloveCompressedVideo& video) {
  const auto log_time_ns = resolve_timestamp_ns(video.timestamp_ns);
  auto topic = normalize_topic(video.topic, "/video/compressed");
  const auto drop_when_unsubscribed_value = drop_when_unsubscribed();
  foxglove::messages::CompressedVideo message;
  message.timestamp = timestamp_from_ns(log_time_ns);
  message.frame_id = video.frame_id;
  message.format = video.format;
  message.data = video.data;
  std::lock_guard<std::mutex> lock(mcap_mutex_);
  auto& channel = compressed_video_channel(topic);
  if (!should_log_channel(topic,
                          compressed_video_channels_,
                          drop_when_unsubscribed_value,
                          mcap_writer_.has_value())) {
    return log_time_ns;
  }
  throw_on_foxglove_error(channel.log(message, log_time_ns), "CompressedVideo log");
  return log_time_ns;
}

uint64_t FoxglovePublisherOp::publish_calibration(
    const FoxgloveCameraCalibration& calibration) {
  const auto log_time_ns = resolve_timestamp_ns(calibration.timestamp_ns);
  auto topic = normalize_topic(calibration.topic, "/camera/calibration");
  const auto drop_when_unsubscribed_value = drop_when_unsubscribed();
  foxglove::messages::CameraCalibration message;
  message.timestamp = timestamp_from_ns(log_time_ns);
  message.frame_id = calibration.frame_id;
  message.width = calibration.width;
  message.height = calibration.height;
  message.distortion_model = calibration.distortion_model;
  message.d = calibration.distortion;
  message.k = calibration.k;
  message.r = calibration.r;
  message.p = calibration.p;
  std::lock_guard<std::mutex> lock(mcap_mutex_);
  auto& channel = calibration_channel(topic);
  if (!should_log_channel(topic,
                          calibration_channels_,
                          drop_when_unsubscribed_value,
                          mcap_writer_.has_value())) {
    return log_time_ns;
  }
  throw_on_foxglove_error(channel.log(message, log_time_ns), "CameraCalibration log");
  return log_time_ns;
}

uint64_t FoxglovePublisherOp::publish_annotations(
    const FoxgloveImageAnnotations& annotations) {
  const auto log_time_ns = resolve_timestamp_ns(annotations.timestamp_ns);
  auto topic = normalize_topic(annotations.topic, "/image/annotations");
  const auto drop_when_unsubscribed_value = drop_when_unsubscribed();
  foxglove::messages::ImageAnnotations message;
  message.timestamp = timestamp_from_ns(log_time_ns);
  for (const auto& box : annotations.boxes) {
    foxglove::messages::PointsAnnotation rectangle;
    rectangle.timestamp = message.timestamp;
    rectangle.type =
        foxglove::messages::PointsAnnotation::PointsAnnotationType::LINE_LOOP;
    rectangle.points = {{box.x, box.y},
                        {box.x + box.width, box.y},
                        {box.x + box.width, box.y + box.height},
                        {box.x, box.y + box.height}};
    rectangle.thickness = 2.0;
    rectangle.outline_color = foxglove::messages::Color{0.0, 1.0, 0.2, 1.0};
    if (!box.label.empty()) {
      rectangle.metadata.push_back({"label", box.label});
    }
    if (box.confidence >= 0.0) {
      rectangle.metadata.push_back({"confidence", fmt::format("{:.4f}", box.confidence)});
    }
    message.points.push_back(std::move(rectangle));

    if (!box.label.empty()) {
      foxglove::messages::TextAnnotation label;
      label.timestamp = message.timestamp;
      label.position = foxglove::messages::Point2{box.x, std::max(0.0, box.y - 4.0)};
      label.text = box.confidence >= 0.0 ? fmt::format("{} {:.2f}", box.label, box.confidence)
                                         : box.label;
      label.font_size = 14.0;
      label.text_color = foxglove::messages::Color{1.0, 1.0, 1.0, 1.0};
      label.background_color = foxglove::messages::Color{0.0, 0.0, 0.0, 0.65};
      message.texts.push_back(std::move(label));
    }
  }
  for (const auto& point_set : annotations.point_sets) {
    if (point_set.points.empty()) {
      continue;
    }

    foxglove::messages::PointsAnnotation points;
    points.timestamp = message.timestamp;
    points.type = point_set.type;
    points.thickness = point_set.thickness;
    points.outline_color = foxglove::messages::Color{
        point_set.color[0], point_set.color[1], point_set.color[2], point_set.color[3]};
    for (const auto& point : point_set.points) {
      points.points.push_back({point.x, point.y});
    }
    if (!point_set.label.empty()) {
      points.metadata.push_back({"label", point_set.label});
    }
    message.points.push_back(std::move(points));
  }
  for (const auto& text : annotations.texts) {
    foxglove::messages::TextAnnotation label;
    label.timestamp = message.timestamp;
    label.position = foxglove::messages::Point2{text.x, text.y};
    label.text = text.text;
    label.font_size = text.font_size;
    label.text_color = foxglove::messages::Color{1.0, 1.0, 1.0, 1.0};
    label.background_color = foxglove::messages::Color{0.0, 0.0, 0.0, 0.45};
    message.texts.push_back(std::move(label));
  }
  std::lock_guard<std::mutex> lock(mcap_mutex_);
  auto& channel = annotation_channel(topic);
  if (!should_log_channel(topic,
                          annotation_channels_,
                          drop_when_unsubscribed_value,
                          mcap_writer_.has_value())) {
    return log_time_ns;
  }
  throw_on_foxglove_error(channel.log(message, log_time_ns), "ImageAnnotations log");
  return log_time_ns;
}

uint64_t FoxglovePublisherOp::publish_point_cloud(const FoxglovePointCloud& point_cloud) {
  const auto log_time_ns = resolve_timestamp_ns(point_cloud.timestamp_ns);
  auto topic = normalize_topic(point_cloud.topic, "/points");
  const auto drop_when_unsubscribed_value = drop_when_unsubscribed();
  foxglove::messages::PointCloud message;
  message.timestamp = timestamp_from_ns(log_time_ns);
  message.frame_id = point_cloud.frame_id;
  message.point_stride = point_cloud.point_stride;
  message.fields = point_cloud.fields;
  message.data = point_cloud.data;
  std::lock_guard<std::mutex> lock(mcap_mutex_);
  auto& channel = point_cloud_channel(topic);
  if (!should_log_channel(topic,
                          point_cloud_channels_,
                          drop_when_unsubscribed_value,
                          mcap_writer_.has_value())) {
    return log_time_ns;
  }
  throw_on_foxglove_error(channel.log(message, log_time_ns), "PointCloud log");
  return log_time_ns;
}

uint64_t FoxglovePublisherOp::publish_frame_transform(
    const FoxgloveFrameTransform& transform) {
  const auto log_time_ns = resolve_timestamp_ns(transform.timestamp_ns);
  auto topic = normalize_topic(transform.topic, "/tf");
  const auto drop_when_unsubscribed_value = drop_when_unsubscribed();
  foxglove::messages::FrameTransform message;
  message.timestamp = timestamp_from_ns(log_time_ns);
  message.parent_frame_id = transform.parent_frame_id;
  message.child_frame_id = transform.child_frame_id;
  message.translation = foxglove::messages::Vector3{
      transform.translation[0], transform.translation[1], transform.translation[2]};
  message.rotation = foxglove::messages::Quaternion{
      transform.rotation[0], transform.rotation[1], transform.rotation[2], transform.rotation[3]};
  std::lock_guard<std::mutex> lock(mcap_mutex_);
  auto& channel = frame_transform_channel(topic);
  if (!should_log_channel(topic,
                          frame_transform_channels_,
                          drop_when_unsubscribed_value,
                          mcap_writer_.has_value())) {
    return log_time_ns;
  }
  throw_on_foxglove_error(channel.log(message, log_time_ns), "FrameTransform log");
  return log_time_ns;
}

uint64_t FoxglovePublisherOp::publish_key_value(const FoxgloveKeyValue& key_value) {
  const auto log_time_ns = resolve_timestamp_ns(key_value.timestamp_ns);
  auto topic = normalize_topic(key_value.topic, "/state");
  const auto drop_when_unsubscribed_value = drop_when_unsubscribed();
  foxglove::messages::KeyValuePair message;
  message.key = key_value.key;
  message.value = key_value.value;
  std::lock_guard<std::mutex> lock(mcap_mutex_);
  auto& channel = key_value_channel(topic);
  if (!should_log_channel(topic,
                          key_value_channels_,
                          drop_when_unsubscribed_value,
                          mcap_writer_.has_value())) {
    return log_time_ns;
  }
  throw_on_foxglove_error(channel.log(message, log_time_ns), "KeyValuePair log");
  return log_time_ns;
}

void FoxgloveTensorAdapterOp::setup(OperatorSpec& spec) {
  spec.input<gxf::Entity>("input");
  spec.output<std::shared_ptr<FoxgloveBatch>>("messages");

  spec.param(topic_, "topic", "Topic", "Foxglove topic to publish", std::string("/image"));
  spec.param(frame_id_, "frame_id", "Frame ID", "Frame ID for Foxglove messages",
             std::string("camera"));
  spec.param(tensor_name_,
             "tensor_name",
             "Tensor name",
             "Name of the tensor to read from the input entity; empty means first tensor",
             std::string(""));
  spec.param(encoding_,
             "encoding",
             "Encoding",
             "Foxglove RawImage encoding. Leave empty to infer from VideoBuffer or tensor shape",
             std::string(""));
  spec.param(width_, "width", "Width", "Image width override", 0u);
  spec.param(height_, "height", "Height", "Image height override", 0u);
  spec.param(step_, "step", "Step", "Row stride override", 0u);
  spec.param(prefer_video_buffer_,
             "prefer_video_buffer",
             "Prefer VideoBuffer",
             "Use a GXF VideoBuffer if one is attached to the entity",
             true);
  spec.param(timestamp_metadata_keys_,
             "timestamp_metadata_keys",
             "Timestamp metadata keys",
             "Comma-separated metadata keys checked before falling back to acquisition timestamp "
             "or current time",
             std::string(kDefaultTimestampMetadataKeys));
  spec.param(allocator_,
             "allocator",
             "Allocator",
             "Optional allocator used for pinned host staging buffers",
             std::shared_ptr<Allocator>{});
}

void FoxgloveTensorAdapterOp::compute(InputContext& op_input,
                                      OutputContext& op_output,
                                      [[maybe_unused]] ExecutionContext& context) {
  auto maybe_entity = op_input.receive<gxf::Entity>("input");
  if (!maybe_entity) {
    throw std::runtime_error("FoxgloveTensorAdapterOp required input is empty");
  }
  auto entity = maybe_entity.value();
  const auto stream = op_input.receive_cuda_stream("input", false, true);
  const auto timestamp_ns =
      timestamp_from_input_metadata(*this, op_input, "input", timestamp_metadata_keys_.get());
  auto batch = std::make_shared<FoxgloveBatch>();

  if (prefer_video_buffer_.get()) {
    auto maybe_video_buffer =
        static_cast<nvidia::gxf::Entity&>(entity).get<nvidia::gxf::VideoBuffer>();
    if (maybe_video_buffer) {
      auto image = image_from_video_buffer(*maybe_video_buffer.value().get(), stream);
      image.timestamp_ns = timestamp_ns;
      log_resolved_topic_once(topic_logged_, name(), image.topic);
      batch->images.push_back(std::move(image));
      batch->key_values = metadata_key_values(*this, timestamp_ns);
      op_output.emit(batch, "messages");
      return;
    }
  }

  auto maybe_tensor = tensor_from_entity(entity, tensor_name_.get(), "FoxgloveTensorAdapterOp");
  auto image = image_from_tensor(*maybe_tensor, stream);
  image.timestamp_ns = timestamp_ns;
  log_resolved_topic_once(topic_logged_, name(), image.topic);
  batch->images.push_back(std::move(image));
  batch->key_values = metadata_key_values(*this, timestamp_ns);
  op_output.emit(batch, "messages");
}

FoxgloveImage FoxgloveTensorAdapterOp::image_from_video_buffer(
    const nvidia::gxf::VideoBuffer& buffer, cudaStream_t stream) const {
  const auto info = buffer.video_frame_info();
  std::string encoding = encoding_.get();
  if (encoding.empty()) {
    encoding = gxf_video_encoding(info.color_format);
  }
  if (encoding.empty()) {
    throw std::runtime_error("Unsupported GXF VideoBuffer format; set encoding explicitly");
  }

  FoxgloveImage image;
  image.topic = normalize_topic(topic_.get(), "/image");
  image.frame_id = frame_id_.get();
  image.encoding = encoding;
  image.width = width_.get() == 0 ? info.width : width_.get();
  image.height = height_.get() == 0 ? info.height : height_.get();
  image.step = step_.get() == 0 ? video_buffer_image_step(info, image.width, image.encoding)
                                : step_.get();
  image.data = copy_host_or_device(buffer.pointer(),
                                   buffer.size(),
                                   buffer.storage_type() ==
                                       nvidia::gxf::MemoryStorageType::kDevice,
                                   allocator_.get(),
                                   &pinned_host_pool_,
                                   stream);
  return image;
}

FoxgloveImage FoxgloveTensorAdapterOp::image_from_tensor(const Tensor& tensor,
                                                         cudaStream_t stream) const {
  const auto layout = tensor_image_layout(tensor, "Tensor image input");

  FoxgloveImage image;
  image.topic = normalize_topic(topic_.get(), "/image");
  image.frame_id = frame_id_.get();
  image.width = width_.get();
  image.height = height_.get();
  if (image.height == 0) {
    image.height = layout.height;
  }
  if (image.width == 0) {
    image.width = layout.width;
  }

  image.encoding = encoding_.get();
  if (image.encoding.empty()) {
    image.encoding = tensor_image_encoding(tensor, layout.channels);
  }
  image.step =
      step_.get() == 0 ? tensor_image_step(tensor, layout, image.encoding, image.width)
                       : step_.get();

  const auto device = tensor.device();
  const bool is_device = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
  image.data =
      copy_host_or_device(tensor.data(), tensor.nbytes(), is_device, allocator_.get(),
                          &pinned_host_pool_, stream);
  return image;
}

void FoxgloveDetectionAdapterOp::setup(OperatorSpec& spec) {
  spec.input<TensorMap>("input");
  spec.output<std::shared_ptr<FoxgloveBatch>>("messages");

  spec.param(annotation_topic_,
             "annotation_topic",
             "Annotation topic",
             "Foxglove ImageAnnotations topic for detection results",
             std::string("/detections"));
  spec.param(boxes_tensor_,
             "boxes_tensor",
             "Boxes tensor",
             "Tensor containing boxes as Nx4 coordinates",
             std::string("boxes"));
  spec.param(scores_tensor_,
             "scores_tensor",
             "Scores tensor",
             "Tensor containing one confidence score per box",
             std::string("scores"));
  spec.param(labels_tensor_,
             "labels_tensor",
             "Labels tensor",
             "Tensor containing one integer class ID per box",
             std::string("labels"));
  spec.param(combined_tensor_,
             "combined_tensor",
             "Combined tensor",
             "Optional tensor containing rows of combined box, score and class data",
             std::string(""));
  spec.param(combined_format_,
             "combined_format",
             "Combined format",
             "combined tensor row format: xyxy_score_label or batch_label_score_xyxy",
             std::string("xyxy_score_label"));
  spec.param(box_format_,
             "box_format",
             "Box format",
             "Box coordinate layout for separate boxes tensors: xyxy or xywh",
             std::string("xyxy"));
  spec.param(label_map_,
             "label_map",
             "Label map",
             "Comma-separated class labels indexed by model class ID",
             std::string(""));
  spec.param(image_width_,
             "image_width",
             "Image width",
             "Source image width used to scale normalized box coordinates",
             0u);
  spec.param(image_height_,
             "image_height",
             "Image height",
             "Source image height used to scale normalized box coordinates",
             0u);
  spec.param(score_threshold_,
             "score_threshold",
             "Score threshold",
             "Minimum confidence score to publish",
             0.25);
  spec.param(normalized_coordinates_,
             "normalized_coordinates",
             "Normalized coordinates",
             "Whether detection coordinates are normalized to [0, 1]",
             false);
  spec.param(clamp_to_image_,
             "clamp_to_image",
             "Clamp to image",
             "Clamp annotation boxes to image bounds when image dimensions are provided",
             true);
  spec.param(timestamp_metadata_keys_,
             "timestamp_metadata_keys",
             "Timestamp metadata keys",
             "Comma-separated metadata keys checked before falling back to acquisition timestamp "
             "or current time",
             std::string(kDefaultTimestampMetadataKeys));
  spec.param(allocator_,
             "allocator",
             "Allocator",
             "Optional allocator used for pinned host staging buffers",
             std::shared_ptr<Allocator>{});
}

std::vector<std::string> FoxgloveDetectionAdapterOp::labels() const {
  return parse_label_map(label_map_.get());
}

void FoxgloveDetectionAdapterOp::compute(InputContext& op_input,
                                         OutputContext& op_output,
                                         [[maybe_unused]] ExecutionContext& context) {
  auto maybe_tensors = op_input.receive<TensorMap>("input");
  if (!maybe_tensors) {
    throw std::runtime_error("FoxgloveDetectionAdapterOp required input is empty");
  }
  const auto stream = op_input.receive_cuda_stream("input", false, true);
  const auto timestamp_ns =
      timestamp_from_input_metadata(*this, op_input, "input", timestamp_metadata_keys_.get());
  const auto& tensors = maybe_tensors.value();
  const auto label_names = labels();

  FoxgloveImageAnnotations annotations;
  annotations.topic = normalize_topic(annotation_topic_.get(), "/detections");
  annotations.timestamp_ns = timestamp_ns;

  const auto combined_tensor = optional_tensor_from_map(tensors, combined_tensor_.get());
  if (combined_tensor) {
    const auto values =
        tensor_to_float_vector(*combined_tensor, allocator_.get(), &pinned_host_pool_, stream);
    const auto [rows, columns] = matrix_shape(*combined_tensor, 6);
    const size_t minimum_columns =
        combined_format_.get() == "batch_label_score_xyxy" ? 7U : 6U;
    if (columns < minimum_columns) {
      throw std::runtime_error(
          fmt::format("Combined detection tensor with format '{}' must have at least {} columns, "
                      "got {}",
                      combined_format_.get(),
                      minimum_columns,
                      columns));
    }
    if (values.size() < rows * columns) {
      throw std::runtime_error("Combined detection tensor data is smaller than its shape");
    }

    for (size_t row = 0; row < rows; ++row) {
      const auto* item = values.data() + row * columns;
      double x1 = 0.0;
      double y1 = 0.0;
      double x2 = 0.0;
      double y2 = 0.0;
      double score = -1.0;
      int64_t class_id = -1;

      if (combined_format_.get() == "xyxy_score_label") {
        x1 = item[0];
        y1 = item[1];
        x2 = item[2];
        y2 = item[3];
        score = item[4];
        class_id = static_cast<int64_t>(std::llround(item[5]));
      } else if (combined_format_.get() == "batch_label_score_xyxy") {
        const auto batch_index = static_cast<int64_t>(std::llround(item[0]));
        if (batch_index != 0) {
          throw std::runtime_error(fmt::format(
              "FoxgloveDetectionAdapterOp combined_format=batch_label_score_xyxy supports "
              "only batch index 0, got {} on row {}; split batched detections before "
              "publishing",
              batch_index,
              row));
        }
        x1 = item[3];
        y1 = item[4];
        x2 = item[5];
        y2 = item[6];
        score = item[2];
        class_id = static_cast<int64_t>(std::llround(item[1]));
      } else {
        throw std::runtime_error(fmt::format(
            "Unsupported combined_format '{}'; expected xyxy_score_label or "
            "batch_label_score_xyxy",
            combined_format_.get()));
      }

      if (score >= 0.0 && score < score_threshold_.get()) {
        continue;
      }
      annotations.boxes.push_back(make_detection_box(x1,
                                                     y1,
                                                     x2,
                                                     y2,
                                                     "xyxy",
                                                     normalized_coordinates_.get(),
                                                     image_width_.get(),
                                                     image_height_.get(),
                                                     clamp_to_image_.get(),
                                                     label_for_id(class_id, label_names),
                                                     score));
    }
  } else {
    const auto boxes_tensor = tensor_from_map(tensors, boxes_tensor_.get(), "detection boxes");
    const auto boxes =
        tensor_to_float_vector(*boxes_tensor, allocator_.get(), &pinned_host_pool_, stream);
    if (boxes.size() % 4 != 0) {
      throw std::runtime_error("Detection boxes tensor size must be divisible by 4");
    }
    const auto count = boxes.size() / 4;

    std::vector<float> scores;
    if (const auto scores_tensor = optional_tensor_from_map(tensors, scores_tensor_.get())) {
      scores = tensor_to_float_vector(*scores_tensor, allocator_.get(), &pinned_host_pool_, stream);
    }
    std::vector<float> class_ids;
    if (const auto labels_tensor = optional_tensor_from_map(tensors, labels_tensor_.get())) {
      class_ids =
          tensor_to_float_vector(*labels_tensor, allocator_.get(), &pinned_host_pool_, stream);
    }

    for (size_t i = 0; i < count; ++i) {
      const double score = i < scores.size() ? static_cast<double>(scores[i]) : -1.0;
      if (score >= 0.0 && score < score_threshold_.get()) {
        continue;
      }

      std::string label = "detection";
      if (i < class_ids.size()) {
        label = label_for_id(static_cast<int64_t>(std::llround(class_ids[i])), label_names);
      }

      annotations.boxes.push_back(make_detection_box(boxes[i * 4 + 0],
                                                     boxes[i * 4 + 1],
                                                     boxes[i * 4 + 2],
                                                     boxes[i * 4 + 3],
                                                     box_format_.get(),
                                                     normalized_coordinates_.get(),
                                                     image_width_.get(),
                                                     image_height_.get(),
                                                     clamp_to_image_.get(),
                                                     label,
                                                     score));
    }
  }

  auto batch = std::make_shared<FoxgloveBatch>();
  log_resolved_topic_once(topic_logged_, name(), annotations.topic);
  batch->annotations.push_back(std::move(annotations));
  batch->key_values = metadata_key_values(*this, timestamp_ns);
  op_output.emit(batch, "messages");
}

void FoxgloveSegmentationMaskAdapterOp::setup(OperatorSpec& spec) {
  spec.input<TensorMap>("input");
  spec.output<std::shared_ptr<FoxgloveBatch>>("messages");

  spec.param(topic_,
             "topic",
             "Topic",
             "Foxglove RawImage topic for segmentation masks",
             std::string("/segmentation"));
  spec.param(frame_id_,
             "frame_id",
             "Frame ID",
             "Frame ID for Foxglove segmentation messages",
             std::string("camera"));
  spec.param(tensor_name_,
             "tensor_name",
             "Tensor name",
             "Segmentation tensor name; empty selects the first tensor",
             std::string("out_tensor"));
  spec.param(encoding_,
             "encoding",
             "Encoding",
             "Foxglove RawImage encoding for the mask",
             std::string("mono8"));
  spec.param(width_, "width", "Width", "Segmentation width override", 0u);
  spec.param(height_, "height", "Height", "Segmentation height override", 0u);
  spec.param(step_, "step", "Step", "Row stride override", 0u);
  spec.param(timestamp_metadata_keys_,
             "timestamp_metadata_keys",
             "Timestamp metadata keys",
             "Comma-separated metadata keys checked before falling back to acquisition timestamp "
             "or current time",
             std::string(kDefaultTimestampMetadataKeys));
  spec.param(allocator_,
             "allocator",
             "Allocator",
             "Optional allocator used for pinned host staging buffers",
             std::shared_ptr<Allocator>{});
}

void FoxgloveSegmentationMaskAdapterOp::compute(InputContext& op_input,
                                                OutputContext& op_output,
                                                [[maybe_unused]] ExecutionContext& context) {
  auto maybe_tensors = op_input.receive<TensorMap>("input");
  if (!maybe_tensors) {
    throw std::runtime_error("FoxgloveSegmentationMaskAdapterOp required input is empty");
  }
  const auto stream = op_input.receive_cuda_stream("input", false, true);
  const auto timestamp_ns =
      timestamp_from_input_metadata(*this, op_input, "input", timestamp_metadata_keys_.get());
  const auto& tensors = maybe_tensors.value();
  const auto& tensor = tensor_from_map(tensors, tensor_name_.get(), "segmentation mask");
  const auto layout = tensor_image_layout(*tensor, "Segmentation tensor");

  uint32_t height = height_.get();
  uint32_t width = width_.get();
  if (height == 0) {
    height = layout.height;
  }
  if (width == 0) {
    width = layout.width;
  }

  FoxgloveImage image;
  image.topic = normalize_topic(topic_.get(), "/segmentation");
  image.frame_id = frame_id_.get();
  image.encoding = encoding_.get().empty() ? "mono8" : encoding_.get();
  image.width = width;
  image.height = height;
  image.step =
      step_.get() == 0 ? tensor_image_step(*tensor, layout, image.encoding, image.width)
                       : step_.get();
  image.timestamp_ns = timestamp_ns;
  image.data = copy_tensor_bytes(*tensor, allocator_.get(), &pinned_host_pool_, stream);

  auto batch = std::make_shared<FoxgloveBatch>();
  log_resolved_topic_once(topic_logged_, name(), image.topic);
  batch->images.push_back(std::move(image));
  batch->key_values = metadata_key_values(*this, timestamp_ns);
  op_output.emit(batch, "messages");
}

void FoxgloveCompressedVideoAdapterOp::setup(OperatorSpec& spec) {
  spec.input<gxf::Entity>("input");
  spec.output<std::shared_ptr<FoxgloveBatch>>("messages");

  spec.param(topic_,
             "topic",
             "Topic",
             "Foxglove CompressedVideo topic",
             std::string("/video/compressed"));
  spec.param(frame_id_,
             "frame_id",
             "Frame ID",
             "Frame ID for Foxglove compressed video messages",
             std::string("camera"));
  spec.param(tensor_name_,
             "tensor_name",
             "Tensor name",
             "Encoded video tensor component name; empty selects the first tensor",
             std::string(""));
  spec.param(format_,
             "format",
             "Format",
             "Foxglove compressed video format such as h264 or h265",
             std::string("h264"));
  spec.param(timestamp_metadata_keys_,
             "timestamp_metadata_keys",
             "Timestamp metadata keys",
             "Comma-separated metadata keys checked before falling back to acquisition timestamp "
             "or current time",
             std::string(kDefaultTimestampMetadataKeys));
  spec.param(allocator_,
             "allocator",
             "Allocator",
             "Optional allocator used for pinned host staging buffers",
             std::shared_ptr<Allocator>{});
}

void FoxgloveCompressedVideoAdapterOp::compute(InputContext& op_input,
                                               OutputContext& op_output,
                                               [[maybe_unused]] ExecutionContext& context) {
  auto maybe_entity = op_input.receive<gxf::Entity>("input");
  if (!maybe_entity) {
    throw std::runtime_error("FoxgloveCompressedVideoAdapterOp required input is empty");
  }
  auto entity = maybe_entity.value();
  const auto stream = op_input.receive_cuda_stream("input", false, true);
  const auto timestamp_ns =
      timestamp_from_input_metadata(*this, op_input, "input", timestamp_metadata_keys_.get());

  auto maybe_tensor =
      tensor_from_entity(entity, tensor_name_.get(), "FoxgloveCompressedVideoAdapterOp");

  FoxgloveCompressedVideo video;
  video.topic = normalize_topic(topic_.get(), "/video/compressed");
  video.frame_id = frame_id_.get();
  video.format = format_.get();
  video.timestamp_ns = timestamp_ns;
  video.data = copy_tensor_bytes(*maybe_tensor, allocator_.get(), &pinned_host_pool_, stream);

  auto batch = std::make_shared<FoxgloveBatch>();
  log_resolved_topic_once(topic_logged_, name(), video.topic);
  batch->compressed_videos.push_back(std::move(video));
  batch->key_values = metadata_key_values(*this, timestamp_ns);
  op_output.emit(batch, "messages");
}

void FoxglovePoseAdapterOp::setup(OperatorSpec& spec) {
  spec.input<TensorMap>("input");
  spec.output<std::shared_ptr<FoxgloveBatch>>("messages");

  spec.param(topic_,
             "topic",
             "Topic",
             "Foxglove FrameTransform topic",
             std::string("/tf"));
  spec.param(tensor_name_,
             "tensor_name",
             "Tensor name",
             "Pose tensor name; empty selects the first tensor",
             std::string(""));
  spec.param(parent_frame_id_,
             "parent_frame_id",
             "Parent frame ID",
             "Parent frame for the transform",
             std::string("world"));
  spec.param(child_frame_id_,
             "child_frame_id",
             "Child frame ID",
             "Child frame for the transform",
             std::string("sensor"));
  spec.param(format_,
             "format",
             "Format",
             "Pose tensor format: matrix4x4 or xyz_quat",
             std::string("matrix4x4"));
  spec.param(timestamp_metadata_keys_,
             "timestamp_metadata_keys",
             "Timestamp metadata keys",
             "Comma-separated metadata keys checked before falling back to acquisition timestamp "
             "or current time",
             std::string(kDefaultTimestampMetadataKeys));
  spec.param(allocator_,
             "allocator",
             "Allocator",
             "Optional allocator used for pinned host staging buffers",
             std::shared_ptr<Allocator>{});
}

void FoxglovePoseAdapterOp::compute(InputContext& op_input,
                                    OutputContext& op_output,
                                    [[maybe_unused]] ExecutionContext& context) {
  auto maybe_tensors = op_input.receive<TensorMap>("input");
  if (!maybe_tensors) {
    throw std::runtime_error("FoxglovePoseAdapterOp required input is empty");
  }
  const auto stream = op_input.receive_cuda_stream("input", false, true);
  const auto timestamp_ns =
      timestamp_from_input_metadata(*this, op_input, "input", timestamp_metadata_keys_.get());
  const auto& tensor = tensor_from_map(maybe_tensors.value(), tensor_name_.get(), "pose transform");
  auto transform = transform_from_values(
      tensor_to_float_vector(*tensor, allocator_.get(), &pinned_host_pool_, stream),
      format_.get());
  transform.topic = normalize_topic(topic_.get(), "/tf");
  transform.parent_frame_id = parent_frame_id_.get();
  transform.child_frame_id = child_frame_id_.get();
  transform.timestamp_ns = timestamp_ns;

  auto batch = std::make_shared<FoxgloveBatch>();
  log_resolved_topic_once(topic_logged_, name(), transform.topic);
  batch->frame_transforms.push_back(std::move(transform));
  batch->key_values = metadata_key_values(*this, timestamp_ns);
  op_output.emit(batch, "messages");
}

}  // namespace holoscan::ops
