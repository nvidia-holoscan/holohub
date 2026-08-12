// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorrt_inference/tensorrt_inference.hpp"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>

#include "v4l2_depth_common/cuda_device_guard.hpp"
#include "v4l2_depth_common/tensor_utils.hpp"

namespace holoscan::examples::v4l2_depth {
namespace {

template <typename T>
struct TensorRtDelete {
  void operator()(T* pointer) const noexcept { delete pointer; }
};

template <typename T>
using TensorRtPtr = std::unique_ptr<T, TensorRtDelete<T>>;

struct ModelIdentity {
  std::uintmax_t descriptor_byte_size{};
  std::uintmax_t external_data_byte_size{};
  std::uint64_t content_hash{};

  bool operator==(const ModelIdentity&) const = default;
};

constexpr std::uint64_t kFnv1aOffsetBasis = 14695981039346656037ULL;
constexpr std::uint64_t kFnv1aPrime = 1099511628211ULL;
constexpr std::size_t kIdentityReadBytes = 64U * 1024U;
constexpr std::size_t kMaximumOutputRank = 8U;

class CacheLock final {
 public:
  explicit CacheLock(const std::filesystem::path& cache_path) {
    std::filesystem::path lock_path = cache_path;
    lock_path += ".lock";
    if (lock_path.has_parent_path()) {
      std::error_code directory_error;
      std::filesystem::create_directories(lock_path.parent_path(), directory_error);
      if (directory_error) {
        error_message_ =
            "could not create TensorRT cache directory: " + directory_error.message();
        return;
      }
    }

    fd_ = ::open(lock_path.c_str(),
                 O_CREAT | O_RDWR | O_CLOEXEC,
                 S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
    if (fd_ < 0) {
      const int error_number = errno;
      error_message_ =
          "could not open TensorRT cache lock '" + lock_path.string() +
          "': " + std::error_code(error_number, std::generic_category()).message();
      return;
    }
    if (::flock(fd_, LOCK_EX) < 0) {
      const int error_number = errno;
      error_message_ =
          "could not acquire TensorRT cache lock '" + lock_path.string() +
          "': " + std::error_code(error_number, std::generic_category()).message();
      static_cast<void>(::close(fd_));
      fd_ = -1;
    }
  }

  CacheLock(const CacheLock&) = delete;
  CacheLock& operator=(const CacheLock&) = delete;

  ~CacheLock() {
    if (fd_ >= 0) {
      static_cast<void>(::flock(fd_, LOCK_UN));
      static_cast<void>(::close(fd_));
    }
  }

  [[nodiscard]] bool active() const noexcept { return fd_ >= 0; }
  [[nodiscard]] const std::string& error_message() const noexcept {
    return error_message_;
  }

 private:
  int fd_{-1};
  std::string error_message_;
};

[[nodiscard]] std::vector<char> read_binary(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("could not open '" + path.string() + "'");
  }
  const std::streamoff end = stream.tellg();
  if (end <= 0) {
    throw std::runtime_error("'" + path.string() + "' is empty");
  }
  std::vector<char> bytes(static_cast<std::size_t>(end));
  stream.seekg(0, std::ios::beg);
  if (!stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("could not read '" + path.string() + "'");
  }
  return bytes;
}

void write_binary(const std::filesystem::path& path, const void* data, std::size_t size) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::filesystem::path temporary = path;
  temporary += ".tmp." + std::to_string(::getpid());
  std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
  if (!stream || !stream.write(static_cast<const char*>(data),
                               static_cast<std::streamsize>(size))) {
    stream.close();
    std::error_code remove_error;
    std::filesystem::remove(temporary, remove_error);
    throw std::runtime_error("could not write TensorRT cache '" + path.string() + "'");
  }
  stream.close();
  if (!stream) {
    std::error_code remove_error;
    std::filesystem::remove(temporary, remove_error);
    throw std::runtime_error("could not finalize TensorRT cache '" + path.string() + "'");
  }
  std::error_code rename_error;
  std::filesystem::rename(temporary, path, rename_error);
  if (rename_error) {
    std::error_code remove_error;
    std::filesystem::remove(temporary, remove_error);
    throw std::runtime_error(
        "could not publish TensorRT cache '" + path.string() + "': " +
        rename_error.message());
  }
}

void hash_bytes(std::uint64_t& hash, const char* data, std::size_t size) {
  for (std::size_t index = 0U; index < size; ++index) {
    hash ^= static_cast<std::uint8_t>(data[index]);
    hash *= kFnv1aPrime;
  }
}

[[nodiscard]] std::uintmax_t hash_model_file(const std::filesystem::path& path,
                                             const char* description,
                                             std::uint64_t& hash) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error(
        std::string("could not open ") + description + " '" + path.string() + "'");
  }

  std::uintmax_t byte_size = 0U;
  std::array<char, kIdentityReadBytes> buffer{};
  while (stream) {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize bytes_read = stream.gcount();
    byte_size += static_cast<std::uintmax_t>(bytes_read);
    hash_bytes(hash, buffer.data(), static_cast<std::size_t>(bytes_read));
  }
  if (!stream.eof()) {
    throw std::runtime_error(
        std::string("could not read ") + description + " '" + path.string() + "'");
  }
  if (byte_size == 0U) {
    throw std::runtime_error(
        std::string(description) + " '" + path.string() + "' is empty");
  }
  return byte_size;
}

[[nodiscard]] ModelIdentity identify_model(const std::filesystem::path& path) {
  ModelIdentity identity{.content_hash = kFnv1aOffsetBasis};
  constexpr char descriptor_tag[] = "onnx-descriptor";
  hash_bytes(identity.content_hash, descriptor_tag, sizeof(descriptor_tag));
  identity.descriptor_byte_size =
      hash_model_file(path, "ONNX model", identity.content_hash);

  std::filesystem::path external_data_path = path;
  external_data_path += ".data";
  std::error_code status_error;
  const std::filesystem::file_status external_status =
      std::filesystem::status(external_data_path, status_error);
  if (status_error &&
      status_error != std::errc::no_such_file_or_directory) {
    throw std::runtime_error(
        "could not inspect ONNX external data '" + external_data_path.string() +
        "': " + status_error.message());
  }
  if (!status_error && std::filesystem::exists(external_status)) {
    if (!std::filesystem::is_regular_file(external_status)) {
      throw std::runtime_error(
          "ONNX external data is not a regular file: '" + external_data_path.string() + "'");
    }
    constexpr char external_data_tag[] = "onnx-external-data";
    hash_bytes(identity.content_hash, external_data_tag, sizeof(external_data_tag));
    identity.external_data_byte_size =
        hash_model_file(external_data_path, "ONNX external data", identity.content_hash);
  }
  return identity;
}

[[nodiscard]] std::filesystem::path cache_identity_path(
    const std::filesystem::path& cache_path) {
  std::filesystem::path identity_path = cache_path;
  identity_path += ".model-id";
  return identity_path;
}

[[nodiscard]] std::optional<ModelIdentity> read_model_identity(
    const std::filesystem::path& path) {
  std::ifstream stream(path);
  std::string version;
  ModelIdentity identity;
  stream >> version >> identity.descriptor_byte_size >> identity.external_data_byte_size >>
      std::hex >> identity.content_hash;
  if (!stream || version != "v2") {
    return std::nullopt;
  }
  stream >> std::ws;
  if (!stream.eof()) {
    return std::nullopt;
  }
  return identity;
}

void write_model_identity(const std::filesystem::path& path,
                          const ModelIdentity& identity) {
  std::ostringstream stream;
  stream << "v2 " << identity.descriptor_byte_size << ' '
         << identity.external_data_byte_size << ' ' << std::hex << std::setw(16)
         << std::setfill('0') << identity.content_hash << '\n';
  const std::string value = std::move(stream).str();
  write_binary(path, value.data(), value.size());
}

[[nodiscard]] bool cache_matches_model(const std::filesystem::path& cache_path,
                                       const ModelIdentity& model_identity) {
  const auto cached_identity = read_model_identity(cache_identity_path(cache_path));
  return cached_identity.has_value() && *cached_identity == model_identity;
}

void write_engine_cache(const std::filesystem::path& cache_path,
                        const void* data,
                        std::size_t size,
                        const ModelIdentity& model_identity) {
  const std::filesystem::path identity_path = cache_identity_path(cache_path);
  std::error_code error;
  std::filesystem::remove(identity_path, error);
  if (error) {
    throw std::runtime_error(
        "could not invalidate TensorRT cache identity '" + identity_path.string() +
        "': " + error.message());
  }
  write_binary(cache_path, data, size);
  write_model_identity(identity_path, model_identity);
}

[[nodiscard]] std::size_t checked_volume(const nvinfer1::Dims& dimensions) {
  if (dimensions.nbDims <= 0) {
    throw std::runtime_error("TensorRT tensor rank must be positive");
  }
  std::size_t result = 1U;
  for (int index = 0; index < dimensions.nbDims; ++index) {
    const int64_t extent = dimensions.d[index];
    if (extent <= 0) {
      throw std::runtime_error("dynamic or zero TensorRT dimensions are not supported");
    }
    const auto unsigned_extent = static_cast<std::size_t>(extent);
    if (result > std::numeric_limits<std::size_t>::max() / unsigned_extent) {
      throw std::runtime_error("TensorRT tensor size overflow");
    }
    result *= unsigned_extent;
  }
  return result;
}

[[nodiscard]] std::size_t checked_float_bytes(const nvinfer1::Dims& dimensions) {
  const std::size_t elements = checked_volume(dimensions);
  constexpr std::size_t maximum_bytes =
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
  if (elements > maximum_bytes / sizeof(float)) {
    throw std::runtime_error("TensorRT tensor byte size overflow");
  }
  return elements * sizeof(float);
}

[[nodiscard]] std::vector<std::int64_t> to_shape(const nvinfer1::Dims& dimensions) {
  std::vector<std::int64_t> shape;
  shape.reserve(static_cast<std::size_t>(dimensions.nbDims));
  for (int index = 0; index < dimensions.nbDims; ++index) {
    shape.push_back(dimensions.d[index]);
  }
  return shape;
}

}  // namespace

struct TensorRtInferenceOp::Impl {
  struct Logger final : nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
      if (severity <= Severity::kWARNING) {
        std::cerr << "[TensorRT] " << message << '\n';
      }
    }
  };

  Logger logger;
  TensorRtPtr<nvinfer1::IRuntime> runtime;
  TensorRtPtr<nvinfer1::ICudaEngine> engine;
  TensorRtPtr<nvinfer1::IExecutionContext> context;
  std::string input_name;
  std::string output_name;
  std::vector<std::int64_t> input_shape;
  std::vector<std::int64_t> output_shape;
  std::size_t input_bytes{};
  std::size_t output_bytes{};

  [[nodiscard]] TensorRtPtr<nvinfer1::IHostMemory> build_onnx(
      const std::filesystem::path& onnx_path) {
    TensorRtPtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
    if (!builder) {
      throw std::runtime_error("createInferBuilder failed");
    }

    const auto explicit_batch =
        1U << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TensorRtPtr<nvinfer1::INetworkDefinition> network{
        builder->createNetworkV2(explicit_batch)};
    TensorRtPtr<nvonnxparser::IParser> parser{
        network ? nvonnxparser::createParser(*network, logger) : nullptr};
    TensorRtPtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (!network || !parser || !config) {
      throw std::runtime_error("could not create TensorRT ONNX build objects");
    }
    if (!parser->parseFromFile(onnx_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
      std::ostringstream reason;
      reason << "ONNX parsing failed for '" << onnx_path.string() << "'";
      const int errors = parser->getNbErrors();
      if (errors > 0 && parser->getError(errors - 1) != nullptr) {
        reason << ": " << parser->getError(errors - 1)->desc();
      }
      throw std::runtime_error(reason.str());
    }
    if (network->getNbInputs() != 1 || network->getNbOutputs() != 1) {
      throw std::runtime_error("the ONNX model must have exactly one input and one output");
    }

    // Depth Anything exports floating-point depth. Preserve a float32 network
    // boundary even when TensorRT uses FP16 internally.
    network->getOutput(0)->setType(nvinfer1::DataType::kFLOAT);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30U);
    if (builder->platformHasFastFp16()) {
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    TensorRtPtr<nvinfer1::IHostMemory> serialized{
        builder->buildSerializedNetwork(*network, *config)};
    if (!serialized) {
      throw std::runtime_error("TensorRT failed to build the ONNX network");
    }
    return serialized;
  }

  void inspect_engine(std::size_t maximum_output_bytes) {
    if (!engine) {
      throw std::runtime_error("TensorRT engine is unavailable");
    }
    int input_count = 0;
    int output_count = 0;
    const int tensor_count = engine->getNbIOTensors();
    for (int index = 0; index < tensor_count; ++index) {
      const char* name = engine->getIOTensorName(index);
      if (name == nullptr) {
        throw std::runtime_error("TensorRT returned an unnamed I/O tensor");
      }
      const auto mode = engine->getTensorIOMode(name);
      if (mode == nvinfer1::TensorIOMode::kINPUT) {
        ++input_count;
        input_name = name;
      } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
        ++output_count;
        output_name = name;
      }
    }
    if (input_count != 1 || output_count != 1) {
      throw std::runtime_error("the TensorRT engine must have exactly one input and one output");
    }
    if (engine->getTensorDataType(input_name.c_str()) != nvinfer1::DataType::kFLOAT ||
        engine->getTensorDataType(output_name.c_str()) != nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error("the TensorRT engine must expose float32 input and output tensors");
    }
    const auto is_linear_float = [this](const std::string& name) {
      return engine->getTensorFormat(name.c_str()) == nvinfer1::TensorFormat::kLINEAR &&
             engine->getTensorVectorizedDim(name.c_str()) == -1 &&
             engine->getTensorLocation(name.c_str()) == nvinfer1::TensorLocation::kDEVICE;
    };
    if (!is_linear_float(input_name) || !is_linear_float(output_name)) {
      throw std::runtime_error(
          "the TensorRT engine must expose linear, non-vectorized device float32 I/O tensors");
    }

    const nvinfer1::Dims input_dimensions = engine->getTensorShape(input_name.c_str());
    const nvinfer1::Dims output_dimensions = engine->getTensorShape(output_name.c_str());
    if (output_dimensions.nbDims > static_cast<int>(kMaximumOutputRank)) {
      throw std::runtime_error("TensorRT output rank exceeds the operator's declared tensor bound");
    }
    input_shape = to_shape(input_dimensions);
    output_shape = to_shape(output_dimensions);
    input_bytes = checked_float_bytes(input_dimensions);
    output_bytes = checked_float_bytes(output_dimensions);
    if (output_bytes > maximum_output_bytes) {
      throw std::runtime_error("TensorRT output exceeds the operator's declared tensor bound");
    }
    context.reset(engine->createExecutionContext());
    if (!context) {
      throw std::runtime_error("createExecutionContext failed");
    }
  }
};

TensorRtInferenceOp::TensorRtInferenceOp(std::string model_path,
                                         std::string engine_cache_path,
                                         std::int32_t cuda_device,
                                         std::size_t max_output_bytes)
    : model_path_(std::move(model_path)),
      engine_cache_path_(std::move(engine_cache_path)),
      cuda_device_(cuda_device),
      max_output_bytes_(max_output_bytes) {
  if (model_path_.empty() || cuda_device_ < 0 || max_output_bytes_ == 0U) {
    throw std::invalid_argument(
        "TensorRtInferenceOp requires a model, nonnegative CUDA device, and positive output bound");
  }
}

TensorRtInferenceOp::~TensorRtInferenceOp() {
  stop();
}

void TensorRtInferenceOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(input, "input")
      .queue_depth(2U)
      .expects_tensor(holoscan::TensorPortLayout{
          .memory_kind = holoscan::MemoryKind::kCudaDevice,
          .dtype = kFloat32Dtype,
      });
  spec.output(output, "output")
      .produces_tensor(holoscan::TensorPortLayout{
          .memory_kind = holoscan::MemoryKind::kCudaDevice,
          .dtype = kFloat32Dtype,
      })
      .tensor_allocation(
          holoscan::TensorAllocationBounds{
              .max_rank = kMaximumOutputRank,
              .max_byte_span = max_output_bytes_,
          });
}

holoscan::Contract TensorRtInferenceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{input});
  return result;
}

void TensorRtInferenceOp::start() {
  const CudaDeviceGuard selected_device{cuda_device_};
  if (!selected_device.active()) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure,
        selected_device.error_message("TensorRtInferenceOp startup"));
  }
  impl_ = std::make_unique<Impl>();
  try {
    const std::filesystem::path model{model_path_};
    const bool model_is_onnx = model.extension() == ".onnx";
    std::vector<char> serialized_bytes;
    TensorRtPtr<nvinfer1::IHostMemory> built;
    std::unique_ptr<CacheLock> cache_lock;
    bool cache_available = model_is_onnx && !engine_cache_path_.empty();
    if (cache_available) {
      cache_lock = std::make_unique<CacheLock>(engine_cache_path_);
      if (!cache_lock->active()) {
        HOLOSCAN_LOG_WARN(
            "{}; continuing without TensorRT engine caching",
            cache_lock->error_message());
        cache_available = false;
      }
    }
    std::optional<ModelIdentity> model_identity;
    if (cache_available) {
      model_identity = identify_model(model);
    }

    std::filesystem::path selected_engine;
    std::error_code cache_probe_error;
    const bool cache_is_regular =
        cache_available &&
        std::filesystem::is_regular_file(engine_cache_path_, cache_probe_error);
    if (cache_is_regular) {
      if (cache_matches_model(engine_cache_path_, *model_identity)) {
        selected_engine = engine_cache_path_;
      } else {
        HOLOSCAN_LOG_WARN(
            "cached engine model identity is missing or stale; rebuilding from ONNX");
      }
    } else if (cache_probe_error &&
               cache_probe_error != std::errc::no_such_file_or_directory) {
      HOLOSCAN_LOG_WARN(
          "could not inspect TensorRT engine cache '{}': {}; rebuilding from ONNX",
          engine_cache_path_,
          cache_probe_error.message());
    } else if (!model_is_onnx) {
      selected_engine = model;
    }

    impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
    if (!impl_->runtime) {
      throw std::runtime_error("createInferRuntime failed");
    }

    if (!selected_engine.empty()) {
      try {
        serialized_bytes = read_binary(selected_engine);
        impl_->engine.reset(impl_->runtime->deserializeCudaEngine(serialized_bytes.data(),
                                                                  serialized_bytes.size()));
        if (!impl_->engine) {
          throw std::runtime_error("TensorRT engine deserialization failed");
        }
        impl_->inspect_engine(max_output_bytes_);
      } catch (const std::exception& error) {
        if (!model_is_onnx) {
          throw;
        }
        impl_->context.reset();
        impl_->engine.reset();
        std::cerr << "[TensorRT] could not use cached engine (" << error.what()
                  << "); rebuilding from ONNX\n";
      }
    }
    if (!impl_->engine) {
      if (!model_is_onnx) {
        throw std::runtime_error("TensorRT engine deserialization failed");
      }
      built = impl_->build_onnx(model);
      impl_->engine.reset(
          impl_->runtime->deserializeCudaEngine(built->data(), built->size()));
      if (!impl_->engine) {
        throw std::runtime_error("deserializing the freshly built TensorRT engine failed");
      }
      impl_->inspect_engine(max_output_bytes_);
      if (cache_available) {
        try {
          const ModelIdentity current_identity = identify_model(model);
          if (current_identity != *model_identity) {
            std::cerr << "warning: ONNX model changed while its TensorRT engine was built; "
                         "the cache was not updated\n";
          } else {
            write_engine_cache(
                engine_cache_path_, built->data(), built->size(), current_identity);
            std::cout << "cached TensorRT engine at " << engine_cache_path_ << '\n';
          }
        } catch (const std::exception& error) {
          std::cerr << "warning: " << error.what() << '\n';
        }
      }
    }

    std::cout << "loaded TensorRT network: input=" << impl_->input_name << " ("
              << impl_->input_bytes << " bytes), output=" << impl_->output_name << " ("
              << impl_->output_bytes << " bytes)\n";
  } catch (const std::exception& error) {
    impl_.reset();
    throw holoscan::RuntimeError(holoscan::ErrorCode::kFailure, error.what());
  }
}

void TensorRtInferenceOp::stop() {
  if (!impl_) {
    return;
  }
  const CudaDeviceGuard selected_device{cuda_device_};
  if (!selected_device.active()) {
    const std::string message =
        selected_device.error_message("TensorRtInferenceOp cleanup");
    std::cerr << message << '\n';
    return;
  }
  impl_->context.reset();
  impl_->engine.reset();
  impl_->runtime.reset();
  impl_.reset();
}

holoscan::expected<void, holoscan::Error> TensorRtInferenceOp::compute(
    holoscan::ExecutionContext& context) {
  if (!impl_ || !impl_->context) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kNotReady, "TensorRT is not initialized"});
  }

  auto sample = input.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  const holoscan::Tensor& tensor = sample->data;
  if (tensor.device().device_type != kDLCUDA ||
      tensor.device().device_id != cuda_device_ ||
      !same_dtype(tensor.dtype(), kFloat32Dtype) ||
      !shape_equals(tensor, impl_->input_shape) ||
      !tensor.is_contiguous() ||
      tensor.nbytes() != static_cast<std::int64_t>(impl_->input_bytes) ||
      tensor.data() == nullptr) {
    return holoscan::make_unexpected(
        invalid_tensor("TensorRT input shape, dtype, placement, or byte size is invalid"));
  }

  auto loan = output.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::span<const std::int64_t>{impl_->output_shape},
      .dtype = kFloat32Dtype,
  });
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (writer->byte_size() != impl_->output_bytes || writer->data() == nullptr ||
      !impl_->context->setTensorAddress(impl_->input_name.c_str(),
                                        const_cast<void*>(tensor.data())) ||
      !impl_->context->setTensorAddress(impl_->output_name.c_str(), writer->data()) ||
      !impl_->context->enqueueV3(context.cuda_stream().get())) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "TensorRT enqueueV3 failed"});
  }
  if (const cudaError_t status = cudaPeekAtLastError(); status != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        std::string("TensorRT CUDA launch failed: ") + cudaGetErrorString(status)});
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return output.emit(std::move(*loan), forwarded_emit_options(sample->metadata));
}

}  // namespace holoscan::examples::v4l2_depth
