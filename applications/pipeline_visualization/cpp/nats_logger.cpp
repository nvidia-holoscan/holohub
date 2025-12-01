/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nats_logger.hpp"

#include <cstring>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/domain/tensor_map.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/logger/logger.hpp>

#include <nats.h>

#include <magic_enum.hpp>
#include <stdexcept>

#include <flatbuffers/message_generated.h>
#include <flatbuffers/tensor_generated.h>

#include "create_tensor.hpp"
#include "holoscan/core/resources/async_data_logger.hpp"

namespace holoscan::data_loggers {

/**
 * @brief Macro to wrap NATS API calls and handle errors.
 *
 * This macro executes a NATS API call and checks its return status.
 * If the call fails (status != NATS_OK), it throws a runtime_error with
 * detailed information about the failure including the statement, line number,
 * file, and error description.
 */
#define NATS_CALL(stmt, ...)                                                        \
  ({                                                                                \
    natsStatus _status = stmt;                                                      \
    if (NATS_OK != _status) {                                                       \
      throw std::runtime_error(                                                     \
          fmt::format("NATS call {} in line {} of file {} failed with '{}': '{}'.", \
                      #stmt,                                                        \
                      __LINE__,                                                     \
                      __FILE__,                                                     \
                      magic_enum::enum_name(_status),                               \
                      natsStatus_GetText(_status)));                                \
    }                                                                               \
  })

/**
 * @brief Custom deleter template for NATS objects.
 *
 * This template provides automatic cleanup for NATS API objects by wrapping
 * their destruction functions. It's used with std::unique_ptr to ensure
 * proper RAII (Resource Acquisition Is Initialization) semantics.
 *
 * @tparam T The type of the NATS object pointer
 * @tparam func The cleanup function to call when the object is destroyed
 */
template <typename T, void func(T)>
struct Deleter {
  typedef T pointer;
  /**
   * @brief Operator to call the cleanup function
   *
   * @param value The NATS object to clean up
   */
  void operator()(T value) const { func(value); }
};

// Type aliases for smart pointers to NATS objects with automatic cleanup
using UniqueNatsOptions =
    std::unique_ptr<natsOptions*, Deleter<natsOptions*, &natsOptions_Destroy>>;
using UniqueNatsConnection =
    std::unique_ptr<natsConnection*, Deleter<natsConnection*, &natsConnection_Destroy>>;
using UniqueNatsSubscription =
    std::unique_ptr<natsSubscription*, Deleter<natsSubscription*, &natsSubscription_Destroy>>;

/**
 * @brief Private implementation class for NatsLogger (PIMPL pattern).
 *
 * This class encapsulates the NATS-specific implementation details,
 * managing the connection, subscription, and message handling logic.
 */
class NatsLogger::Impl {
 public:
  /**
   * @brief Error handler callback for NATS connection errors.
   *
   * This static method is registered with the NATS connection to handle
   * any errors that occur during communication with the NATS server.
   *
   * @param nc The NATS connection that encountered the error
   * @param subscription The subscription associated with the error (if any)
   * @param err The error status code
   * @param closure User-provided closure data (unused)
   */
  static void error_handler(natsConnection* nc, natsSubscription* subscription, natsStatus err,
                            void* closure) {
    HOLOSCAN_LOG_ERROR("NATS error '{}' {}", magic_enum::enum_name(err), natsStatus_GetText(err));
  }

  /**
   * @brief Initialize NATS connection.
   *
   * Establishes a connection to the NATS server with the specified URL.
   * Configures the connection with error handling and optimized send options.
   *
   * @param nats_url The URL of the NATS server (e.g., "nats://localhost:4222")
   * @throws std::runtime_error if connection fails
   */
  void connect_to_nats(const std::string& nats_url) {
    // Create NATS options object with automatic cleanup
    UniqueNatsOptions nats_options;
    nats_options.reset([]() -> natsOptions* {
      natsOptions* nats_options = nullptr;
      NATS_CALL(natsOptions_Create(&nats_options));
      return nats_options;
    }());

    // Configure connection options
    NATS_CALL(natsOptions_SetURL(nats_options.get(), nats_url.c_str()));
    NATS_CALL(natsOptions_SetErrorHandler(nats_options.get(), &error_handler, nullptr));

    // Send the data right away, reducing latency
    // The alternative would be to call natsConnection_Flush(),
    // but this call requires a round-trip with the server, which is less
    // efficient than using this option.
    NATS_CALL(natsOptions_SetSendAsap(nats_options.get(), true));

    // Establish connection to NATS server with configured options
    nats_connection_.reset([&nats_options]() -> natsConnection* {
      natsConnection* nats_connection = nullptr;
      NATS_CALL(natsConnection_Connect(&nats_connection, nats_options.get()));
      return nats_connection;
    }());
  }

  /**
   * @brief Callback handler for incoming NATS messages.
   *
   * This static method is invoked when a message is received on a subscribed subject.
   * It extracts the message data and logs it for debugging purposes.
   *
   * @param nc The NATS connection that received the message
   * @param sub The subscription that received the message
   * @param msg The received message (must be destroyed after processing)
   * @param closure User-provided closure data (unused)
   */
  static void message_handler(natsConnection* nc, natsSubscription* sub, natsMsg* msg,
                              void* closure) {
    // Extract message metadata
    std::string subject = natsMsg_GetSubject(msg);
    const char* reply_cstr = natsMsg_GetReply(msg);
    std::string reply = reply_cstr ? reply_cstr : "";
    const char* data = natsMsg_GetData(msg);
    int data_len = natsMsg_GetDataLength(msg);

    // Log the received message
    std::string data_str(data, data_len);
    HOLOSCAN_LOG_INFO("Received a message on '{} {}': {}", subject, reply, data_str);

    // Clean up the message to prevent memory leaks
    natsMsg_Destroy(msg);
  }

  /**
   * @brief Setup NATS subscriber for incoming messages.
   *
   * Creates a subscription to the specified NATS subject. Messages received
   * on this subject will be handled by the message_handler callback.
   *
   * @param subscribe_subject The NATS subject to subscribe to
   * @throws std::runtime_error if NATS connection is not established
   */
  void setup_subscriber(const std::string& subscribe_subject) {
    if (!nats_connection_) {
      throw std::runtime_error("NATS connection not established");
    }

    // Create subscription with message handler callback
    subscriber_.reset([this, &subscribe_subject]() -> natsSubscription* {
      natsSubscription* subscriber = nullptr;
      NATS_CALL(natsConnection_Subscribe(
          &subscriber, nats_connection_.get(), subscribe_subject.c_str(), message_handler, this));
      return subscriber;
    }());
  }

  /**
   * @brief Publish data to NATS subject.
   *
   * This method throws on unexpected errors like NATS issues.
   *
   * @param subject The NATS subject to publish to.
   * @param data The data to publish.
   * @param data_size Size of the data in bytes.
   * @return true if publish was successful, false otherwise.
   */
  bool publish_data(const std::string& subject, const void* data, size_t data_size) {
    // Verify connection is available before attempting to publish
    if (!nats_connection_) {
      HOLOSCAN_LOG_ERROR("NATS connection not available for publishing");
      return false;
    }

    // Publish the data to the specified NATS subject
    NATS_CALL(natsConnection_Publish(nats_connection_.get(), subject.c_str(), data, data_size));

    return true;
  }

  /**
   * @brief Check if the message should be logged.
   *
   * @param logger The logger instance.
   * @param io_type The type of I/O port.
   * @param unique_id The unique identifier for the message.
   * @param timestamp_ns The timestamp in nanoseconds.
   * @return true if the message should be logged, false otherwise.
   */
  bool should_log_message(NatsLogger* logger, IOSpec::IOType io_type, const std::string& unique_id,
                          int64_t timestamp_ns) {
    // Check filtering conditions based on I/O type
    if ((io_type == IOSpec::IOType::kOutput) && !logger->should_log_output()) {
      return false;
    }
    if ((io_type == IOSpec::IOType::kInput) && !logger->should_log_input()) {
      return false;
    }
    // Check if this specific message ID should be logged
    if (!logger->should_log_message(unique_id)) {
      return false;
    }

    // First message with this ID - allow logging
    if (last_publish_time_.find(unique_id) == last_publish_time_.end()) {
      last_publish_time_[unique_id] = timestamp_ns;
      return true;
    }

    // Rate limiting: Check if enough time has passed since last publish
    // Convert publish_rate (Hz) to nanosecond interval
    const float rate_hz = logger->publish_rate_.get();
    if (rate_hz <= 0.0F) {
      last_publish_time_[unique_id] = timestamp_ns;
      return true;
    }
    const int64_t period_ns = static_cast<int64_t>(1'000'000'000.0F / rate_hz);
    if (timestamp_ns - last_publish_time_[unique_id] >= period_ns) {
      last_publish_time_[unique_id] = timestamp_ns;
      return true;
    }
    return false;
  }

  // NATS connection and subscription objects with automatic cleanup
  UniqueNatsConnection nats_connection_;  ///< Connection to the NATS server
  UniqueNatsSubscription subscriber_;     ///< Subscription for receiving control messages

  /// Map tracking the last publish timestamp (in nanoseconds) for each unique message ID
  /// Used for rate limiting to prevent excessive publishing
  std::map<std::string, int64_t> last_publish_time_;

  std::shared_ptr<AsyncNatsBackend> backend_;
};

/**
 * @brief Setup method called during component initialization.
 *
 * Configures the NatsLogger with parameters from the component specification.
 * Creates the implementation object and registers NATS-specific parameters.
 *
 * @param spec The component specification to configure parameters
 */
void NatsLogger::setup(ComponentSpec& spec) {
  // Call parent setup for common data logger parameters
  holoscan::AsyncDataLoggerResource::setup(spec);

  // Create the implementation object (PIMPL pattern)
  assert(!impl_);
  impl_ = std::make_shared<Impl>();
  impl_->backend_ = std::make_shared<AsyncNatsBackend>(this);

  set_backend(impl_->backend_);

  // Register NATS-specific configuration parameters
  spec.param(
      nats_url_, "nats_url", "NATS URL", "NATS server URL", std::string("nats://0.0.0.0:4222"));
  spec.param(subject_prefix_, "subject_prefix", "Subject Prefix", "NATS subject prefix");
  spec.param(publish_rate_, "publish_rate", "Publish Rate", "Publish rate in Hz", 5.0f);
}

/**
 * @brief Initialize the NATS logger.
 *
 * Establishes connection to the NATS server and sets up the control message subscriber.
 * This method is called after setup() during component initialization.
 */
void NatsLogger::initialize() {
  // Call parent initialize for base functionality
  holoscan::AsyncDataLoggerResource::initialize();

  // Establish connection to NATS server
  impl_->connect_to_nats(nats_url_.get());
  // Subscribe to control messages on the configured subject
  impl_->setup_subscriber(subject_prefix_.get() + ".control");
}

bool NatsLogger::log_data(const std::any& data, const std::string& unique_id,
                          int64_t acquisition_timestamp,
                          const std::shared_ptr<MetadataDictionary>& metadata,
                          IOSpec::IOType io_type, std::optional<cudaStream_t> stream) {
  const int64_t timestamp_ns = get_timestamp();
  // Check if this message should be logged based on filters and rate limiting
  if (!impl_->should_log_message(this, io_type, unique_id, timestamp_ns)) {
    return true;
  }
  return AsyncDataLoggerResource::log_data(
      data, unique_id, acquisition_timestamp, metadata, io_type, stream);
}

bool NatsLogger::log_tensor_data(const std::shared_ptr<Tensor>& tensor,
                                 const std::string& unique_id, int64_t acquisition_timestamp,
                                 const std::shared_ptr<MetadataDictionary>& metadata,
                                 IOSpec::IOType io_type, std::optional<cudaStream_t> stream) {
  const int64_t timestamp_ns = get_timestamp();
  // Check if this message should be logged based on filters and rate limiting
  if (!impl_->should_log_message(this, io_type, unique_id, timestamp_ns)) {
    return true;
  }

  return AsyncDataLoggerResource::log_tensor_data(
      tensor, unique_id, acquisition_timestamp, metadata, io_type, stream);
}

bool NatsLogger::log_tensormap_data(const TensorMap& tensor_map, const std::string& unique_id,
                                    int64_t acquisition_timestamp,
                                    const std::shared_ptr<MetadataDictionary>& metadata,
                                    IOSpec::IOType io_type, std::optional<cudaStream_t> stream) {
  const int64_t timestamp_ns = get_timestamp();
  // Check if this message should be logged based on filters and rate limiting
  if (!impl_->should_log_message(this, io_type, unique_id, timestamp_ns)) {
    return true;
  }

  return AsyncDataLoggerResource::log_tensormap_data(
      tensor_map, unique_id, acquisition_timestamp, metadata, io_type, stream);
}

bool NatsLogger::log_backend_specific(const std::any& data, const std::string& unique_id,
                                      int64_t acquisition_timestamp,
                                      const std::shared_ptr<MetadataDictionary>& metadata,
                                      IOSpec::IOType io_type, std::optional<cudaStream_t> stream) {
  // Check for empty data
  if (!data.has_value()) {
    return true;
  }

  // metadata is often updated in-place, so we need to make a copy here at the time of logging if
  // it is to be logged.
  std::shared_ptr<MetadataDictionary> metadata_copy;
  if (metadata) {
    metadata_copy = std::make_shared<MetadataDictionary>(*metadata);
  }

  // Runtime type checking for GXF Entity types
  static const std::type_index nvidia_gxf_entity_type(typeid(nvidia::gxf::Entity));
  static const std::type_index holoscan_gxf_entity_type(typeid(holoscan::gxf::Entity));

  std::type_index data_type_index(data.type());
  bool is_nvidia_entity = (data_type_index == nvidia_gxf_entity_type);
  bool is_holoscan_entity = (data_type_index == holoscan_gxf_entity_type);

  if (is_nvidia_entity || is_holoscan_entity) {
    std::string entity_type_str =
        is_nvidia_entity ? "nvidia::gxf::Entity" : "holoscan::gxf::Entity";

    // Extract tensor components from the entity and log them
    try {
      nvidia::gxf::Entity gxf_entity;

      if (is_nvidia_entity) {
        gxf_entity = std::any_cast<nvidia::gxf::Entity>(data);
      } else {
        // holoscan::gxf::Entity inherits from nvidia::gxf::Entity
        auto holoscan_entity = std::any_cast<holoscan::gxf::Entity>(data);
        gxf_entity = static_cast<nvidia::gxf::Entity>(std::move(holoscan_entity));
      }

      // Note: This function currently logs Tensors and VideoBuffers via separate data logging
      // calls, so if both types are present in the entity, there will be separate log entries for
      // each type. This can be revisited in the future as needed if we need to combine both into
      // a single log entry.

      // Find and log tensor components within the entity
      auto tensor_components_expected = gxf_entity.findAllHeap<nvidia::gxf::Tensor>();
      if (!tensor_components_expected) {
        HOLOSCAN_LOG_ERROR("{}: Failed to enumerate tensor components: {}",
                           name(),
                           GxfResultStr(tensor_components_expected.error()));
        return false;
      }
      if (!tensor_components_expected->empty()) {
        TensorMap tensor_map;
        for (const auto& gxf_tensor : tensor_components_expected.value()) {
          // Do zero-copy conversion to holoscan::Tensor
          auto maybe_dl_ctx = (*gxf_tensor->get()).toDLManagedTensorContext();
          if (!maybe_dl_ctx) {
            HOLOSCAN_LOG_ERROR(
                "{}: Failed to get std::shared_ptr<DLManagedTensorContext> from "
                "nvidia::gxf::Tensor",
                name());
            continue;
          }
          auto holoscan_tensor = std::make_shared<Tensor>(maybe_dl_ctx.value());
          tensor_map.insert({gxf_tensor->name(), holoscan_tensor});
        }

        if (tensor_map.size() > 0) {
          // Log the tensor map found in the entity
          if (!log_tensormap_data(
                  tensor_map, unique_id, acquisition_timestamp, metadata_copy, io_type, stream)) {
            HOLOSCAN_LOG_ERROR("{}: Logging of TensorMap data from Entity failed", name());
            return false;
          }
        }
      }

      // Find and log any VideoBuffer components within the entity
      auto video_buffer_components_expected = gxf_entity.findAllHeap<nvidia::gxf::VideoBuffer>();
      if (!video_buffer_components_expected) {
        HOLOSCAN_LOG_ERROR("{}: Failed to enumerate VideoBuffer components: {}",
                           name(),
                           GxfResultStr(video_buffer_components_expected.error()));
        return false;
      }
      if (!video_buffer_components_expected->empty()) {
        for (const auto& maybe_buffer_handle : video_buffer_components_expected.value()) {
          if (!maybe_buffer_handle) {
            continue;
          }
          auto buffer_handle = maybe_buffer_handle.value();
          if (!log_data(buffer_handle,
                        unique_id + "." + buffer_handle.name(),
                        acquisition_timestamp,
                        metadata_copy,
                        io_type,
                        stream)) {
            HOLOSCAN_LOG_ERROR("{}: Logging of VideoBuffer data from Entity failed", name());
            return false;
          }
        }
      }
      // TODO(unknown): handle any other component types we want to log (AudioBuffer, etc.)?
    } catch (const std::bad_any_cast& e) {
      HOLOSCAN_LOG_ERROR("{}: Failed to cast entity data to expected type '{}': {}",
                         name(),
                         entity_type_str,
                         e.what());
      return false;
    }

    return true;
  }

  HOLOSCAN_LOG_WARN(
      "Backend-specific data of type `{}` at `{}` not supported", data.type().name(), unique_id);

  return AsyncDataLoggerResource::log_backend_specific(
      data, unique_id, acquisition_timestamp, metadata, io_type, stream);
}

NatsLogger::AsyncNatsBackend::AsyncNatsBackend(NatsLogger* nats_logger)
    : nats_logger_(nats_logger) {}

bool NatsLogger::AsyncNatsBackend::initialize() {
  return true;
}

void NatsLogger::AsyncNatsBackend::shutdown() {}

bool NatsLogger::AsyncNatsBackend::process_data_entry(const DataEntry& entry) {
  if (entry.type == DataEntry::Type::Generic) {
    HOLOSCAN_LOG_WARN("Generic data of type `{}` at `{}` not supported",
                      std::get<std::any>(entry.data).type().name(),
                      entry.unique_id);
    return false;
  } else if (entry.type == DataEntry::Type::TensorData) {
    // Serializes tensor data using FlatBuffers format and publishes it to NATS.
    // The message includes metadata such as unique_id, timestamps, and I/O type.
    // Optionally includes the actual tensor content based on configuration.
    auto tensor = std::get<std::shared_ptr<Tensor>>(entry.data);
    ::flatbuffers::FlatBufferBuilder builder;
    auto offset = pipeline_visualization::flatbuffers::CreateMessage(
        builder,
        builder.CreateString(entry.unique_id),
        static_cast<pipeline_visualization::flatbuffers::IOType>(entry.io_type),
        entry.acquisition_timestamp,
        entry.emit_timestamp,
        nats_logger_->should_log_tensor_data_content()
            ? pipeline_visualization::flatbuffers::Payload::
                  Payload_pipeline_visualization_flatbuffers_Tensor
            : pipeline_visualization::flatbuffers::Payload::Payload_NONE,
        nats_logger_->should_log_tensor_data_content()
            ? pipeline_visualization::flatbuffers::CreateTensor(builder, tensor, entry.stream)
                  .Union()
            : 0);
    builder.Finish(offset);

    // Publish the serialized message to the data subject
    return nats_logger_->impl_->publish_data(nats_logger_->subject_prefix_.get() + ".data",
                                             builder.GetBufferPointer(),
                                             builder.GetSize());
  } else if (entry.type == DataEntry::Type::TensorMapData) {
    auto tensor_map = std::get<holoscan::TensorMap>(entry.data);
    for (const auto& [key, tensor] : tensor_map) {
      ::flatbuffers::FlatBufferBuilder builder;
      auto offset = pipeline_visualization::flatbuffers::CreateMessage(
          builder,
          builder.CreateString(entry.unique_id + "." + key),
          static_cast<pipeline_visualization::flatbuffers::IOType>(entry.io_type),
          entry.acquisition_timestamp,
          entry.emit_timestamp,
          nats_logger_->should_log_tensor_data_content()
              ? pipeline_visualization::flatbuffers::Payload::
                    Payload_pipeline_visualization_flatbuffers_Tensor
              : pipeline_visualization::flatbuffers::Payload::Payload_NONE,
          nats_logger_->should_log_tensor_data_content()
              ? pipeline_visualization::flatbuffers::CreateTensor(builder, tensor, entry.stream)
                    .Union()
              : 0);
      builder.Finish(offset);
      if (!nats_logger_->impl_->publish_data(nats_logger_->subject_prefix_.get() + ".data",
                                             builder.GetBufferPointer(),
                                             builder.GetSize())) {
        return false;
      }
    }
    return true;
  } else {
    HOLOSCAN_LOG_ERROR("AsyncNatsBackend: Unknown data type: {}",
                       magic_enum::enum_name(entry.type));
    return false;
  }
}

bool NatsLogger::AsyncNatsBackend::process_large_data_entry(const DataEntry& entry) {
  return process_data_entry(entry);
}

}  // namespace holoscan::data_loggers
