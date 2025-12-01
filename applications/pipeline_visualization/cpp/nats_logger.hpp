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

#ifndef PIPELINE_VISUALIZATION_CPP_NATS_LOGGER_HPP
#define PIPELINE_VISUALIZATION_CPP_NATS_LOGGER_HPP

#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/io_spec.hpp>
#include <holoscan/core/metadata.hpp>
#include <holoscan/core/resources/async_data_logger.hpp>

#include <memory>
#include <string>

namespace holoscan {
// Forward declarations
class Tensor;
class TensorMap;
}  // namespace holoscan

namespace holoscan::data_loggers {

/**
 * @brief NATS-based data logger resource for Holoscan applications.
 *
 * This class provides a data logger implementation that publishes data to NATS subjects.
 * It inherits from DataLoggerResource and implements the required logging methods.
 */
class NatsLogger : public holoscan::AsyncDataLoggerResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(NatsLogger, AsyncDataLoggerResource)

  NatsLogger() = default;

  /**
   * @brief Setup the NATS logger with configuration parameters.
   *
   * @param spec The component specification to configure.
   */
  void setup(ComponentSpec& spec) override;

  /**
   * @brief Initialize the NATS connection and subscriber.
   */
  void initialize() override;

  /**
   * @brief Log generic data to NATS.
   *
   * @param data The data to log.
   * @param unique_id Unique identifier for the message.
   * @param acquisition_timestamp Timestamp when data was acquired.
   * @param metadata Associated metadata dictionary.
   * @param io_type The type of I/O port.
   * @param stream Optional CUDA stream for GPU operations.
   * @return true if logging was successful, false otherwise.
   */
  bool log_data(const std::any& data, const std::string& unique_id,
                int64_t acquisition_timestamp = -1,
                const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                IOSpec::IOType io_type = IOSpec::IOType::kOutput,
                std::optional<cudaStream_t> stream = std::nullopt) override;

  /**
   * @brief Log tensor data to NATS.
   *
   * @param tensor The tensor to log.
   * @param unique_id Unique identifier for the message.
   * @param acquisition_timestamp Timestamp when data was acquired.
   * @param metadata Associated metadata dictionary.
   * @param io_type The type of I/O port.
   * @param stream Optional CUDA stream for GPU operations.
   * @return true if logging was successful, false otherwise.
   */
  bool log_tensor_data(const std::shared_ptr<Tensor>& tensor, const std::string& unique_id,
                       int64_t acquisition_timestamp = -1,
                       const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                       IOSpec::IOType io_type = IOSpec::IOType::kOutput,
                       std::optional<cudaStream_t> stream = std::nullopt) override;

  /**
   * @brief Log tensor map data to NATS.
   *
   * @param tensor_map The tensor map to log.
   * @param unique_id Unique identifier for the message.
   * @param acquisition_timestamp Timestamp when data was acquired.
   * @param metadata Associated metadata dictionary.
   * @param io_type The type of I/O port.
   * @param stream Optional CUDA stream for GPU operations.
   * @return true if logging was successful, false otherwise.
   */
  bool log_tensormap_data(const TensorMap& tensor_map, const std::string& unique_id,
                          int64_t acquisition_timestamp = -1,
                          const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                          IOSpec::IOType io_type = IOSpec::IOType::kOutput,
                          std::optional<cudaStream_t> stream = std::nullopt) override;

  /**
   * @brief Log backend-specific data to NATS.
   *
   * @param data The data to log.
   * @param unique_id Unique identifier for the message.
   * @param acquisition_timestamp Timestamp when data was acquired.
   * @param metadata Associated metadata dictionary.
   * @param io_type The type of I/O port.
   * @param stream Optional CUDA stream for GPU operations.
   * @return true if logging was successful, false otherwise.
   */
  bool log_backend_specific(const std::any& data, const std::string& unique_id,
                            int64_t acquisition_timestamp = -1,
                            const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                            IOSpec::IOType io_type = IOSpec::IOType::kOutput,
                            std::optional<cudaStream_t> stream = std::nullopt) override;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;

  // Configuration parameters
  Parameter<std::string> nats_url_;
  Parameter<std::string> subject_prefix_;
  Parameter<float> publish_rate_;

  class AsyncNatsBackend : public AsyncDataLoggerBackend {
   public:
    /**
     * @brief Constructor for AsyncNatsBackend.
     *
     * @param nats_logger Pointer to the parent NatsLogger instance
     */
    explicit AsyncNatsBackend(NatsLogger* nats_logger);
    AsyncNatsBackend() = delete;
    /**
     * @brief Initialize the async backend.
     *
     * @return true if initialization was successful
     */
    bool initialize() override;
    /**
     * @brief Shutdown the async backend.
     *
     * Performs any necessary cleanup when the backend is shutting down.
     */
    void shutdown() override;
    /**
     * @brief Process a data entry from the async queue.
     *
     * Handles different types of data entries (Generic, TensorData, TensorMapData)
     * by serializing them appropriately and publishing to NATS subjects.
     *
     * @param entry The data entry to process
     * @return true if processing was successful, false otherwise
     */
    bool process_data_entry(const DataEntry& entry) override;
    /**
     * @brief Process a large data entry from the async queue.
     *
     * Currently delegates to process_data_entry for handling. This method could be
     * extended to implement special handling for large data payloads if needed.
     *
     * @param entry The large data entry to process
     * @return true if processing was successful, false otherwise
     */
    bool process_large_data_entry(const DataEntry& entry) override;

   private:
    /**
     * @brief Pointer to the parent NatsLogger instance.
     *
     * This is used to access the NatsLogger instance and its configuration parameters.
     */
    NatsLogger* nats_logger_;
  };
};

}  // namespace holoscan::data_loggers

#endif  // PIPELINE_VISUALIZATION_CPP_NATS_LOGGER_HPP
