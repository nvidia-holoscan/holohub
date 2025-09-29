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

#ifndef GST_SINK_RESOURCE_HPP
#define GST_SINK_RESOURCE_HPP

#include <memory>
#include <string>

#include <gst/gst.h>
#include <holoscan/holoscan.hpp>

namespace holoscan {

/**
 * @brief Holoscan Resource wrapper for GstSimpleCustomSink
 * 
 * This class provides a Holoscan Resource interface to the custom GStreamer sink,
 * allowing it to be used seamlessly within Holoscan applications and operators.
 */
class GstSinkResource : public holoscan::Resource {
 public:
  using SharedPtr = std::shared_ptr<GstSinkResource>;

  /**
   * @brief Default constructor
   */
  GstSinkResource() = default;

  /**
   * @brief Constructor with sink name
   * @param sink_name Name for the GStreamer element instance
   */
  explicit GstSinkResource(const std::string& sink_name)
      : sink_name_(sink_name) {}

  /**
   * @brief Constructor with full configuration
   * @param sink_name Name for the GStreamer element instance
   * @param save_buffers Whether to save received buffers to files
   * @param output_dir Directory to save buffers (if save_buffers is true)
   * @param data_rate Target data rate for display purposes
   */
  GstSinkResource(const std::string& sink_name, 
                  bool save_buffers = false,
                  const std::string& output_dir = "/tmp",
                  double data_rate = 30.0)
      : sink_name_(sink_name),
        save_buffers_(save_buffers),
        output_dir_(output_dir),
        data_rate_(data_rate) {}

  // Move semantics
  GstSinkResource(GstSinkResource&& other) noexcept = default;
  GstSinkResource& operator=(GstSinkResource&& other) noexcept = default;

  /**
   * @brief Destructor - cleans up GStreamer resources
   */
  ~GstSinkResource();

  /**
   * @brief Initialize the GStreamer sink resource
   */
  void initialize() override;

  /**
   * @brief Check if the resource is valid and ready to use
   * @return true if the sink element is created and ready
   */
  bool valid() const {
    return sink_element_ != nullptr;
  }

  /**
   * @brief Get the underlying GStreamer element
   * @return Pointer to the GstElement (do not unref manually)
   */
  GstElement* get_element() const {
    return sink_element_;
  }

  /**
   * @brief Get the sink name
   * @return The name of the sink element
   */
  const std::string& get_sink_name() const {
    return sink_name_;
  }

  /**
   * @brief Set whether to save buffers to files
   * @param save_buffers true to enable buffer saving
   */
  void set_save_buffers(bool save_buffers);

  /**
   * @brief Get current save buffers setting
   * @return true if buffer saving is enabled
   */
  bool get_save_buffers() const {
    return save_buffers_;
  }

  /**
   * @brief Set output directory for saved buffers
   * @param output_dir Path to directory for saving buffers
   */
  void set_output_dir(const std::string& output_dir);

  /**
   * @brief Get current output directory
   * @return Path to output directory
   */
  const std::string& get_output_dir() const {
    return output_dir_;
  }

  /**
   * @brief Set target data rate
   * @param data_rate Target data rate per second
   */
  void set_data_rate(double data_rate);

  /**
   * @brief Get current data rate setting
   * @return Current data rate setting
   */
  double get_data_rate() const {
    return data_rate_;
  }

  /**
   * @brief Get statistics about processed buffers
   * @return Number of buffers processed so far
   */
  uint32_t get_buffer_count() const;


 private:
  std::string sink_name_;
  bool save_buffers_ = false;
  std::string output_dir_ = "/tmp";
  double data_rate_ = 30.0;
  
  GstElement* sink_element_ = nullptr;

  /**
   * @brief Configure the sink element properties
   */
  void configure_properties();
};

using GstSinkResourcePtr = GstSinkResource::SharedPtr;

}  // namespace holoscan

#endif /* GST_SINK_RESOURCE_HPP */
