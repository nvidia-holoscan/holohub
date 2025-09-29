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

extern "C" {
  GType gst_simple_custom_sink_get_type(void);
}

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
  ~GstSinkResource() {
    HOLOSCAN_LOG_DEBUG("Destroying GstSinkResource");
    if (sink_element_ && GST_IS_ELEMENT(sink_element_)) {
      gst_element_set_state(sink_element_, GST_STATE_NULL);
      gst_object_unref(sink_element_);
      sink_element_ = nullptr;
    }
    HOLOSCAN_LOG_DEBUG("GstSinkResource destroyed");
  }

  /**
   * @brief Initialize the GStreamer sink resource
   */
  void initialize() override {
    HOLOSCAN_LOG_INFO("Initializing GstSinkResource");
    
    // Initialize GStreamer if not already done
    if (!gst_is_initialized()) {
      gst_init(nullptr, nullptr);
    }

    // Register our custom sink element type
    gst_element_register(nullptr, "simplecustomsink", GST_RANK_NONE, 
                        gst_simple_custom_sink_get_type());

    // Create the sink element
    sink_element_ = gst_element_factory_make("simplecustomsink", 
                                           sink_name_.empty() ? nullptr : sink_name_.c_str());
    
    if (!sink_element_) {
      HOLOSCAN_LOG_ERROR("Failed to create GStreamer sink element");
      return;
    }

    // Configure properties
    configure_properties();

    HOLOSCAN_LOG_INFO("GstSinkResource initialized successfully");
  }

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
  void set_save_buffers(bool save_buffers) {
    save_buffers_ = save_buffers;
    if (sink_element_) {
      g_object_set(sink_element_, "save-frames", save_buffers_, nullptr);
    }
  }

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
  void set_output_dir(const std::string& output_dir) {
    output_dir_ = output_dir;
    if (sink_element_) {
      g_object_set(sink_element_, "output-dir", output_dir_.c_str(), nullptr);
    }
  }

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
  void set_data_rate(double data_rate) {
    data_rate_ = data_rate;
    if (sink_element_) {
      g_object_set(sink_element_, "fps", data_rate_, nullptr);
    }
  }

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
  uint32_t get_buffer_count() const {
    if (!sink_element_) {
      return 0;
    }
    
    // This would require exposing frame_count as a property in the sink
    // For now, return 0 as a placeholder
    return 0;
  }

  /**
   * @brief Create a pipeline with this sink as the endpoint
   * @param pipeline_description GStreamer pipeline description (without sink)
   * @return Pointer to created pipeline (caller owns the reference)
   */
  GstElement* create_pipeline(const std::string& pipeline_description) {
    if (!valid()) {
      HOLOSCAN_LOG_ERROR("Sink resource not initialized");
      return nullptr;
    }

    // Create pipeline
    GstElement* pipeline = gst_pipeline_new("holoscan-pipeline");
    if (!pipeline) {
      HOLOSCAN_LOG_ERROR("Failed to create pipeline");
      return nullptr;
    }

    // Parse the pipeline description to create a bin
    GError* error = nullptr;
    GstElement* source_bin = gst_parse_bin_from_description(pipeline_description.c_str(), TRUE, &error);
    if (error) {
      HOLOSCAN_LOG_ERROR("Failed to parse pipeline: {}", error->message);
      g_error_free(error);
      gst_object_unref(pipeline);
      return nullptr;
    }

    // Add elements to pipeline
    gst_bin_add_many(GST_BIN(pipeline), source_bin, sink_element_, nullptr);

    // Link the source bin to our sink
    if (!gst_element_link(source_bin, sink_element_)) {
      HOLOSCAN_LOG_ERROR("Failed to link pipeline elements to sink");
      gst_object_unref(pipeline);
      return nullptr;
    }

    return pipeline;
  }

 private:
  std::string sink_name_;
  bool save_buffers_ = false;
  std::string output_dir_ = "/tmp";
  double data_rate_ = 30.0;
  
  GstElement* sink_element_ = nullptr;

  /**
   * @brief Configure the sink element properties
   */
  void configure_properties() {
    if (!sink_element_) {
      return;
    }

    g_object_set(sink_element_,
                 "save-frames", save_buffers_,
                 "output-dir", output_dir_.c_str(),
                 "fps", data_rate_,
                 nullptr);

    HOLOSCAN_LOG_DEBUG("Configured sink properties: save-frames={}, output-dir={}, fps={}",
                      save_buffers_, output_dir_, data_rate_);
  }
};

using GstSinkResourcePtr = GstSinkResource::SharedPtr;

}  // namespace holoscan

#endif /* GST_SINK_RESOURCE_HPP */
