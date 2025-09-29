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

#include "gst_sink_resource.hpp"
#include <gst/gst.h>

extern "C" {
  GType gst_simple_custom_sink_get_type(void);
}

namespace holoscan {

GstSinkResource::~GstSinkResource() {
  HOLOSCAN_LOG_DEBUG("Destroying GstSinkResource");
  if (sink_element_ && GST_IS_ELEMENT(sink_element_)) {
    gst_element_set_state(sink_element_, GST_STATE_NULL);
    gst_object_unref(sink_element_);
    sink_element_ = nullptr;
  }
  HOLOSCAN_LOG_DEBUG("GstSinkResource destroyed");
}

void GstSinkResource::initialize() {
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

void GstSinkResource::set_save_buffers(bool save_buffers) {
  save_buffers_ = save_buffers;
  if (sink_element_) {
    g_object_set(sink_element_, "save-frames", save_buffers_, nullptr);
  }
}

void GstSinkResource::set_output_dir(const std::string& output_dir) {
  output_dir_ = output_dir;
  if (sink_element_) {
    g_object_set(sink_element_, "output-dir", output_dir_.c_str(), nullptr);
  }
}

void GstSinkResource::set_data_rate(double data_rate) {
  data_rate_ = data_rate;
  if (sink_element_) {
    g_object_set(sink_element_, "fps", data_rate_, nullptr);
  }
}

uint32_t GstSinkResource::get_buffer_count() const {
  if (!sink_element_) {
    return 0;
  }
  
  // This would require exposing frame_count as a property in the sink
  // For now, return 0 as a placeholder
  return 0;
}


void GstSinkResource::configure_properties() {
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

}  // namespace holoscan
