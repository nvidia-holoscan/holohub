/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Analog Devices, Inc. All rights reserved.
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

#include "iio_buffer_read.hpp"
#include <gxf/core/gxf.h>
#include <iio.h>
#include <cstring>
#include "iio_params.hpp"

using namespace holoscan::ops;

void IIOBufferRead::stop() {
  HOLOSCAN_LOG_INFO("IIOBufferRead stop");
  if (buffer_) {
    iio_buffer_destroy(buffer_);
    buffer_ = nullptr;
  }
  if (ctx_) {
    iio_context_destroy(ctx_);
    ctx_ = nullptr;
  }
}

void IIOBufferRead::setup(OperatorSpec& spec) {
  HOLOSCAN_LOG_INFO("IIOBufferRead setup");
  spec.output<std::shared_ptr<iio_buffer_info_t>>("buffer");

  spec.param<std::string>(ctx_p_, "ctx", "IIO Context", "The URI of the IIO Context");
  spec.param<std::string>(dev_p_, "dev", "IIO Device", "Name of the IIO Device");
  spec.param<bool>(is_cyclic, "is_cyclic", "Is cyclic", "Is the buffer cyclic?");
  spec.param<size_t>(
      samples_count_p_, "samples_count", "Samples count", "Number of samples to read");
  spec.param<std::vector<std::string>>(
      enabled_channel_names_p_,
      "enabled_channel_names",
      "Names of the enabled IIO Channels",
      "The names of the channels that are enabled when pushing the buffer");
  spec.param<std::vector<bool>>(
      enabled_channel_types_p_,
      "enabled_channel_output",
      "Types of the enabled IIO Channels",
      "The types of the channels that are enabled when pushing the buffer");
}

void IIOBufferRead::initialize() {
  HOLOSCAN_LOG_INFO("IIOBufferRead initialize");
  Operator::initialize();

  if (ctx_p_.get().empty()) {
    HOLOSCAN_LOG_ERROR("IIO Context is not set. Cannot use operator.");
    ctx_empty_ = true;
    return;
  }

  if (dev_p_.get().empty()) {
    HOLOSCAN_LOG_ERROR("IIO Device is not set. Cannot use operator.");
    dev_empty_ = true;
    return;
  }

  if (enabled_channel_names_p_.get().empty() || enabled_channel_types_p_.get().empty()) {
    HOLOSCAN_LOG_ERROR(
        "It is mandatory to enable at least one channel before creating the "
        "IIO buffer.");
    channels_empty_ = true;
    return;
  }

  if (samples_count_p_.get() == 0) {
    HOLOSCAN_LOG_ERROR("Samples count is not set. Cannot use operator.");
    samples_count_zero_ = true;
    return;
  }

  ctx_ = iio_create_context_from_uri(ctx_p_.get().c_str());
  if (!ctx_) {
    HOLOSCAN_LOG_ERROR("Failed to create IIO context from URI: {}", ctx_p_.get());
    ctx_creation_failed_ = true;
    return;
  }

  dev_ = iio_context_find_device(ctx_, dev_p_.get().c_str());
  if (!dev_) {
    HOLOSCAN_LOG_ERROR("Failed to find IIO device: {}", dev_p_.get());
    dev_not_found_ = true;
    return;
  }

  std::vector<std::string>& enabled_channel_names = enabled_channel_names_p_.get();
  std::vector<bool>& enabled_channel_types = enabled_channel_types_p_.get();
  for (size_t i = 0; i < enabled_channel_names.size(); ++i) {
    std::string& chn_name = enabled_channel_names[i];
    bool chn_type = enabled_channel_types[i];
    iio_channel* chn = iio_device_find_channel(dev_, chn_name.c_str(), chn_type);
    if (!chn) {
      HOLOSCAN_LOG_ERROR("Failed to find {} channel {}", chn_type ? "output" : "input", chn_name);
      chan_not_found_ = true;
      return;
    }

    if (!iio_channel_is_enabled(chn)) {
      HOLOSCAN_LOG_INFO("Enabled channel: {}", chn_name);
      iio_channel_enable(chn);
    } else {
      HOLOSCAN_LOG_INFO("Channel {} is already enabled", chn_name);
    }
  }

  ssize_t sample_size = iio_device_get_sample_size(dev_);
  if (sample_size < 0) {
    HOLOSCAN_LOG_ERROR(
        "Failed to get sample size from device {}; Err code: {}", dev_p_.get(), sample_size);
    sample_size_failed_ = true;
    return;
  }
  sample_size_ = static_cast<size_t>(sample_size);

  buffer_ = nullptr;
}

void IIOBufferRead::compute(InputContext&, OutputContext& op_output, ExecutionContext& context) {
  HOLOSCAN_LOG_DEBUG("IIOBufferRead compute");

  // Check if initialization failed and interrupt graph execution
  if (ctx_empty_) {
    HOLOSCAN_LOG_ERROR("Cannot proceed: IIO Context is not set");
    GxfGraphInterrupt(context.context());
    return;
  }
  if (dev_empty_) {
    HOLOSCAN_LOG_ERROR("Cannot proceed: IIO Device is not set");
    GxfGraphInterrupt(context.context());
    return;
  }
  if (channels_empty_) {
    HOLOSCAN_LOG_ERROR("Cannot proceed: No channels enabled for IIO buffer");
    GxfGraphInterrupt(context.context());
    return;
  }
  if (samples_count_zero_) {
    HOLOSCAN_LOG_ERROR("Cannot proceed: Samples count is not set");
    GxfGraphInterrupt(context.context());
    return;
  }
  if (ctx_creation_failed_) {
    HOLOSCAN_LOG_ERROR("Cannot proceed: IIO context creation failed for URI: {}", ctx_p_.get());
    GxfGraphInterrupt(context.context());
    return;
  }
  if (dev_not_found_) {
    HOLOSCAN_LOG_ERROR("Cannot proceed: IIO device '{}' not found", dev_p_.get());
    GxfGraphInterrupt(context.context());
    return;
  }
  if (chan_not_found_) {
    HOLOSCAN_LOG_ERROR("Cannot proceed: One or more IIO channels not found");
    GxfGraphInterrupt(context.context());
    return;
  }
  if (sample_size_failed_) {
    HOLOSCAN_LOG_ERROR("Cannot proceed: Failed to get sample size from device '{}'", dev_p_.get());
    GxfGraphInterrupt(context.context());
    return;
  }
  auto buffer_info = std::shared_ptr<iio_buffer_info_t>(new iio_buffer_info_t);
  buffer_info->buffer = nullptr;
  buffer_info->samples_count = 0;
  buffer_info->is_cyclic = is_cyclic.get();
  buffer_info->device_name = dev_p_.get();

  // Populate enabled channels information
  std::vector<std::string>& enabled_channel_names = enabled_channel_names_p_.get();
  std::vector<bool>& enabled_channel_types = enabled_channel_types_p_.get();
  for (size_t i = 0; i < enabled_channel_names.size(); ++i) {
    // Get the channel to retrieve complete info
    iio_channel* chn =
        iio_device_find_channel(dev_, enabled_channel_names[i].c_str(), enabled_channel_types[i]);

    if (chn) {
      // Use the helper function to create channel info
      iio_channel_info_t chan_info = create_channel_info_from_iio_channel(chn);
      buffer_info->enabled_channels.push_back(chan_info);
    } else {
      HOLOSCAN_LOG_WARN("Failed to find channel {} for info retrieval", enabled_channel_names[i]);
      // Create a default channel info with the provided name and type
      iio_channel_info_t chan_info;
      chan_info.name = enabled_channel_names[i];
      chan_info.is_output = enabled_channel_types[i];
      chan_info.index = 0;
      memset(&chan_info.format, 0, sizeof(struct iio_data_format));
      buffer_info->enabled_channels.push_back(chan_info);
    }
  }

  if (!buffer_) {
    HOLOSCAN_LOG_INFO(
        "Creating buffer with {} samples of size {} bytes", samples_count_p_.get(), sample_size_);
    buffer_ = iio_device_create_buffer(dev_, samples_count_p_.get(), false);
    if (!buffer_) {
      HOLOSCAN_LOG_ERROR("Failed to create buffer, error code {}", errno);
      GxfGraphInterrupt(context.context());
      return;
    }
  }

  ssize_t res = iio_buffer_refill(buffer_);
  if (res < 0) {
    HOLOSCAN_LOG_ERROR("Failed to refill buffer. Error code: {}", res);
    return;
  }

  size_t bytes_read = static_cast<size_t>(res);
  buffer_info->samples_count = bytes_read / sample_size_;

  void* buffer_data = iio_buffer_start(buffer_);
  buffer_info->buffer = new int8_t[bytes_read];
  memcpy(buffer_info->buffer, buffer_data, bytes_read);

  op_output.emit(buffer_info, "buffer");
}
