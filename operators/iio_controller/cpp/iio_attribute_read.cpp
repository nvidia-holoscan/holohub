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

#include "iio_attribute_read.hpp"
#include <gxf/core/gxf.h>
#include <iio.h>

using namespace holoscan::ops;

void IIOAttributeRead::setup(OperatorSpec& spec) {
  HOLOSCAN_LOG_INFO("IIOAttributeRead setup");
  spec.output<std::string>("value");

  spec.param<std::string>(ctx_p_, "ctx", "IIO Context", "The URI of the IIO Context");
  spec.param<std::string>(dev_p_, "dev", "IIO Device", "Name of the IIO Device", "");
  spec.param<std::string>(chan_p_, "chan", "IIO Channel", "Name of the IIO Channel", "");
  spec.param<bool>(channel_is_output_,
                   "channel_is_output",
                   "IIO Channel is Output",
                   "Is the channel output?",
                   false);
  spec.param<std::string>(attr_name_p_, "attr_name", "IIO Attribute", "Name of the IIO Attribute");
}

void IIOAttributeRead::initialize() {
  HOLOSCAN_LOG_INFO("IIOAttributeRead initialize");
  Operator::initialize();

  // Cannot work without the name of an attribute
  attr_type_ = attr_type_t::UNKNOWN;
  attr_name_ = attr_name_p_.get();

  // Cannot work without a context
  ctx_ = iio_create_context_from_uri(ctx_p_.get().c_str());
  if (!ctx_) {
    HOLOSCAN_LOG_ERROR("Failed to create IIO context from URI: {}", ctx_p_.get());
    ctx_creation_failed_ = true;
    return;
  }
  attr_type_ = attr_type_t::CONTEXT;

  if (!dev_p_.get().empty()) {
    dev_ = iio_context_find_device(ctx_, dev_p_.get().c_str());
    if (dev_ == nullptr) {
      HOLOSCAN_LOG_ERROR("Failed to find IIO device: {}", dev_p_.get());
      dev_not_found_ = true;
      return;
    }
    attr_type_ = attr_type_t::DEVICE;
  } else {
    dev_ = nullptr;
  }

  if (dev_ && !chan_p_.get().empty()) {
    chan_ = iio_device_find_channel(dev_, chan_p_.get().c_str(), channel_is_output_.get());
    if (chan_ == nullptr) {
      HOLOSCAN_LOG_ERROR(
          "Failed to find IIO channel: {} (output: {})", chan_p_.get(), channel_is_output_.get());
      chan_not_found_ = true;
      return;
    }
    attr_type_ = attr_type_t::CHANNEL;
  } else {
    chan_ = nullptr;
  }
}

void IIOAttributeRead::compute(InputContext&, OutputContext& op_output, ExecutionContext& context) {
  HOLOSCAN_LOG_DEBUG("IIOAttributeRead compute");

  // Check if initialization failed and interrupt graph execution
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
    HOLOSCAN_LOG_ERROR("Cannot proceed: IIO channel '{}' not found", chan_p_.get());
    GxfGraphInterrupt(context.context());
    return;
  }

  switch (attr_type_) {
    case attr_type_t::CONTEXT: {
      HOLOSCAN_LOG_DEBUG("Context attribute");

      uint ctx_attr_index = 0;
      uint ctx_attr_count = iio_context_get_attrs_count(ctx_);
      const char* value;

      for (ctx_attr_index = 0; ctx_attr_index < ctx_attr_count; ctx_attr_index++) {
        value = iio_context_get_attr_value(ctx_, attr_name_.c_str());
        if (value == nullptr) {
          HOLOSCAN_LOG_ERROR("Failed to get context attribute value");
          return;
        }

        HOLOSCAN_LOG_DEBUG("ctx attr: {} = {}", attr_name_, value);
        op_output.emit(std::string(value), "value");
        return;
      }
      break;
    }
    case attr_type_t::DEVICE: {
      HOLOSCAN_LOG_DEBUG("Device attribute");
      // In the current API we don't know the type of the attribute, so we try all possibilities
      ret = iio_device_attr_read(dev_, attr_name_.c_str(), buffer, sizeof(buffer));
      if (ret < 0) {
        HOLOSCAN_LOG_DEBUG("Failed to read device attribute {}", attr_name_);
      } else {
        op_output.emit(std::string(buffer), "value");
        return;
      }

      ret = iio_device_debug_attr_read(dev_, attr_name_.c_str(), buffer, sizeof(buffer));
      if (ret < 0) {
        HOLOSCAN_LOG_DEBUG("Failed to read device debug attribute {}", attr_name_);
      } else {
        op_output.emit(std::string(buffer), "value");
        return;
      }

      ret = iio_device_buffer_attr_read(dev_, attr_name_.c_str(), buffer, sizeof(buffer));
      if (ret < 0) {
        HOLOSCAN_LOG_DEBUG("Failed to read device buffer attribute {}", attr_name_);
      } else {
        op_output.emit(std::string(buffer), "value");
        return;
      }

      break;
    }
    case attr_type_t::CHANNEL: {
      HOLOSCAN_LOG_DEBUG("Channel attribute");
      // This is a channel attribute (or mistake)
      ret = iio_channel_attr_read(chan_, attr_name_p_.get().c_str(), buffer, sizeof(buffer));
      if (ret < 0) {
        HOLOSCAN_LOG_DEBUG("Failed to read channel attribute {}", attr_name_p_.get());
        return;
      }

      op_output.emit(std::string(buffer), "value");
      break;
    }
    default:
      HOLOSCAN_LOG_ERROR("Unknown attribute type");
      return;
  }
}
