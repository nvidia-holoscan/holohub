// Copyright 2025 Analog Devices, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iio_configurator.hpp"
#include <iio.h>
#include <yaml-cpp/node/detail/iterator_fwd.h>
#include <gxf/core/gxf.h>
#include <holoscan/logger/logger.hpp>

using namespace holoscan::ops;

void IIOConfigurator::setup(OperatorSpec& spec) {
  HOLOSCAN_LOG_INFO("IIOConfigurator setup");
  // We will receive only a YAML file that we need to parse
  spec.param<std::string>(cfg_path_p_,
                          "cfg",
                          "Configuration file path",
                          "The path to the YAML configuration file for IIO setup.");
}

void IIOConfigurator::parse_setup(const YAML::Node& setup_node, iio_context* ctx) {
  HOLOSCAN_LOG_INFO("Parsing setup configuration");

  if (!setup_node["devices"]) {
    HOLOSCAN_LOG_ERROR("No devices found in setup configuration");
    return;
  }

  auto& devices = setup_node["devices"];
  for (const auto& device : devices) {
    if (device.IsMap()) {
      for (const auto& dev : device) {
        std::string dev_name = dev.first.as<std::string>();
        iio_device* iio_dev = iio_context_find_device(ctx, dev_name.c_str());
        if (!iio_dev) {
          HOLOSCAN_LOG_ERROR("Failed to find device: {}", dev_name);
          // Note: Using continue instead of stopping execution for missing devices in setup
          // as this may be expected for some configurations
          continue;
        }

        parse_device(dev.second, iio_dev);
      }
    } else {
      HOLOSCAN_LOG_ERROR("Invalid device configuration in setup");
    }
  }
}

void IIOConfigurator::parse_device(const YAML::Node& device_node, iio_device* dev) {
  HOLOSCAN_LOG_INFO("DEVICE: {}", iio_device_get_name(dev));
  if (device_node["attrs"]) {
    auto& attrs = device_node["attrs"];
    for (const auto& attr : attrs) { parse_attribute(attr, dev); }
  }

  if (device_node["debug-attrs"]) {
    auto& debug_attrs = device_node["debug-attrs"];
    for (const auto& attr : debug_attrs) { parse_attribute(attr, dev, IIODeviceAttrType::DEBUG); }
  }

  if (device_node["buffer-attrs"]) {
    auto& debug_attrs = device_node["buffer-attrs"];
    for (const auto& attr : debug_attrs) { parse_attribute(attr, dev, IIODeviceAttrType::BUFFER); }
  }

  if (device_node["channels"]) {
    auto& channels = device_node["channels"];
    auto& input_channels = channels["input"];
    auto& output_channels = channels["output"];

    if (input_channels) {
      for (const auto& input_channel : input_channels) {
        for (const auto& channel : input_channel) {
          std::string channel_name = channel.first.as<std::string>();
          iio_channel* chan = iio_device_find_channel(dev, channel_name.c_str(), false);
          if (!chan) {
            HOLOSCAN_LOG_ERROR("Failed to find input channel: {}", channel_name);
            // Note: Using continue instead of stopping execution for missing channels in setup
            // as this may be expected for some configurations
            continue;
          }

          HOLOSCAN_LOG_INFO("INPUT CHANNEL: {}", channel_name);
          parse_channel(channel.second, chan);
        }
      }
    }

    if (output_channels) {
      for (const auto& output_channel : output_channels) {
        for (const auto& channel : output_channel) {
          std::string channel_name = channel.first.as<std::string>();
          iio_channel* chan = iio_device_find_channel(dev, channel_name.c_str(), true);
          if (!chan) {
            HOLOSCAN_LOG_ERROR("Failed to find output channel: {}", channel_name);
            // Note: Using continue instead of stopping execution for missing channels in setup
            // as this may be expected for some configurations
            continue;
          }

          HOLOSCAN_LOG_INFO("OUTPUT CHANNEL: {}", channel_name);
          parse_channel(channel.second, chan);
        }
      }
    }
  }
}

void IIOConfigurator::parse_attribute(const YAML::Node& attr_node, iio_device* dev,
                                      IIODeviceAttrType type) {
  YAML::const_iterator it = attr_node.begin();  // There should only be one attribute
  if (it == attr_node.end()) {
    HOLOSCAN_LOG_ERROR("No attributes found in device configuration");
    return;
  }
  std::string attr_name = it->first.as<std::string>();
  std::string attr_value = it->second.as<std::string>();

  ssize_t res = 0;
  std::string type_str;

  switch (type) {
    case IIODeviceAttrType::DEVICE:
      res = iio_device_attr_write(dev, attr_name.c_str(), attr_value.c_str());
      break;
    case IIODeviceAttrType::DEBUG:
      res = iio_device_debug_attr_write(dev, attr_name.c_str(), attr_value.c_str());
      type_str = "debug";
      break;
    case IIODeviceAttrType::BUFFER:
      res = iio_device_buffer_attr_write(dev, attr_name.c_str(), attr_value.c_str());
      type_str = "buffer";
      break;
  }

  if (res < 0) {
    HOLOSCAN_LOG_ERROR("Failed to write {} attribute '{}' to device '{}'",
                       type_str.empty() ? "" : type_str,
                       attr_name,
                       iio_device_get_name(dev));
    return;
  } else {
    HOLOSCAN_LOG_INFO(
        "{}{} => {}", type_str.empty() ? "" : "(" + type_str + ") ", attr_name, attr_value);
  }

  ++it;  // Move to the next attribute (shouldn't be any)
  if (it != attr_node.end()) {
    HOLOSCAN_LOG_ERROR("Multiple attributes found in channel configuration, expected only one");
    return;
  }
}

void IIOConfigurator::parse_channel(const YAML::Node& channel_node, iio_channel* chan) {
  auto& channel_attrs = channel_node["attrs"];
  if (channel_attrs) {
    for (const auto& attr : channel_attrs) { parse_attribute(attr, chan); }
  }
}

void IIOConfigurator::parse_attribute(const YAML::Node& attr_node, iio_channel* chan) {
  YAML::const_iterator it = attr_node.begin();  // There should only be one attribute
  if (it == attr_node.end()) {
    HOLOSCAN_LOG_ERROR("No attributes found in channel configuration");
    return;
  }
  std::string attr_name = it->first.as<std::string>();
  std::string attr_value = it->second.as<std::string>();

  ssize_t res = iio_channel_attr_write(chan, attr_name.c_str(), attr_value.c_str());
  if (res < 0) {
    HOLOSCAN_LOG_ERROR(
        "Failed to write attribute '{}' to channel '{}'", attr_name, iio_channel_get_id(chan));
    return;
  } else {
    HOLOSCAN_LOG_INFO("{}/{} => {}", iio_channel_get_id(chan), attr_name, attr_value);
  }

  ++it;  // Move to the next attribute (shouldn't be any)
  if (it != attr_node.end()) {
    HOLOSCAN_LOG_ERROR("Multiple attributes found in channel configuration, expected only one");
    return;
  }
}

void IIOConfigurator::compute(InputContext&, OutputContext&, ExecutionContext& context) {
  HOLOSCAN_LOG_INFO("IIOConfigurator compute");

  auto config = holoscan::Config(cfg_path_p_.get());
  auto& yaml_nodes = config.yaml_nodes();
  bool found_cfg = false;
  for (const auto& node : yaml_nodes) {
    if (node["cfg"]) {
      found_cfg = true;
      auto& cfg = node["cfg"];
      std::string uri = cfg["uri"].as<std::string>();
      HOLOSCAN_LOG_INFO("IIOConfigurator URI: {}", uri);

      iio_context* ctx = iio_create_context_from_uri(uri.c_str());
      if (ctx == nullptr) {
        HOLOSCAN_LOG_ERROR("Failed to create IIO context from URI: {}", uri);
        HOLOSCAN_LOG_ERROR(
            "Cannot proceed: IIO context creation failed - stopping graph execution");
        GxfGraphInterrupt(context.context());
        return;
      }

      if (!cfg["setup"]) {
        HOLOSCAN_LOG_ERROR("No setup configuration found in YAML");
        HOLOSCAN_LOG_ERROR(
            "Cannot proceed: Missing setup configuration - stopping graph execution");
        GxfGraphInterrupt(context.context());
        return;
      }

      auto& setup = cfg["setup"];
      parse_setup(setup, ctx);
    }
  }

  if (!found_cfg) {
    HOLOSCAN_LOG_ERROR("No 'cfg' node found in the YAML configuration file");
    HOLOSCAN_LOG_ERROR("Cannot proceed: Missing configuration node - stopping graph execution");
    GxfGraphInterrupt(context.context());
    return;
  }
}
