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

#pragma once
#include <iio.h>
#include <cstring>
#include <string>
#include <vector>

enum class attr_type_t {
  CONTEXT,
  DEVICE,
  CHANNEL,
  UNKNOWN,
};

struct iio_channel_info_t {
  std::string name;
  bool is_output;
  unsigned int index;             // Channel index
  struct iio_data_format format;  // Channel data format
};

struct iio_buffer_info_t {
  size_t samples_count;
  void* buffer;
  std::vector<iio_channel_info_t> enabled_channels;
  bool is_cyclic;
  std::string device_name;
};

/**
 * @brief Creates an iio_channel_info_t structure from an IIO channel pointer
 *
 * This helper function simplifies the creation of channel information structures
 * by automatically extracting all relevant data from an IIO channel, including
 * the channel name, direction (input/output), index, and data format.
 *
 * This is particularly useful when:
 * - Building buffer information for IIOBufferWrite operations
 * - Creating channel metadata for custom operators
 * - Needing complete channel information without manual field population
 *
 * @param channel Pointer to the IIO channel (must not be null)
 * @return iio_channel_info_t Complete channel information structure
 *
 * @note If the channel pointer is null, returns a default-initialized structure
 * @note If channel index or format retrieval fails, those fields are set to default values
 */
inline iio_channel_info_t create_channel_info_from_iio_channel(const struct iio_channel* channel) {
  iio_channel_info_t info;

  if (!channel) {
    // Return default-initialized struct if channel is null
    info.name = "";
    info.is_output = false;
    info.index = 0;
    memset(&info.format, 0, sizeof(struct iio_data_format));
    return info;
  }

  // Get channel name
  const char* name = iio_channel_get_id(channel);
  info.name = name ? name : "";

  // Get channel direction
  info.is_output = iio_channel_is_output(channel);

  // Get channel index
  long idx = iio_channel_get_index(channel);
  info.index = (idx >= 0) ? static_cast<unsigned int>(idx) : 0;

  // Get channel data format
  const struct iio_data_format* fmt = iio_channel_get_data_format(channel);
  if (fmt) {
    info.format = *fmt;
  } else {
    memset(&info.format, 0, sizeof(struct iio_data_format));
  }

  return info;
}
