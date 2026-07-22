/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
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

#include "mhd_loader.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <sstream>

#include <zlib.h>

#include "volume.hpp"

namespace holoscan::ops {

namespace {

template <typename T, size_t N>
bool parse_exact_vector(const std::string& value, std::array<T, N>& result) {
  std::istringstream value_stream(value);
  std::array<T, N> parsed;

  for (T& component : parsed) {
    if (!(value_stream >> component)) { return false; }
  }

  value_stream >> std::ws;
  if (!value_stream.eof()) { return false; }

  result = parsed;
  return true;
}

}  // namespace

bool is_mhd(const std::string& file_name) {
  std::filesystem::path path(file_name);

  if (path.extension() == ".mhd") { return true; }

  return false;
}

bool load_mhd(const std::string& file_name, Volume& volume) {
  bool compressed = false;
  bool has_dims = false;
  bool has_element_type = false;
  bool has_ndims = false;
  std::string data_file_name;
  nvidia::gxf::PrimitiveType primitive_type{};
  std::array<int32_t, 3> dims{};

  {
    std::stringstream meta_header;
    {
      std::ifstream file;
      file.open(file_name, std::ios::in);
      if (!file.is_open()) {
        holoscan::log_error("MHD could not open {}", file_name);
        return false;
      }
      meta_header << file.rdbuf();
    }
    // get the parameters
    std::string parameter;
    while (std::getline(meta_header, parameter, '=')) {
      // remove spaces
      parameter.erase(
          std::remove_if(
              parameter.begin(), parameter.end(), [](unsigned char x) { return std::isspace(x); }),
          parameter.end());

      // get the value
      std::string value;
      std::getline(meta_header, value);
      // remove leading spaces
      auto it = value.begin();
      while ((it != value.end()) && (std::isspace(*it))) { it = value.erase(it); }

      if (parameter == "NDims") {
        std::istringstream value_stream(value);
        int dimensions;
        if (!(value_stream >> dimensions)) {
          holoscan::log_error("MHD invalid NDims value");
          return false;
        }
        value_stream >> std::ws;
        if (!value_stream.eof()) {
          holoscan::log_error("MHD invalid NDims value");
          return false;
        }
        if (dimensions != 3) {
          holoscan::log_error(
              "MHD expected a three dimensional input, instead NDims is {}", dimensions);
          return false;
        }
        has_ndims = true;
      } else if (parameter == "CompressedData") {
        if (value == "True") {
          compressed = true;
        } else if (value == "False") {
          compressed = false;
        } else {
          holoscan::log_error("MHD unexpected value for {}: {}", parameter, value);
          return false;
        }
      } else if (parameter == "DimSize") {
        std::array<int32_t, 3> parsed_dims;
        if (!parse_exact_vector(value, parsed_dims)) {
          holoscan::log_error("MHD DimSize must contain exactly three integers");
          return false;
        }
        if (std::any_of(
                parsed_dims.begin(), parsed_dims.end(), [](int32_t dim) { return dim <= 0; })) {
          holoscan::log_error("MHD DimSize values must be positive");
          return false;
        }
        dims = {parsed_dims[2], parsed_dims[1], parsed_dims[0]};
        has_dims = true;
      } else if (parameter == "ElementSpacing") {
        std::array<float, 3> spacing;
        if (!parse_exact_vector(value, spacing)) {
          holoscan::log_error("MHD ElementSpacing must contain exactly three numbers");
          return false;
        }
        if (std::any_of(spacing.begin(), spacing.end(), [](float value) {
              return !std::isfinite(value) || value <= 0.f;
            })) {
          holoscan::log_error("MHD ElementSpacing values must be finite and positive");
          return false;
        }
        volume.spacing_ = spacing;
      } else if (parameter == "ElementType") {
        if (value == "MET_CHAR") {
          primitive_type = nvidia::gxf::PrimitiveType::kInt8;
        } else if (value == "MET_UCHAR") {
          primitive_type = nvidia::gxf::PrimitiveType::kUnsigned8;
        } else if (value == "MET_SHORT") {
          primitive_type = nvidia::gxf::PrimitiveType::kInt16;
        } else if (value == "MET_USHORT") {
          primitive_type = nvidia::gxf::PrimitiveType::kUnsigned16;
        } else if (value == "MET_INT") {
          primitive_type = nvidia::gxf::PrimitiveType::kInt32;
        } else if (value == "MET_UINT") {
          primitive_type = nvidia::gxf::PrimitiveType::kUnsigned32;
        } else if (value == "MET_FLOAT") {
          primitive_type = nvidia::gxf::PrimitiveType::kFloat32;
        } else {
          holoscan::log_error("MHD unexpected value for {}: {}", parameter, value);
          return false;
        }
        has_element_type = true;
      } else if (parameter == "ElementDataFile") {
        const std::string path = file_name.substr(0, file_name.find_last_of("/\\") + 1);
        data_file_name = path + value;
      } else if (parameter == "AnatomicalOrientation") {
        volume.SetOrientation(value);
      }
    }
  }

  if (!has_ndims || !has_dims || !has_element_type || data_file_name.empty()) {
    holoscan::log_error(
        "MHD is missing one or more required fields: NDims, DimSize, ElementType, ElementDataFile");
    return false;
  }

  size_t data_size = nvidia::gxf::PrimitiveTypeSize(primitive_type);
  for (const int32_t dim : dims) {
    if (data_size > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      holoscan::log_error("MHD volume size exceeds the supported range");
      return false;
    }
    data_size *= static_cast<size_t>(dim);
  }
  std::unique_ptr<uint8_t> data(new uint8_t[data_size]);

  std::ifstream file;

  file.open(data_file_name, std::ios::in | std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    holoscan::log_error("MHD could not open {}", data_file_name);
    return false;
  }
  const std::streampos file_size = file.tellg();
  file.seekg(0, std::ios_base::beg);

  if (compressed) {
    // need to uncompress, first read to 'compressed_data' vector and then uncompress to 'data'
    std::vector<uint8_t> compressed_data(file_size);

    // read
    file.read(reinterpret_cast<char*>(compressed_data.data()), compressed_data.size());

    // uncompress
    z_stream strm{};
    int result = inflateInit2(&strm, 32 + MAX_WBITS);
    if (result != Z_OK) {
      holoscan::log_error("MHD failed to uncompress {}, inflateInit2 failed with error code {}",
                          data_file_name,
                          result);
      return false;
    }

    strm.next_in = compressed_data.data();
    strm.avail_in = compressed_data.size();
    strm.next_out = data.get();
    strm.avail_out = data_size;

    result = inflate(&strm, Z_FINISH);
    inflateEnd(&strm);
    if (result != Z_STREAM_END) {
      holoscan::log_error(
          "MHD failed to uncompress {}, inflate failed with error code {}", data_file_name, result);
      return false;
    }
  } else {
    if (data_size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
      holoscan::log_error("MHD raw payload size exceeds the supported range");
      return false;
    }

    const auto expected_size = static_cast<std::streamsize>(data_size);
    file.read(reinterpret_cast<char*>(data.get()), expected_size);
    if (!file || file.gcount() != expected_size) {
      holoscan::log_error(
          "MHD raw payload {} is truncated: expected {} bytes, read {} bytes",
          data_file_name,
          data_size,
          file.gcount());
      return false;
    }
  }

  // allocate the tensor
  if (!volume.tensor_->reshapeCustom(nvidia::gxf::Shape(dims),
                                     primitive_type,
                                     nvidia::gxf::PrimitiveTypeSize(primitive_type),
                                     nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                     volume.storage_type_,
                                     volume.allocator_)) {
    holoscan::log_error("MHD failed to reshape tensor");
    return false;
  }

  // copy the data
  switch (volume.storage_type_) {
    case nvidia::gxf::MemoryStorageType::kDevice:
      if (cudaMemcpy(volume.tensor_->pointer(),
                     reinterpret_cast<const void*>(data.get()),
                     data_size,
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        holoscan::log_error("MHD failed to copy to GPU memory");
        return false;
      }
      break;
    case nvidia::gxf::MemoryStorageType::kHost:
    case nvidia::gxf::MemoryStorageType::kSystem:
      memcpy(volume.tensor_->pointer(), data.get(), data_size);
      break;
    default:
      holoscan::log_error("MHD unhandled storage type {}", int(volume.storage_type_));
      return false;
  }

  return true;
}

}  // namespace holoscan::ops
