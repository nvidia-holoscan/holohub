/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <array>
#include <filesystem>

#include <zlib.h>

#include "volume.hpp"

namespace holoscan::ops {

bool is_mhd(const std::string& file_name) {
  std::filesystem::path path(file_name);

  if (path.extension() == ".mhd") {
    return true;
  }

  return false;
}

bool load_mhd(const std::string& file_name, Volume& volume) {
  bool compressed = false;
  std::string data_file_name;
  nvidia::gxf::PrimitiveType primitive_type;
  std::array<int32_t, 3> dims;

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
      while ((it != value.end()) && (std::isspace(*it))) {
        it = value.erase(it);
      }

      if (parameter == "NDims") {
        int dims = std::stoi(value);
        if (dims != 3) {
          holoscan::log_error("MHD expected a three dimensional input, instead NDims is {}", dims);
          return false;
        }
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
        std::stringstream value_stream(value);
        std::string value;
        for (int index = 0; std::getline(value_stream, value, ' ') && (index < 3); ++index) {
          dims[2 - index] = std::stoi(value);
        }
      } else if (parameter == "ElementSpacing") {
        std::stringstream value_stream(value);
        std::string value;
        for (int index = 0; std::getline(value_stream, value, ' ') && (index < 3); ++index) {
          volume.spacing_[index] = std::stof(value);
        }
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
      } else if (parameter == "ElementDataFile") {
        const std::string path = file_name.substr(0, file_name.find_last_of("/\\") + 1);
        data_file_name = path + value;
      } else if (parameter == "AnatomicalOrientation") {
        volume.SetOrientation(value);
      }
    }
  }

  const size_t data_size =
      dims[0] * dims[1] * dims[2] * nvidia::gxf::PrimitiveTypeSize(primitive_type);
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
    file.read(reinterpret_cast<char*>(data.get()), data_size);
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
