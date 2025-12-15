
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include "nrrd_loader.hpp"

#include <array>
#include <filesystem>
#include <string>

#include <zlib.h>

#include "volume.hpp"

namespace holoscan::ops {

std::string remove_all_spaces(const std::string& str) {
  std::string result(str);
  result.erase(
      std::remove_if(result.begin(), result.end(), [](unsigned char x) { return std::isspace(x); }),
      result.end());
  return result;
}

std::string trim(const std::string& str) {
  std::string result(str);
  // remove leading spaces
  auto it = result.begin();
  while ((it != result.end()) && (std::isspace(*it))) {
    it = result.erase(it);
  }
  // remove trailing spaces
  it = --result.end();
  while ((it != result.begin()) && (std::isspace(*it))) {
    it = result.erase(it);
  }
  return result;
}

std::vector<double> parse_vector(std::string str) {
  std::vector<double> result;
  // ensures the string is surround by parenthesis and then remove them
  if ((str[0] != '(') || (str[str.length() - 1] != ')')) {
    throw std::runtime_error("parse_vector: string is not surrounded by (matching) parenthesis");
  }

  str = str.substr(1, str.length() - 2);

  std::stringstream ss(str);
  std::string token;
  while (std::getline(ss, token, ',')) {
    result.push_back(std::stod(token));
  }
  return result;
}

std::vector<std::string> split_string_by_space(std::string str) {
  std::vector<std::string> result;
  size_t start_index = 0;
  while (true) {
    while ((start_index < str.length()) && isspace(str[start_index])) {
      start_index++;
    }
    if (start_index >= str.length()) return result;
    size_t end_index = start_index;
    while ((end_index < str.length()) && !isspace(str[end_index])) {
      end_index++;
    }
    result.push_back(str.substr(start_index, end_index - start_index));
    start_index = end_index;
  }
}

bool is_nrrd(const std::string& file_name) {
  std::filesystem::path path(file_name);

  if (path.extension() == ".nhdr" || path.extension() == ".nrrd") {
    return true;
  }

  return false;
}

bool load_nrrd_data(bool compressed, std::ifstream& file, const size_t data_size,
                    const std::unique_ptr<uint8_t>& data) {
  if (compressed) {
    // need to uncompress, first read to 'compressed_data' vector and then uncompress to 'data'
    const std::streampos data_start = file.tellg();
    file.seekg(0, std::ios_base::end);
    const std::streampos file_size = file.tellg() - data_start;
    file.seekg(data_start);
    std::vector<uint8_t> compressed_data(file_size);

    // read
    file.read(reinterpret_cast<char*>(compressed_data.data()), compressed_data.size());

    // uncompress
    z_stream strm{};
    int result = inflateInit2(&strm, 32 + MAX_WBITS);
    if (result != Z_OK) {
      holoscan::log_error("NRRD failed to uncompress, inflateInit2 failed with error code {}",
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
      holoscan::log_error("NRRD failed to uncompress, inflate failed with error code {}", result);
      return false;
    }

  } else {
    file.read(reinterpret_cast<char*>(data.get()), data_size);
  }

  return true;
}

bool parse_headers(const std::string& file_name, const std::string& key, const std::string& value,
                   bool& compressed, std::array<int32_t, 3>& dims, Volume& volume,
                   nvidia::gxf::PrimitiveType& primitive_type, std::string& data_file_name) {
  if (key == "dimension") {
    int dims = std::stoi(value);
    if (dims != 3) {
      holoscan::log_error("NRRD expected a three dimensional input, instead NDims is {}", dims);
      return false;
    }
  } else if (key == "encoding") {
    if ((value == "gz") || (value == "gzip")) {
      compressed = true;
    } else if (value == "raw") {
      compressed = false;
    } else {
      holoscan::log_error("NRRD unexpected value for {}: {}", key, value);
      return false;
    }
  } else if (key == "sizes") {
    std::stringstream value_stream(value);
    std::string value;
    for (int index = 0; std::getline(value_stream, value, ' ') && (index < 3); ++index) {
      dims[2 - index] = std::stoi(value);
    }
  } else if (key == "spacings") {
    std::stringstream value_stream(value);
    std::string value;
    for (int index = 0; std::getline(value_stream, value, ' ') && (index < 3); ++index) {
      volume.spacing_[index] = std::stof(value);
    }
  } else if (key == "type") {
    if ((value == "signed char") || (value == "int8") || (value == "int8_t")) {
      primitive_type = nvidia::gxf::PrimitiveType::kInt8;
    } else if ((value == "uchar") || (value == "unsigned char") || (value == "uint8") ||
               (value == "uint8_t")) {
      primitive_type = nvidia::gxf::PrimitiveType::kUnsigned8;
    } else if ((value == "short") || (value == "short int") || (value == "signed short") ||
               (value == "signed short int") || (value == "int16") || (value == "int16_t")) {
      primitive_type = nvidia::gxf::PrimitiveType::kInt16;
    } else if ((value == "ushort") || (value == "unsigned short") ||
               (value == "unsigned short int") || (value == "uint16") || (value == "uint16_t")) {
      primitive_type = nvidia::gxf::PrimitiveType::kUnsigned16;
    } else if ((value == "int") || (value == "signed int") || (value == "int32") ||
               (value == "int32_t")) {
      primitive_type = nvidia::gxf::PrimitiveType::kInt32;
    } else if ((value == "uint") || (value == "unsigned int") || (value == "uint32") ||
               (value == "uint32_t")) {
      primitive_type = nvidia::gxf::PrimitiveType::kUnsigned32;
    } else if (value == "float") {
      primitive_type = nvidia::gxf::PrimitiveType::kFloat32;
    } else {
      holoscan::log_error("NRRD unexpected value for {}: {}", key, value);
      return false;
    }
  } else if (key == "datafile") {
    const std::string path = file_name.substr(0, file_name.find_last_of("/\\") + 1);
    data_file_name = path + value;
  } else if (key == "space") {
    std::string orientation;
    if (value == "3D-right-handed") {
      orientation = "RAS";
    } else if (value == "3D-left-handed") {
      orientation = "LAS";
    } else {
      std::stringstream values(value);
      std::string space;
      while (std::getline(values, space, '-')) {
        if (space == "left") {
          orientation += "L";
        } else if (space == "right") {
          orientation += "R";
        } else if (space == "anterior") {
          orientation += "A";
        } else if (space == "posterior") {
          orientation += "P";
        } else if (space == "superior") {
          orientation += "S";
        } else if (space == "inferior") {
          orientation += "I";
        } else {
          holoscan::log_error("NRRD unexpected space string {}", space);
          return false;
        }
      }
    }
    volume.SetOrientation(orientation);
  } else if (key == "spaceorigin") {
    auto space_origin = parse_vector(value);
    std::copy_n(space_origin.begin(), 3, volume.space_origin_.begin());
  } else if (key == "spacedirections") {
    auto values = split_string_by_space(value);
    for (int index = 0; (index < values.size()) && (index < 3); ++index) {
      auto space_directions = parse_vector(values[index]);
      volume.spacing_[index] = std::sqrt(space_directions[0] * space_directions[0] +
                                         space_directions[1] * space_directions[1] +
                                         space_directions[2] * space_directions[2]);
    }
  }
  return true;
}

bool load_nrrd(const std::string& file_name, Volume& volume) {
  bool compressed = false;
  std::string data_file_name;
  nvidia::gxf::PrimitiveType primitive_type;
  std::array<int32_t, 3> dims;
  int byte_skip = 0;

  std::ifstream file;
  file.open(file_name, std::ios::in);
  if (!file.is_open()) {
    holoscan::log_error("NRRD could not open {}", file_name);
    return false;
  }
  // get the parameters
  std::string line;
  while (std::getline(file, line)) {
    if (file.tellg() != -1) {
      byte_skip = file.tellg();
    }

    // After the header, there is a single blank line containing zero characters. This separates the
    // header from the data, which follows
    if (line.empty()) {
      break;
    }

    // Comment lines start with a pound, "#", with no proceeding whitespace, ignore.
    if (line.substr(0, 1) == "#") {
      continue;
    }

    size_t delimiterPos = line.find(':');
    if (delimiterPos == std::string::npos) {
      continue;
    }
    std::string key = remove_all_spaces(line.substr(0, delimiterPos));
    std::string value = trim(line.substr(delimiterPos + 1));

    if (!parse_headers(
            file_name, key, value, compressed, dims, volume, primitive_type, data_file_name)) {
      // error already logged in the function
      return false;
    }
  }

  const size_t data_size =
      dims[0] * dims[1] * dims[2] * nvidia::gxf::PrimitiveTypeSize(primitive_type);
  std::unique_ptr<uint8_t> data(new uint8_t[data_size]);

  if (data_file_name.empty()) {
    // seek to start of data
    file.seekg(byte_skip, std::ios_base::beg);
  } else {
    // if there is a separate data file, open that
    file.open(data_file_name, std::ios::in);
  }

  if (!load_nrrd_data(compressed, file, data_size, data)) {
    holoscan::log_error("NRRD failed to load data {}",
                        data_file_name.empty() ? file_name : data_file_name);
    return false;
  }

  // allocate the tensor
  if (!volume.tensor_->reshapeCustom(nvidia::gxf::Shape(dims),
                                     primitive_type,
                                     nvidia::gxf::PrimitiveTypeSize(primitive_type),
                                     nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                     volume.storage_type_,
                                     volume.allocator_)) {
    holoscan::log_error("NRRD failed to reshape tensor");
    return false;
  }

  // copy the data
  switch (volume.storage_type_) {
    case nvidia::gxf::MemoryStorageType::kDevice:
      if (cudaMemcpy(volume.tensor_->pointer(),
                     reinterpret_cast<const void*>(data.get()),
                     data_size,
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        holoscan::log_error("NRRD failed to copy to GPU memory");
        return false;
      }
      break;
    case nvidia::gxf::MemoryStorageType::kHost:
    case nvidia::gxf::MemoryStorageType::kSystem:
      memcpy(volume.tensor_->pointer(), data.get(), data_size);
      break;
    default:
      holoscan::log_error("NRRD unhandled storage type {}", int(volume.storage_type_));
      return false;
  }

  return true;
}

}  // namespace holoscan::ops
