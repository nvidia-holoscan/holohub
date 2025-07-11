/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/core/resources/gxf/cuda_stream_pool.hpp>
#include <holoscan/holoscan.hpp>
// Orsi: Holoscan native operators
#include <format_converter.hpp>

#include <string>

enum class VideoSource {
  REPLAYER,
#ifdef USE_VIDEOMASTER
  VIDEOMASTER,
#endif
  AJA
};

class OrsiApp : public holoscan::Application {
 protected:
  std::string datapath = "data";

  // video source members
  VideoSource video_source_ = VideoSource::REPLAYER;
  std::shared_ptr<holoscan::Operator> source;
  uint32_t width = 1920;
  uint32_t height = 1080;
  const int n_channels = 4;
  const int bpp = 4;
  std::string video_format_converter_in_tensor_name = "";
  std::string video_buffer_out = "";

  // Alpha Channel Op needed for Videomaster and AJA video sources
  std::shared_ptr<holoscan::Operator> drop_alpha_channel;

  // initialize video
  void initVideoSource(const std::shared_ptr<holoscan::CudaStreamPool>& cuda_stream_pool);

 public:
  void set_source(const std::string& source);
  void set_datapath(const std::string& path);
  bool init(int argc, char** argv);
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path);
