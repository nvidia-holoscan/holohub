/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/holoscan.hpp>
#include <string>

enum class VideoSource { 
  REPLAYER, 
#ifdef USE_VIDEOMASTER
  VIDEOMASTER
#endif
};

class OrsiApp : public holoscan::Application {

protected:

  VideoSource video_source_ = VideoSource::REPLAYER;
  std::string datapath = "data";

public:
  void set_source(const std::string& source);
  void set_datapath(const std::string& path);
  bool init(int argc, char **argv);
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path);
