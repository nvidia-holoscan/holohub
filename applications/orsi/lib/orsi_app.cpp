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

#include "orsi_app.hpp"
#include <getopt.h>

void OrsiApp::set_source(const std::string& source) {
  #ifdef USE_VIDEOMASTER
  if (source == "videomaster") { video_source_ = VideoSource::VIDEOMASTER; }
  #endif
  if (source == "replayer") { video_source_ = VideoSource::REPLAYER; }
}

void OrsiApp::set_datapath(const std::string& path) {
     datapath = path;
}

bool OrsiApp::init(int argc, char **argv)
{
  std::string data_path = "";
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name, data_path)) {
    return false;
  }

  if (config_name != "") {
    this->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/app_config.yaml";
    this->config(config_path);
  }

  auto source = this->from_config("source").as<std::string>();
  this->set_source(source);

  if (data_path != "") this->set_datapath(data_path);

  return true;
}

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path) {
  static struct option long_options[] = {
      {"data",    required_argument, 0,  'd' },
      {0,         0,                 0,  0 }
  };

  while (int c = getopt_long(argc, argv, "d",
                   long_options, NULL))  {
    if (c == -1 || c == '?') break;

    switch (c) {
      case 'd':
        data_path = optarg;
        break;
      default:
        std::cout << "Unknown arguments returned: " << c << std::endl;
        return false;
    }
  }

  if (optind < argc) {
    config_name = argv[optind++];
  }
  return true;
}