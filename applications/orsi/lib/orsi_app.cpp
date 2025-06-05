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


#ifdef USE_VIDEOMASTER
#include <videomaster_source.hpp>
#endif

#ifdef AJA_SOURCE
#include <aja_source.hpp>
#endif

#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

#include <getopt.h>


void OrsiApp::set_source(const std::string& source) {
#ifdef USE_VIDEOMASTER
  if (source == "videomaster") { video_source_ = VideoSource::VIDEOMASTER; }
#endif
  if (source == "replayer") { video_source_ = VideoSource::REPLAYER; }
#ifdef AJA_SOURCE
  if (source == "aja" ) { video_source_ = VideoSource::AJA; }
#endif
}

void OrsiApp::set_datapath(const std::string& path) {
  datapath = path;
}

bool OrsiApp::init(int argc, char** argv) {
  std::string data_path = "";
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name, data_path)) { return false; }

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

void OrsiApp::initVideoSource(const  std::shared_ptr<holoscan::CudaStreamPool>& cuda_stream_pool) {
    using namespace holoscan;

    const bool use_rdma = from_config("external_source.rdma").as<bool>();
    const bool overlay_enabled = (video_source_ != VideoSource::REPLAYER) &&
                    from_config("external_source.enable_overlay").as<bool>();

    std::shared_ptr<Resource> allocator_resource =
          make_resource<UnboundedAllocator>("unbounded_allocator");

    switch (video_source_) {
#ifdef USE_VIDEOMASTER
      case VideoSource::VIDEOMASTER:
        source = make_operator<ops::VideoMasterSourceOp>(
            "videomaster", from_config("videomaster"), Arg("pool") = allocator_resource);
        break;
#endif
#ifdef AJA_SOURCE
      case VideoSource::AJA:
        width = from_config("aja.width").as<uint32_t>();
        height = from_config("aja.height").as<uint32_t>();
        source = make_operator<ops::AJASourceOp>("aja",
         from_config("aja"), from_config("external_source"));
       break;
#endif
      default:
        source = make_operator<ops::VideoStreamReplayerOp>(
            "replayer", from_config("replayer"), Arg("directory", datapath));
        break;
    }

    if (video_source_ == VideoSource::AJA
#ifdef USE_VIDEOMASTER
        || video_source_ == VideoSource::VIDEOMASTER
#endif
    ) {
      video_format_converter_in_tensor_name = "source_video";
      std::string yaml_config = "drop_alpha_channel_aja";
#ifdef USE_VIDEOMASTER
      if (video_source_ == VideoSource::VIDEOMASTER) {
        yaml_config = "drop_alpha_channel_videomaster";
      }
#endif
      uint64_t drop_alpha_block_size = width * height * n_channels * bpp;
      uint64_t drop_alpha_num_blocks = 2;
      drop_alpha_channel = make_operator<ops::orsi::FormatConverterOp>(
          "drop_alpha_channel",
          from_config(yaml_config),
          Arg("allocator") = make_resource<BlockMemoryPool>(
              "pool", 1, drop_alpha_block_size, drop_alpha_num_blocks),
          Arg("cuda_stream_pool") = cuda_stream_pool);
    }

    switch (video_source_) {
#ifdef AJA_SOURCE
      case VideoSource::AJA:
        video_buffer_out = "video_buffer_output";
        break;
#endif
#ifdef USE_VIDEOMASTER
      case VideoSource::VIDEOMASTER:
        video_buffer_out = "signal";
        break;
#endif
      case VideoSource::REPLAYER:
      default:
        break;
    }
}

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path) {
  static struct option long_options[] = {{"data", required_argument, 0, 'd'}, {0, 0, 0, 0}};

  while (int c = getopt_long(argc, argv, "d", long_options, NULL)) {
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

  if (optind < argc) { config_name = argv[optind++]; }
  return true;
}
