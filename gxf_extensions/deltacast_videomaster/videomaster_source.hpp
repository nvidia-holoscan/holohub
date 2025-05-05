/*
 * Copyright (c) 2022, DELTACAST.TV.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef NVIDIA_HOLOSCAN_GXF_EXTENSIONS_VIDEOMASTER_SOURCE_HPP_
#define NVIDIA_HOLOSCAN_GXF_EXTENSIONS_VIDEOMASTER_SOURCE_HPP_

#include <array>
#include <string>
#include <vector>

#include "gxf/std/transmitter.hpp"
#include "videomaster_base.hpp"

namespace nvidia {
namespace holoscan {
namespace videomaster {

class VideoMasterSource : public VideoMasterBase {
 public:
  VideoMasterSource();

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

  gxf_result_t start() override;
  gxf_result_t tick() override;

 private:
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> _signal;

  gxf::Expected<void> transmit_buffer_data(void* buffer, uint32_t buffer_size);
  gxf::Parameter<uint32_t> _width;
  gxf::Parameter<uint32_t> _height;
  gxf::Parameter<bool> _progressive;
  gxf::Parameter<uint32_t> _framerate;
};

}  // namespace videomaster
}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_HOLOSCAN_GXF_EXTENSIONS_VIDEOMASTER_SOURCE_HPP
