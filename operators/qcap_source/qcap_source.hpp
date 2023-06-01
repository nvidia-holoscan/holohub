/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 YUAN High-Tech Development Co., Ltd. All rights reserved.
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
#ifndef HOLOSCAN_OPERATORS_QCAP_SOURCE_QCAP_SOURCE_HPP
#define HOLOSCAN_OPERATORS_QCAP_SOURCE_QCAP_SOURCE_HPP

#include "holoscan/core/gxf/gxf_operator.hpp"

#include <string>
#include <utility>
#include <vector>

namespace holoscan::ops {

/**
 * @brief Operator class to get the video stream from YUAN Hight-Tech capture card.
 *
 * This wraps a GXF Codelet(`yuan::holoscan::QCAPSource`).
 */
class QCAPSourceOp : public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(QCAPSourceOp, holoscan::ops::GXFOperator)

  QCAPSourceOp() = default;

  const char* gxf_typename() const override { return "yuan::holoscan::QCAPSource"; }

  void setup(OperatorSpec& spec) override;

  void initialize() override;

 private:
  Parameter<holoscan::IOSpec*> video_buffer_output_;
  Parameter<std::string> device_specifier_;
  Parameter<uint32_t> channel_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<uint32_t> framerate_;
  Parameter<bool> use_rdma_;
  Parameter<std::string> pixel_format_;
  Parameter<std::string> input_type_;
  Parameter<uint32_t> mst_mode_;
  Parameter<uint32_t> sdi12g_mode_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_QCAP_SOURCE_QCAP_SOURCE_HPP */
