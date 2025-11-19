/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/holoscan.hpp>
#include <string>

namespace holoscan::apps {

/**
 * @brief Server application that processes endoscopy video and broadcasts results
 * 
 * This application:
 * - Replays endoscopy video from disk
 * - Runs LSTM inference for tool tracking
 * - Displays processed results locally with HolovizOp
 * - Broadcasts processed frames to connected clients via UCXX
 */
class UcxxEndoscopyServerApp : public holoscan::Application {
 public:
  void set_datapath(const std::string& path) { datapath_ = path; }
  void set_hostname(const std::string& hostname) { hostname_ = hostname; }
  void set_port(int port) { port_ = port; }

  void compose() override;

 private:
  std::string datapath_ = "";
  std::string hostname_ = "0.0.0.0";
  int port_ = 50008;
};

}  // namespace holoscan::apps

