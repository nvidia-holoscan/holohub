/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef MAIN_CPP
#define MAIN_CPP

#include <string>

#include <gxf/core/expected.hpp>
#include <gxf/std/ipc_server.hpp>
#include <holoscan/core/executors/gxf/gxf_executor.hpp>
#include <holoscan/holoscan.hpp>
// #include <grpcpp/grpcpp.h>#include <holoscan/holoscan.hpp>

#include "client_fragment.cc"
#include "server_fragment.cc"

using namespace holoscan;
using namespace holoscan::grpc_hello_world;
using namespace nvidia::gxf;

namespace holoscan {
namespace grpc_hello_world {

class App : public holoscan::Application {
 public:
  void compose() override {
    // IPCServer::Service service_sample_action = {
    //     "hello_world_service",
    //     IPCServer::kAction,
    //     {.action = std::bind(
    //          &App::hello_world_service, this, std::placeholders::_1, std::placeholders::_2)}};

    // auto gxf_executor = dynamic_cast<holoscan::gxf::GXFExecutor&>(executor());
    // gxf_executor.register_ipc_service(service_sample_action);
    auto grpc_server = make_fragment<GrpcServerFragment>("server", "0.0.0.0:50051");

    auto grpc_client = make_fragment<GrpcClientFragment>("client", "localhost:50051");

    add_flow(grpc_client, grpc_server, {{"say_my_name", "say_hello"}});
  }

  //  private:
  //   Expected<void> hello_world_service(const std::string& resource, const std::string& data) {
  //     std::cout << "Hello World" << std::endl;
  //     return nvidia::gxf::Success;
  //   }
};

}  // namespace grpc_hello_world
}  // namespace holoscan

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();
  app->run();
  return 0;
}
#endif /* MAIN_CPP */
