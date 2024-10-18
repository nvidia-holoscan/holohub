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

#ifndef HOLOSCAN_OPERATORS_AZURE_KINECT_CAMERA
#define HOLOSCAN_OPERATORS_AZURE_KINECT_CAMERA

#include "holoscan/core/operator.hpp"
#include "holoscan/holoscan.hpp"
#include <k4a/k4a.hpp>

namespace holoscan::ops {

/**
 * @brief Captures frames from an Azure Kinect camera.
 */
class AzureKinectCameraOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AzureKinectCameraOp)

  AzureKinectCameraOp() = default;

  void setup(OperatorSpec& spec) override;

  void start() override;
  void stop() override;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:

  gxf_result_t open_device();

  Parameter<std::string> device_serial_{"ANY"};
  Parameter<unsigned int> capture_timeout_ms_{33};
  Parameter<std::shared_ptr<Allocator>> allocator_;

  k4a::device m_handle;  //!< Azure handle going with the camera
  k4a_device_configuration_t m_camera_config{K4A_DEVICE_CONFIG_INIT_DISABLE_ALL};  //!< Structure containing the device information
  k4a::capture m_capture;
  k4a::calibration m_device_calibration;

};

}  // namespace holoscan::ops

#endif  // HOLOSCAN_OPERATORS_AZURE_KINECT_CAMERA
