/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "prohawkop.hpp"
#include "opencv2/imgproc.hpp"

namespace holoscan::ops {

void ProhawkOp::setup(OperatorSpec& spec) {
  spec.input<gxf::Entity>("input");
  spec.output<gxf::Entity>("output1");

  printf("Starting Prohawk Restoration...\n");
}

void ProhawkOp::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
  auto value1 = op_input.receive<gxf::Entity>("input").value();

  auto tensor = value1.get<Tensor>();
  auto image_shape = tensor->shape();
  auto rows = image_shape[0];
  auto cols = image_shape[1];
  size_t data_size = tensor->nbytes();
  std::vector<uint8_t> in_data(data_size);
  cudaMemcpy(in_data.data(), tensor->data(), data_size, cudaMemcpyDeviceToHost);

  switch (filter1) {
    case 0:
      p->defRadiusX = 60;
      p->defRadiusY = 60;
      p->defJustify = 0;
      p->defAlignment = -1;
      p->defBlendValue = 128;
      p->defInterpolation = 0;
      p->caBlendValue = 256;
      p->defDivideCircle = 16;
      p->defDivideRadius = 16;
      p->defThreshold = 8;
      p->defAccumulation = 128;
      p->maThreshold = 3072;
      p->maAccumulation = 128;
      p->caRatio = 384;
      p->caBlendValue = 256;
      p->caAccumulation = 128;
      p->AFS_Enabled = true;
      selectedFilter = "AFS Enabled";
      break;

    case 1:
      p->defRadiusX = 60;
      p->defRadiusY = 60;
      p->defJustify = 0;
      p->defAlignment = -1;
      p->defBlendValue = 128;
      p->defInterpolation = 0;
      p->caBlendValue = 256;
      p->defDivideCircle = 16;
      p->defDivideRadius = 16;
      p->defThreshold = 8;
      p->defAccumulation = 128;
      p->maThreshold = 3072;
      p->maAccumulation = 128;
      p->caRatio = 384;
      p->caBlendValue = 256;
      p->caAccumulation = 128;
      p->AFS_Enabled = false;
      selectedFilter = "LowLight";
      break;

    case 2:
      p->defRadiusX = 24;
      p->defRadiusY = 24;
      p->defJustify = 0;
      p->defAlignment = -1;
      p->defBlendValue = 128;
      p->defInterpolation = 0;
      p->caBlendValue = 0;
      p->defDivideCircle = 6;
      p->defDivideRadius = 4;
      p->defThreshold = 16;
      p->defAccumulation = 12;
      p->maThreshold = 2048;
      p->maAccumulation = 128;
      p->caRatio = 400;
      p->caBlendValue = 0;
      p->caAccumulation = 128;
      p->AFS_Enabled = false;
      selectedFilter = "Vascular Detail";
      break;

    case 3:

      p->caAccumulation = 0;
      p->caBlendValue = 256;
      p->caRatio = 512;
      p->defAccumulation = 0;
      p->defAlignment = 100;
      p->defBlendValue = 128;
      p->defDivideCircle = 6;
      p->defDivideRadius = 4;
      p->defInterpolation = 0;
      p->defJustify = 44;
      p->defRadiusY = 161;
      p->defRadiusX = 161;
      p->defThreshold = 8;
      p->maAccumulation = 0;
      p->maThreshold = 29582;
      selectedFilter = "Vaper";
      break;

    default:
      p->defRadiusX = 60;
      p->defRadiusY = 60;
      p->defJustify = 0;
      p->defAlignment = -1;
      p->defBlendValue = 128;
      p->defInterpolation = 0;
      p->caBlendValue = 256;
      p->defDivideCircle = 16;
      p->defDivideRadius = 16;
      p->defThreshold = 8;
      p->defAccumulation = 128;
      p->maThreshold = 3072;
      p->maAccumulation = 128;
      p->caRatio = 384;
      p->caBlendValue = 256;
      p->caAccumulation = 128;
      p->AFS_Enabled = true;
      selectedFilter = "AFS Enabled";

      if (filter1 == 100) {
        selectedFilter = "Restoration Disabled";
      }
  }

  p->width = cols;
  p->height = rows;
  p->srcBits = PTGDE_BITS_8U;
  p->srcColor = PTGDE_COLOR_BGR;
  p->dstBits = p->srcBits;
  p->dstColor = p->srcColor;

  cv::Mat sbsmat;
  cv::Mat rgb_image;
  cv::Mat bgr_image;
  cv::Mat dst;

  cv::Mat input_image(rows, cols, CV_8UC3, in_data.data(), cv::Mat::AUTO_STEP);
  cv::cvtColor(input_image, bgr_image, cv::COLOR_RGB2BGR);
  cv::Mat output_image = cv::Mat(bgr_image.size(), bgr_image.type());

  p->srcBuffer = bgr_image.data;
  p->srcStride = static_cast<int>(bgr_image.step);
  p->dstBuffer = output_image.data;
  p->dstStride = static_cast<int>(output_image.step);

  if (prohawkStartFlag == false) {
    printf("Starting Prohawk Vision Holoscan Operator...\n");
  }
  if (filter1 != 100) de.setFrame(p);
  if (prohawkStartFlag == false) {
    printf("Prohawk Vision Holoscan Operator started.\n");
    prohawkStartFlag = true;
  }

  cv::cvtColor(output_image, rgb_image, cv::COLOR_BGR2RGB);
  if (filter1 != 100) {
    cudaMemcpy(tensor->data(), rgb_image.data, data_size, cudaMemcpyHostToDevice);
  }

  op_output.emit(value1, "output1");
  // op_output.emit(value1, "output2");

  if (filter1 != 100) {
    cv::hconcat(bgr_image, output_image, sbsmat);

  } else {
    cv::hconcat(bgr_image, bgr_image, sbsmat);
  }

  if (sbsview == false) {
    cv::namedWindow("Holoscan-SDK Prohawk Test", cv::WINDOW_GUI_EXPANDED);
    cv::Mat menuimage(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::putText(menuimage,
                "Selected Filter:" + selectedFilter,
                cv::Point(50, 50),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::putText(menuimage,
                "AFS: 0",
                cv::Point(50, 100),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::putText(menuimage,
                "LowLight: 1",
                cv::Point(50, 150),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::putText(menuimage,
                "Vascular Detail: 2",
                cv::Point(50, 200),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::putText(menuimage,
                "Vaper: 3",
                cv::Point(50, 250),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::putText(menuimage,
                "Disable Resoration: d",
                cv::Point(50, 300),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::putText(menuimage,
                "Side-by-Side View: v",
                cv::Point(50, 350),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::putText(menuimage,
                "Display Menu Items: m",
                cv::Point(50, 400),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::putText(menuimage,
                "Quit: q",
                cv::Point(50, 450),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);

    cv::imshow("Holoscan-SDK Prohawk Test", menuimage);
  } else {
    cv::namedWindow("Holoscan-SDK Prohawk Test", cv::WINDOW_GUI_EXPANDED);
    cv::putText(sbsmat,
                "Selected Filter:" + selectedFilter,
                cv::Point(50, 50),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::putText(sbsmat,
                "AFS: 0",
                cv::Point(50, 100),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::putText(sbsmat,
                "LowLight: 1",
                cv::Point(50, 150),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::putText(sbsmat,
                "Vascular Detail: 2",
                cv::Point(50, 200),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::putText(sbsmat,
                "Vaper: 3",
                cv::Point(50, 250),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::putText(sbsmat,
                "Disable Resoration: d",
                cv::Point(50, 300),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::putText(sbsmat,
                "Enable Side-by-Side View: v",
                cv::Point(50, 350),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::putText(sbsmat,
                "Display Menu Items: m",
                cv::Point(50, 400),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::putText(sbsmat,
                "Quit: q",
                cv::Point(50, 450),
                cv::FONT_HERSHEY_DUPLEX,
                1.2,
                CV_RGB(255, 255, 255),
                2);
    cv::imshow("Holoscan-SDK Prohawk Test", sbsmat);
  }
  filter_tmp = cv::waitKey(33);
  if (filter_tmp == 48) {
    printf("AFS\n");
    filter1 = 0;
  }
  if (filter_tmp == 49) {
    printf("Lowlight filter set\n");
    filter1 = 1;
  }
  if (filter_tmp == 50) {
    printf("Vascular filter set \n");
    filter1 = 2;
  }
  if (filter_tmp == 51) {
    printf("Vaper filter set \n");
    filter1 = 3;
  }
  if (filter_tmp == 100) {
    printf("Disabling Restoration\n");
    filter1 = 100;
  }
  if (filter_tmp == 118) {
    printf("Enable Side-by-Side View \n");
    filter1 = 118;
    sbsview = true;
  }
  if (filter_tmp == 109) {
    printf("Disable Side-by-Side View\n");
    filter1 = 109;
    sbsview = false;
  }
  if (filter_tmp == 113) {
    (printf("Quit\n"));
    filter1 = 113;
    exit(0);
  }
}

}  // namespace holoscan::ops
