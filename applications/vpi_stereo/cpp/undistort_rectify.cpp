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

#include "undistort_rectify.h"
#include <math.h>
#include <npp.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "stereo_depth_kernels.h"

namespace holoscan::ops {

// Rectification Maps Data Structure
UndistortRectifyOp::RectificationMap::RectificationMap(float* M, float* d, float* R, float* P,
                                                       int width, int height) {
  setParameters(M, d, R, P, width, height);
}
UndistortRectifyOp::RectificationMap::~RectificationMap() {
  cudaFree(mapx_);
  cudaFree(mapy_);
}
void UndistortRectifyOp::RectificationMap::setParameters(float* M, float* d, float* R, float* P,
                                                         int width, int height) {
  if (mapx_ != NULL) {
    cudaFree(mapx_);
  }
  if (mapy_ != NULL) {
    cudaFree(mapy_);
  }
  cudaMalloc((void**)&mapx_, height * width * sizeof(float));
  cudaMalloc((void**)&mapy_, height * width * sizeof(float));
  float *d_M, *d_d, *d_R, *d_P;
  cudaMalloc((void**)&d_M, 9 * sizeof(float));
  cudaMalloc((void**)&d_d, 5 * sizeof(float));
  cudaMalloc((void**)&d_R, 9 * sizeof(float));
  cudaMalloc((void**)&d_P, 12 * sizeof(float));
  cudaMemcpy(d_M, M, 9 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d, d, 5 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_R, R, 9 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_P, P, 12 * sizeof(float), cudaMemcpyHostToDevice);
  makeRectificationMap(d_M, d_d, d_R, d_P, mapx_, mapy_, width, height, 0);
  cudaFree(d_M);
  cudaFree(d_d);
  cudaFree(d_R);
  cudaFree(d_P);

  width_ = width;
  height_ = height;
}

void UndistortRectifyOp::stereoRectify(float* M1, float* d1, float* M2, float* d2, float* R,
                                       float* t, int width, int height, float* R1, float* R2,
                                       float* P1, float* P2, float* Q, int* roi) {
  Eigen::Matrix3f M1_mat = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(M1);
  Eigen::Matrix3f M2_mat = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(M2);

  Eigen::Matrix3f R_mat = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(R);
  Eigen::Vector3f t_vec = Eigen::Map<Eigen::Vector3f>(t);

  Eigen::AngleAxisf om(R_mat);
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_A(Eigen::AngleAxisf(-0.5f * om.angle(), om.axis()));
  Eigen::Vector3f t_A = R_A * t_vec;
  Eigen::Vector3f uu = Eigen::Vector3f::Zero();
  uu[0] = t_A[0] > 0 ? 1 : -1;
  Eigen::Vector3f ww_axis = (t_A.cross(uu)).normalized();
  float nt = t_A.norm();
  float ww_angle = acos(fabs(t_A[0]) / nt);
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_w(Eigen::AngleAxisf(ww_angle, ww_axis));

  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R1_mat = R_w * (R_A.transpose());
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R2_mat = R_w * R_A;
  std::copy(R1_mat.data(), R1_mat.data() + 9, R1);
  std::copy(R2_mat.data(), R2_mat.data() + 9, R2);

  std::vector<std::pair<float, float>> border_points;
  border_points.push_back(std::make_pair(0.0f, 0.0f));
  border_points.push_back(std::make_pair((float)width, 0.0f));
  border_points.push_back(std::make_pair(0.0f, (float)height));
  border_points.push_back(std::make_pair((float)width, (float)height));

  float f = (M1_mat(1, 1) + M2_mat(1, 1)) / 2;
  std::vector<std::pair<float, float>> border_points_undistorted_1 =
      UndistortRectifyOp::originalToRectified(border_points, M1, d1, R1, f);
  std::vector<std::pair<float, float>> border_points_undistorted_2 =
      UndistortRectifyOp::originalToRectified(border_points, M2, d2, R2, f);
  float c1x = 0;
  float c1y = 0;
  float c2x = 0;
  float c2y = 0;
  for (int i = 0; i < border_points.size(); i++) {
    c1x += border_points_undistorted_1[i].first;
    c1y += border_points_undistorted_1[i].second;
    c2x += border_points_undistorted_2[i].first;
    c2y += border_points_undistorted_2[i].second;
  }

  float cx = static_cast<float>(width - 1) / 2.0f - (c1x + c2x) / (2 * border_points.size());
  float cy = static_cast<float>(height - 1) / 2.0f - (c1y + c2y) / (2 * border_points.size());

  float x11 = std::max(border_points_undistorted_1[0].first, border_points_undistorted_1[2].first);
  float y11 =
      std::max(border_points_undistorted_1[0].second, border_points_undistorted_1[1].second);
  float x12 = std::min(border_points_undistorted_1[1].first, border_points_undistorted_1[3].first);
  float y12 =
      std::min(border_points_undistorted_1[2].second, border_points_undistorted_1[3].second);

  float x21 = std::max(border_points_undistorted_2[0].first, border_points_undistorted_2[2].first);
  float y21 =
      std::max(border_points_undistorted_2[0].second, border_points_undistorted_2[1].second);
  float x22 = std::min(border_points_undistorted_2[1].first, border_points_undistorted_2[3].first);
  float y22 =
      std::min(border_points_undistorted_2[2].second, border_points_undistorted_2[3].second);

  Eigen::Matrix<float, 3, 4, Eigen::RowMajor> P1_mat =
      Eigen::Matrix<float, 3, 4, Eigen::RowMajor>::Zero();
  Eigen::Matrix<float, 3, 4, Eigen::RowMajor> P2_mat =
      Eigen::Matrix<float, 3, 4, Eigen::RowMajor>::Zero();
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Q_mat =
      Eigen::Matrix<float, 4, 4, Eigen::RowMajor>::Zero();

  P1_mat(0, 0) = f;
  P1_mat(0, 2) = cx;
  P1_mat(1, 1) = f;
  P1_mat(1, 2) = cy;
  P1_mat(2, 2) = 1;

  P2_mat(0, 0) = f;
  P2_mat(0, 2) = cx;
  P2_mat(1, 1) = f;
  P2_mat(1, 2) = cy;
  P2_mat(2, 2) = 1;

  Eigen::Vector3f t2 = R2_mat * t_vec;
  Q_mat(0, 0) = 1;
  Q_mat(0, 3) = -cx;
  Q_mat(1, 1) = 1;
  Q_mat(1, 3) = -cy;
  Q_mat(2, 3) = f;
  Q_mat(3, 3) = -1.0f / t2[0];
  std::copy(P1_mat.data(), P1_mat.data() + 12, P1);
  std::copy(P2_mat.data(), P2_mat.data() + 12, P2);
  std::copy(Q_mat.data(), Q_mat.data() + 16, Q);

  roi[0] = (int)std::max(std::max(x11, x21) + cx, 0.0f);
  roi[1] = (int)std::max(std::max(y11, y21) + cy, 0.0f);
  roi[2] = (int)std::min(std::min(x12, x22) - roi[0] + cx, (float)width - 1.0f);
  roi[3] = (int)std::min(std::min(y12, y22) - roi[1] + cy, (float)height - 1.0f);
}
std::vector<std::pair<float, float>> UndistortRectifyOp::originalToRectified(
    std::vector<std::pair<float, float>> pts_in, float* M, float* d, float* R, float f) {
  int N = pts_in.size();
  std::vector<std::pair<float, float>> border_points_undistorted =
      UndistortRectifyOp::undistortPoints(pts_in, M, d);

  std::vector<std::pair<float, float>> pts_out(N);

  for (int n = 0; n < N; n++) {
    float x = (border_points_undistorted[n].first - M[2]) / M[0];
    float y = (border_points_undistorted[n].second - M[5]) / M[4];  // z=1
    float z2 = R[6] * x + R[7] * y + R[8];
    pts_out[n].first = f * (R[0] * x + R[1] * y + R[2]) / z2;
    pts_out[n].second = f * (R[3] * x + R[4] * y + R[5]) / z2;
  }
  return pts_out;
}

std::vector<std::pair<float, float>> UndistortRectifyOp::distortPoints(
    std::vector<std::pair<float, float>> pts_in, float* M, float* d) {
  int N = pts_in.size();
  std::vector<std::pair<float, float>> pts_out(N);
  for (int n = 0; n < N; n++) {
    float x = (pts_in[n].first - M[2]) / M[0];
    float y = (pts_in[n].second - M[5]) / M[4];
    float r2 = x * x + y * y;
    float xy = x * y;
    float rad = 1.0f + d[0] * r2 + d[1] * (r2 * r2) + d[4] * (r2 * r2 * r2);
    pts_out[n].first = (rad * x + 2 * d[2] * xy + d[3] * (r2 + 2 * (x * x))) * M[0] + M[2];
    pts_out[n].second = (rad * y + d[2] * (r2 + 2 * y * y) + 2 * d[3] * xy) * M[4] + M[5];
  }
  return pts_out;
}

std::vector<std::pair<float, float>> UndistortRectifyOp::undistortPoints(
    std::vector<std::pair<float, float>> pts_in, float* M, float* d, float tol, int max_it) {
  int N = pts_in.size();
  std::vector<std::pair<float, float>> pts_out = pts_in;
  for (int i = 0; i < max_it; i++) {
    std::vector<std::pair<float, float>> pts_in_est =
        UndistortRectifyOp::distortPoints(pts_out, M, d);
    float err2 = 0;
    std::vector<float> err_x(N);
    std::vector<float> err_y(N);

    for (int n = 0; n < N; n++) {
      err_x[n] = pts_in[n].first - pts_in_est[n].first;
      err_y[n] = pts_in[n].second - pts_in_est[n].second;
      err2 += err_x[n] * err_x[n] + err_y[n] * err_y[n];
    }
    if (err2 < (tol * tol)) {
      return pts_out;
    } else {
      for (int n = 0; n < N; n++) {
        pts_out[n].first = pts_out[n].first + err_x[n];
        pts_out[n].second = pts_out[n].second + err_y[n];
      }
    }
  }
  std::cout << "Warning: Max iterations reached" << std::endl;
  return pts_out;
}

void UndistortRectifyOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("input");
  spec.output<holoscan::gxf::Entity>("output");
}

void UndistortRectifyOp::compute(InputContext& op_input, OutputContext& op_output,
                                 ExecutionContext& context) {
  auto maybe_tensormap = op_input.receive<holoscan::TensorMap>("input");
  const auto tensormap = maybe_tensormap.value();

  if (tensormap.size() != 1) {
    throw std::runtime_error("Expecting single tensor input");
  }

  auto tensor = tensormap.begin()->second;

  int height = tensor->shape()[0];
  int width = tensor->shape()[1];
  int nChannels = tensor->shape()[2];

  if (rectification_map_->mapx_ == NULL || rectification_map_->mapy_ == NULL) {
    throw std::runtime_error("Rectification maps must be created before dewarping");
  }

  if (width != rectification_map_->width_ || height != rectification_map_->height_) {
    throw std::runtime_error("Dimensions do not match rectification map");
  }
  if (!(nChannels == 1 || nChannels == 3 || nChannels == 4)) {
    throw std::runtime_error("Number of channels in input must be 1, 3 or 4");
  }

  auto pointer = std::shared_ptr<void*>(new void*, [](void** pointer) {
    if (pointer != nullptr) {
      if (*pointer != nullptr) {
        cudaFree(*pointer);
      }
      delete pointer;
    }
  });

  cudaMalloc(pointer.get(), width * height * nChannels * sizeof(uint8_t));
  NppStatus status;
  if (nChannels == 1) {
    nppiRemap_8u_C1R(static_cast<Npp8u*>(tensor->data()),
                     {width, height},
                     width * nChannels * sizeof(Npp8u),
                     {0, 0, width, height},
                     rectification_map_->mapx_,
                     width * sizeof(Npp32f),
                     rectification_map_->mapy_,
                     width * sizeof(Npp32f),
                     static_cast<Npp8u*>(*pointer.get()),
                     width * nChannels * sizeof(Npp8u),
                     {width, height},
                     NPPI_INTER_LINEAR);
  } else if (nChannels == 3) {
    status = nppiRemap_8u_C3R(static_cast<Npp8u*>(tensor->data()),
                              {width, height},
                              width * nChannels * sizeof(Npp8u),
                              {0, 0, width, height},
                              rectification_map_->mapx_,
                              width * sizeof(Npp32f),
                              rectification_map_->mapy_,
                              width * sizeof(Npp32f),
                              static_cast<Npp8u*>(*pointer.get()),
                              width * nChannels * sizeof(Npp8u),
                              {width, height},
                              NPPI_INTER_LINEAR);
  } else if (nChannels == 4) {
    nppiRemap_8u_C4R(static_cast<Npp8u*>(tensor->data()),
                     {width, height},
                     width * nChannels * sizeof(Npp8u),
                     {0, 0, width, height},
                     rectification_map_->mapx_,
                     width * sizeof(Npp32f),
                     rectification_map_->mapy_,
                     width * sizeof(Npp32f),
                     static_cast<Npp8u*>(*pointer.get()),
                     width * nChannels * sizeof(Npp8u),
                     {width, height},
                     NPPI_INTER_LINEAR);
  } else {
    throw std::runtime_error("Number of channels in input must be 1, 3 or 4");
  }
  auto out_message = nvidia::gxf::Entity::New(context.context());
  auto gxf_tensor = out_message.value().add<nvidia::gxf::Tensor>("");
  nvidia::gxf::Shape shape = nvidia::gxf::Shape{height, width, nChannels};
  int element_size = nvidia::gxf::PrimitiveTypeSize(nvidia::gxf::PrimitiveType::kUnsigned8);
  gxf_tensor.value()->wrapMemory(shape,
                                 nvidia::gxf::PrimitiveType::kUnsigned8,
                                 element_size,
                                 nvidia::gxf::ComputeTrivialStrides(shape, element_size),
                                 nvidia::gxf::MemoryStorageType::kDevice,
                                 *pointer,
                                 [orig_pointer = pointer](void*) mutable {
                                   orig_pointer.reset();  // decrement ref count
                                   return nvidia::gxf::Success;
                                 });

  op_output.emit(out_message.value(), "output");
}

}  // namespace holoscan::ops
