/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "depth_to_point_cloud.hpp"

#include <limits>
#include <memory>
#include <string>

#include <cuda_runtime.h>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>

#include <gxf/std/tensor.hpp>

#include "deproject.hpp"

#define CUDA_TRY(stmt)                                                                        \
  {                                                                                           \
    cudaError_t cuda_status = stmt;                                                           \
    if (cudaSuccess != cuda_status) {                                                         \
      HOLOSCAN_LOG_ERROR("CUDA runtime call {} in line {} of file {} failed with '{}' ({}).", \
                         #stmt,                                                               \
                         __LINE__,                                                            \
                         __FILE__,                                                            \
                         cudaGetErrorString(cuda_status),                                     \
                         static_cast<int>(cuda_status));                                      \
      throw std::runtime_error("CUDA runtime call failed");                                   \
    }                                                                                         \
  }

namespace holoscan::ops {

namespace {

// Map a DLPack dtype to the kernel's depth element type.
DepthDType to_depth_dtype(const DLDataType& dtype) {
  if (dtype.code == kDLFloat && dtype.bits == 32) { return DepthDType::kFloat32; }
  if (dtype.code == kDLUInt && dtype.bits == 16) { return DepthDType::kUint16; }
  throw std::runtime_error(
      "DepthToPointCloudOp: unsupported depth dtype (expected float32 or uint16)");
}

// Fetch a tensor from an entity, by name if given, otherwise the first tensor.
std::shared_ptr<Tensor> get_tensor(const holoscan::gxf::Entity& message, const std::string& name) {
  auto maybe = name.empty() ? message.get<Tensor>() : message.get<Tensor>(name.c_str());
  if (!maybe) {
    throw std::runtime_error("DepthToPointCloudOp: input tensor '" + name + "' not found");
  }
  return maybe;
}

}  // namespace

void DepthToPointCloudOp::setup(OperatorSpec& spec) {
  auto& depth_in = spec.input<gxf::Entity>("depth");
  // Optional inputs must not block execution when unconnected.
  auto& intrinsics_in = spec.input<gxf::Entity>("intrinsics").condition(ConditionType::kNone);
  auto& color_in = spec.input<gxf::Entity>("color").condition(ConditionType::kNone);
  auto& cloud_out = spec.output<gxf::Entity>("point_cloud");
  (void)depth_in;
  (void)intrinsics_in;
  (void)color_in;
  (void)cloud_out;

  spec.param(fx_, "fx", "Focal length x", "Focal length in pixels (x).", 0.0f);
  spec.param(fy_, "fy", "Focal length y", "Focal length in pixels (y).", 0.0f);
  spec.param(cx_, "cx", "Principal point x", "Principal point in pixels (x).", 0.0f);
  spec.param(cy_, "cy", "Principal point y", "Principal point in pixels (y).", 0.0f);
  spec.param(depth_scale_, "depth_scale", "Depth scale",
             "Multiplier converting raw depth to meters (e.g. 0.001 for uint16 mm).", 0.001f);
  spec.param(depth_min_, "depth_min", "Min depth", "Minimum valid depth in meters.", 0.0f);
  spec.param(depth_max_, "depth_max", "Max depth", "Maximum valid depth in meters.", 100.0f);
  spec.param(invalid_value_, "invalid_value", "Invalid value",
             "Value written to X/Y/Z for invalid pixels.",
             std::numeric_limits<float>::quiet_NaN());
  spec.param(depth_tensor_name_, "depth_tensor_name", "Depth tensor name",
             "Name of the depth tensor in the input message (empty = first tensor).",
             std::string(""));
  spec.param(color_tensor_name_, "color_tensor_name", "Color tensor name",
             "Name of the color tensor in the color message (empty = first tensor).",
             std::string(""));
  spec.param(output_tensor_name_, "output_tensor_name", "Output tensor name",
             "Name of the emitted point-cloud tensor.", std::string("point_cloud"));
  spec.param(output_color_tensor_name_, "output_color_tensor_name", "Output color tensor name",
             "Name of the emitted colors tensor.", std::string("colors"));
  spec.param(allocator_, "allocator", "Allocator", "Device allocator for output tensors.");

  cuda_stream_handler_.define_params(spec);
}

void DepthToPointCloudOp::compute(InputContext& op_input, OutputContext& op_output,
                                    ExecutionContext& context) {
  auto depth_message = op_input.receive<gxf::Entity>("depth").value();

  if (cuda_stream_handler_.from_message(context.context(), depth_message) != GXF_SUCCESS) {
    throw std::runtime_error("DepthToPointCloudOp: failed to get CUDA stream from input");
  }
  const cudaStream_t stream = cuda_stream_handler_.get_cuda_stream(context.context());

  // --- Depth tensor: dtype, dimensions, device pointer ---
  auto depth_tensor = get_tensor(depth_message, depth_tensor_name_.get());
  const DepthDType depth_dtype = to_depth_dtype(depth_tensor->dtype());
  const auto& shape = depth_tensor->shape();
  // Accept only [H, W] or [H, W, 1]; a higher-rank tensor (e.g. [H, W, 3]) would otherwise be
  // silently reinterpreted as a single-channel depth buffer.
  if (shape.size() < 2 || shape.size() > 3 ||
      (shape.size() == 3 && static_cast<int>(shape[2]) != 1)) {
    throw std::runtime_error("DepthToPointCloudOp: depth tensor must be [H, W] or [H, W, 1]");
  }
  const int height = static_cast<int>(shape[0]);
  const int width = static_cast<int>(shape[1]);

  // --- Optional per-frame intrinsics override ---
  CameraIntrinsics intr{fx_.get(), fy_.get(), cx_.get(), cy_.get()};
  if (auto maybe_intr = op_input.receive<gxf::Entity>("intrinsics")) {
    auto intr_tensor = get_tensor(maybe_intr.value(), std::string(""));
    const DLDataType idt = intr_tensor->dtype();
    if (intr_tensor->size() < 4 || idt.code != kDLFloat || idt.bits != 32) {
      throw std::runtime_error(
          "DepthToPointCloudOp: intrinsics tensor must be float32 [fx, fy, cx, cy]");
    }
    float host[4];
    // Tiny (16 B) config read; pixel data stays GPU-resident. The intrinsics tensor is
    // produced by an upstream operator on `stream`, so the copy must be ordered on that
    // same stream (a plain cudaMemcpy on the default stream would not wait for it).
    // Known cost: this per-frame stream sync runs ONLY when the optional `intrinsics`
    // port is connected; pipelines that pass static fx/fy/cx/cy via parameters (the common
    // case) skip this branch entirely and incur no sync.
    CUDA_TRY(cudaMemcpyAsync(host, intr_tensor->data(), sizeof(host), cudaMemcpyDefault, stream));
    CUDA_TRY(cudaStreamSynchronize(stream));
    intr = CameraIntrinsics{host[0], host[1], host[2], host[3]};
  }

  // Focal lengths must be non-zero (they divide the deprojection); the defaults are 0.0, so a
  // caller that neither sets the fx/fy params nor connects the intrinsics input is rejected here
  // rather than dividing by zero in the kernel.
  if (intr.fx <= 0.0f || intr.fy <= 0.0f) {
    throw std::runtime_error(
        "DepthToPointCloudOp: fx and fy must be positive (set the fx/fy parameters or connect the "
        "intrinsics input)");
  }

  // --- Optional color input ---
  const void* color_ptr = nullptr;
  int color_channels = 0;
  if (auto maybe_color = op_input.receive<gxf::Entity>("color")) {
    auto color_tensor = get_tensor(maybe_color.value(), color_tensor_name_.get());
    // The kernel reinterprets the color buffer as uchar3/uchar4, so the element type must be
    // uint8 (a float or other-width tensor would be misread byte-for-byte).
    const DLDataType cdt = color_tensor->dtype();
    if (cdt.code != kDLUInt || cdt.bits != 8) {
      throw std::runtime_error("DepthToPointCloudOp: color tensor must be uint8");
    }
    const auto& cshape = color_tensor->shape();
    // Require an explicit channel dimension; a 2D [H, W] tensor is rejected rather than
    // silently assumed to be 3-channel (which would read past the buffer in the kernel).
    if (cshape.size() < 3) {
      throw std::runtime_error(
          "DepthToPointCloudOp: color tensor must be H x W x 3 (uchar3) or H x W x 4 (uchar4)");
    }
    color_channels = static_cast<int>(cshape[2]);
    if (color_channels != 3 && color_channels != 4) {
      throw std::runtime_error(
          "DepthToPointCloudOp: color tensor must be H x W x 3 (uchar3) or H x W x 4 (uchar4)");
    }
    if (static_cast<int>(cshape[0]) != height || static_cast<int>(cshape[1]) != width) {
      throw std::runtime_error(
          "DepthToPointCloudOp: color image dimensions must match the depth image");
    }
    color_ptr = color_tensor->data();
  }

  // --- Allocate outputs ---
  auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      context.context(), allocator_.get()->gxf_cid());
  if (!allocator) {
    throw std::runtime_error("DepthToPointCloudOp: failed to create allocator handle");
  }
  auto out_message = nvidia::gxf::Entity::New(context.context());
  if (!out_message) {
    throw std::runtime_error("DepthToPointCloudOp: failed to create output entity");
  }

  auto xyz_tensor = out_message.value().add<nvidia::gxf::Tensor>(output_tensor_name_.get().c_str());
  if (!xyz_tensor) {
    throw std::runtime_error("DepthToPointCloudOp: failed to add point_cloud tensor to message");
  }
  xyz_tensor.value()->reshape<float>(nvidia::gxf::Shape{height, width, 3},
                                     nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
  if (!xyz_tensor.value()->pointer()) {
    throw std::runtime_error("DepthToPointCloudOp: failed to allocate point_cloud tensor");
  }

  uchar3* out_color = nullptr;
  if (color_ptr != nullptr) {
    auto color_out =
        out_message.value().add<nvidia::gxf::Tensor>(output_color_tensor_name_.get().c_str());
    if (!color_out) {
      throw std::runtime_error("DepthToPointCloudOp: failed to add colors tensor to message");
    }
    color_out.value()->reshape<uint8_t>(nvidia::gxf::Shape{height, width, 3},
                                        nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
    if (!color_out.value()->pointer()) {
      throw std::runtime_error("DepthToPointCloudOp: failed to allocate colors tensor");
    }
    out_color = reinterpret_cast<uchar3*>(color_out.value()->pointer());
  }

  CUDA_TRY(launch_deproject(depth_tensor->data(),
                            depth_dtype,
                            depth_scale_.get(),
                            intr,
                            depth_min_.get(),
                            depth_max_.get(),
                            invalid_value_.get(),
                            color_ptr,
                            color_channels,
                            reinterpret_cast<float3*>(xyz_tensor.value()->pointer()),
                            out_color,
                            width,
                            height,
                            stream));

  if (cuda_stream_handler_.to_message(out_message) != GXF_SUCCESS) {
    throw std::runtime_error("DepthToPointCloudOp: failed to add CUDA stream to output");
  }

  auto result = gxf::Entity(std::move(out_message.value()));
  op_output.emit(result, "point_cloud");
}

}  // namespace holoscan::ops
