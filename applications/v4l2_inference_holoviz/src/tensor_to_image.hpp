// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "image_conversion.hpp"

namespace camera_inference {

// One process runs one graph. Main observes image emissions without retaining tensors.
inline std::atomic<int> processed_frames{0};

class TensorToImageOp final : public holoscan::Operator<> {
 public:
  TensorToImageOp(int width, int height)
      : width_(width), height_(height), pixels_(static_cast<std::size_t>(width) * height * 3) {}

  void setup(holoscan::OperatorSpec& spec) override {
    // HoloInfer declares only memory placement before loading the model. Validate
    // the actual tensor's float32 type and exact NCHW shape in compute().
    spec.input(input, "prediction")
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice}});
    spec.output(frame, "frame")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 3U},
            .bounds = holoscan::tensor_bounds(static_cast<std::size_t>(width_) * height_ * 3),
            .storage = holoscan::TensorOutputStorage::kRuntimePool})
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>()});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto tensor = input.receive_data();
    if (!tensor) {
      return holoscan::make_unexpected(std::move(tensor).error());
    }
    try {
      const auto size = validate_rgb_layout(tensor->shape(), tensor->strides(), width_, height_);
      if (tensor->data_as<float>() == nullptr) {
        throw std::invalid_argument("Model output must contain float32 RGB pixels");
      }
      if (auto ready = tensor->copy_to_host(
              pixels_.data(), size * sizeof(float), context.cuda_stream());
          !ready) {
        return holoscan::make_unexpected(std::move(ready).error());
      }
      const std::array<std::int64_t, 3> shape{height_, width_, 3};
      auto loan = frame.allocate_tensor(
          holoscan::TensorLoanRequest{.shape = shape, .dtype = DLDataType{kDLUInt, 8U, 1U}});
      if (!loan) {
        return holoscan::make_unexpected(std::move(loan).error());
      }
      auto writer = loan->write_host();
      if (!writer) {
        return holoscan::make_unexpected(std::move(writer).error());
      }
      planar_rgb_float_to_u8(pixels_, {static_cast<std::uint8_t*>(writer->data()), size});
      if (auto committed = std::move(*writer).commit(); !committed) {
        return holoscan::make_unexpected(std::move(committed).error());
      }
      holoscan::schema::ImageT descriptor{};
      descriptor.width = width_;
      descriptor.height = height_;
      descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
      descriptor.color_space = holoscan::schema::ColorSpace_UNSPECIFIED;
      auto emitted = frame.emit_tensor(std::move(*loan), descriptor);
      if (emitted) {
        processed_frames.fetch_add(1);
      }
      return emitted;
    } catch (const std::exception& error) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInvalidArgument, error.what()});
    }
  }

  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::schema::ImageT> frame;

 private:
  int width_;
  int height_;
  std::vector<float> pixels_;
};

}  // namespace camera_inference
