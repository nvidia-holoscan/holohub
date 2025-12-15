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

#include "vpi_stereo.h"
#include <vpi/Context.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>
#include <sstream>
#include <stdexcept>

#define CHECK_VPI(STMT)                                                               \
  do {                                                                                \
    VPIStatus status = (STMT);                                                        \
    if (status != VPI_SUCCESS) {                                                      \
      char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];                                     \
      vpiGetLastStatusMessage(buffer, sizeof(buffer));                                \
      std::ostringstream ss;                                                          \
      ss << "line " << __LINE__ << " " << vpiStatusGetName(status) << ": " << buffer; \
      throw std::runtime_error(ss.str());                                             \
    }                                                                                 \
  } while (0);

namespace holoscan::ops {

// inputs are expected to be a rectified stereo pair of Tensors that hold rgb888 data in HWC
// layout. input1 should be the left camera view, and input2 the right.
// output is a disparity map Tensor of float data in HWC layout with size width x height x 1
void VPIStereoOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("input1");
  spec.input<holoscan::gxf::Entity>("input2");
  spec.output<holoscan::gxf::Entity>("output");
  spec.param(width_, "width");
  spec.param(height_, "height");
  spec.param(maxDisparity_, "maxDisparity");
  spec.param(downscaleFactor_, "downscaleFactor");
}

// VPI stereo disparity estimator can use this combination of accelerators on most Tegra platforms.
#define OFA_PVA_VIC (VPI_BACKEND_OFA | VPI_BACKEND_PVA | VPI_BACKEND_VIC)

void VPIStereoOp::start() {
  // set the VPI stereo disparity parameters
  CHECK_VPI(vpiInitStereoDisparityEstimatorCreationParams(&createParams_));
  createParams_.maxDisparity = maxDisparity_;
  createParams_.downscaleFactor = downscaleFactor_;
  CHECK_VPI(vpiInitStereoDisparityEstimatorParams(&submitParams_));
  widthDownscaled_ = width_ / downscaleFactor_;
  heightDownscaled_ = height_ / downscaleFactor_;

  // prefer to offload GPU if other accelerators are available. Check the VPI context to discover
  // which backends are available.
  backends_ = OFA_PVA_VIC;
  inFmt_ = VPI_IMAGE_FORMAT_Y8_ER_BL;
  VPIContext vpiCtx;
  CHECK_VPI(vpiContextGetCurrent(&vpiCtx));
  uint64_t vpiCtxFlags;
  CHECK_VPI(vpiContextGetFlags(vpiCtx, &vpiCtxFlags));
  if ((vpiCtxFlags & OFA_PVA_VIC) != OFA_PVA_VIC) {
    printf("Info: OFA|PVA|VIC not available! Falling back to CUDA backend.\n");
    backends_ = VPI_BACKEND_CUDA;
    inFmt_ = VPI_IMAGE_FORMAT_Y8_ER;
  } else {
    // VPI's accelerator-based stereo has a unique flavor of confidence map that performs better
    // than the default. Select it here.
    submitParams_.confidenceType = VPI_STEREO_CONFIDENCE_INFERENCE;
  }
  CHECK_VPI(vpiStreamCreate(0, &stream_));
  CHECK_VPI(vpiCreateStereoDisparityEstimator(
      backends_, width_, height_, inFmt_, &createParams_, &payload_));

  // intermediate images
  CHECK_VPI(vpiImageCreate(
      width_, height_, VPI_IMAGE_FORMAT_RGB8, backends_ | VPI_BACKEND_CUDA, &inLeftRGB_));
  CHECK_VPI(vpiImageCreate(
      width_, height_, VPI_IMAGE_FORMAT_RGB8, backends_ | VPI_BACKEND_CUDA, &inRightRGB_));
  CHECK_VPI(vpiImageCreate(width_, height_, inFmt_, backends_ | VPI_BACKEND_CUDA, &inLeftMono_));
  CHECK_VPI(vpiImageCreate(width_, height_, inFmt_, backends_ | VPI_BACKEND_CUDA, &inRightMono_));
  CHECK_VPI(vpiImageCreate(widthDownscaled_,
                           heightDownscaled_,
                           VPI_IMAGE_FORMAT_U16,
                           backends_ | VPI_BACKEND_CUDA,
                           &outConf16_));
  CHECK_VPI(vpiImageCreate(widthDownscaled_,
                           heightDownscaled_,
                           VPI_IMAGE_FORMAT_S16,
                           backends_ | VPI_BACKEND_CUDA,
                           &outDisp16_));
  CHECK_VPI(vpiImageCreate(widthDownscaled_,
                           heightDownscaled_,
                           VPI_IMAGE_FORMAT_F32,
                           backends_ | VPI_BACKEND_CUDA,
                           &outDisp_));
}

void VPIStereoOp::stop() {
  vpiStreamDestroy(stream_);
  vpiPayloadDestroy(payload_);
  vpiImageDestroy(inLeftRGB_);
  vpiImageDestroy(inRightRGB_);
  vpiImageDestroy(inLeftMono_);
  vpiImageDestroy(inRightMono_);
  vpiImageDestroy(outConf16_);
  vpiImageDestroy(outDisp16_);
  vpiImageDestroy(outDisp_);
}

void VPIStereoOp::compute(InputContext& op_input, OutputContext& op_output,
                          ExecutionContext& context) {
  auto maybe_tensormap1 = op_input.receive<holoscan::TensorMap>("input1");
  const auto tensormap1 = maybe_tensormap1.value();

  auto maybe_tensormap2 = op_input.receive<holoscan::TensorMap>("input2");
  const auto tensormap2 = maybe_tensormap2.value();

  auto tensor1 = tensormap1.begin()->second;
  auto tensor2 = tensormap2.begin()->second;

  // input HWC layout, RGB8
  int orig_height = tensor1->shape()[0];
  int orig_width = tensor1->shape()[1];
  int nChannels = tensor1->shape()[2];

  if ((orig_height != tensor2->shape()[0]) || (orig_width != tensor2->shape()[1]) ||
      (nChannels != tensor2->shape()[2])) {
    throw std::runtime_error("Input tensor shapes do not match");
  }

  if ((tensormap1.size() != 1) || (tensormap2.size() != 1)) {
    throw std::runtime_error("Expecting two single-tensor inputs");
  }

  if (!(nChannels == 3)) {
    throw std::runtime_error("Input tensors expected to have 3 channels");
  }

  // wrap tensor buffers in VPIImages
  VPIImage inLeftRGB, inRightRGB, outDisp;
  VPIImageData data;
  data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
  data.buffer.pitch.format = VPI_IMAGE_FORMAT_RGB8;
  data.buffer.pitch.numPlanes = 1;
  data.buffer.pitch.planes[0].data = tensor1->data();
  data.buffer.pitch.planes[0].pitchBytes = orig_width * 3;
  data.buffer.pitch.planes[0].width = orig_width;
  data.buffer.pitch.planes[0].height = orig_height;
  data.buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_3U8;
  CHECK_VPI(vpiImageCreateWrapper(&data, NULL, VPI_BACKEND_CUDA, &inLeftRGB));
  // right is same size as left, just different pointer
  data.buffer.pitch.planes[0].data = tensor2->data();
  CHECK_VPI(vpiImageCreateWrapper(&data, NULL, VPI_BACKEND_CUDA, &inRightRGB));

  // allocate output buffer for F32 disparity
  auto pointerDisp = std::shared_ptr<void*>(new void*, [](void** pointer) {
    if (pointer != nullptr) {
      if (*pointer != nullptr) {
        cudaFree(*pointer);
      }
      delete pointer;
    }
  });
  cudaMalloc(pointerDisp.get(), orig_width * orig_height * sizeof(float));
  // wrap output for VPI
  data.buffer.pitch.format = VPI_IMAGE_FORMAT_F32;
  data.buffer.pitch.planes[0].data = *pointerDisp.get();
  data.buffer.pitch.planes[0].pitchBytes = orig_width * sizeof(float);
  data.buffer.pitch.planes[0].width = orig_width;
  data.buffer.pitch.planes[0].height = orig_height;
  data.buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_F32;
  CHECK_VPI(vpiImageCreateWrapper(&data, NULL, VPI_BACKEND_CUDA, &outDisp));

  // first, rescale the inputs to the target resolution if necessary
  VPIImage inLeft = inLeftRGB;
  VPIImage inRight = inRightRGB;
  if ((width_ != orig_width) || (height_ != orig_height)) {
    CHECK_VPI(vpiSubmitRescale(stream_,
                               VPI_BACKEND_CUDA,
                               inLeftRGB,
                               inLeftRGB_,
                               VPI_INTERP_CATMULL_ROM,
                               VPI_BORDER_CLAMP,
                               0));
    CHECK_VPI(vpiSubmitRescale(stream_,
                               VPI_BACKEND_CUDA,
                               inRightRGB,
                               inRightRGB_,
                               VPI_INTERP_CATMULL_ROM,
                               VPI_BORDER_CLAMP,
                               0));
    inLeft = inLeftRGB_;
    inRight = inRightRGB_;
  }

  // next, convert from RGB888 to a grayscale format
  /// Note: RGB888 -> Y8 conversion is only supported on CUDA backend
  /// (VIC would be able to convert RGBA8888, but for the purpose of this example we use RGB888)
  CHECK_VPI(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, inLeft, inLeftMono_, NULL));
  CHECK_VPI(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, inRight, inRightMono_, NULL));

  // then, estimate stereo disparity
  CHECK_VPI(vpiSubmitStereoDisparityEstimator(stream_,
                                              backends_,
                                              payload_,
                                              inLeftMono_,
                                              inRightMono_,
                                              outDisp16_,
                                              outConf16_,
                                              &submitParams_));

  // finally, convert stereo output from 16 bit (10.5 fixed point) to float
  VPIConvertImageFormatParams convParams;
  CHECK_VPI(vpiInitConvertImageFormatParams(&convParams));
  /// Note: VPI format conversion with scale is only supported on CUDA backend
  convParams.scale = 1.f / 32.f;
  // rescale output back to the original size if necessary. Using NEAREST since interpolation would
  // introduce false disparity values at discontinuities.
  if ((widthDownscaled_ != orig_width) || (heightDownscaled_ != orig_height)) {
    CHECK_VPI(
        vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, outDisp16_, outDisp_, &convParams));
    CHECK_VPI(vpiSubmitRescale(
        stream_, VPI_BACKEND_CUDA, outDisp_, outDisp, VPI_INTERP_NEAREST, VPI_BORDER_CLAMP, 0));
  } else {
    CHECK_VPI(
        vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, outDisp16_, outDisp, &convParams));
  }
  // VPI algorithms execute asynchronously. Sync the stream to ensure they are complete.
  CHECK_VPI(vpiStreamSync(stream_));

  // clean up VPIImage wrappers
  vpiImageDestroy(inLeftRGB);
  vpiImageDestroy(inRightRGB);
  vpiImageDestroy(outDisp);

  // post FP32 disparity tensor as output message
  nvidia::gxf::Shape shape = nvidia::gxf::Shape{orig_height, orig_width, 1};

  auto out_message = nvidia::gxf::Entity::New(context.context());
  auto gxf_tensor_disparity = out_message.value().add<nvidia::gxf::Tensor>("");

  gxf_tensor_disparity.value()->wrapMemory(shape,
                                           nvidia::gxf::PrimitiveType::kFloat32,
                                           sizeof(float),
                                           nvidia::gxf::ComputeTrivialStrides(shape, sizeof(float)),
                                           nvidia::gxf::MemoryStorageType::kDevice,
                                           *pointerDisp,
                                           [orig_pointer = pointerDisp](void*) mutable {
                                             orig_pointer.reset();  // decrement ref count
                                             return nvidia::gxf::Success;
                                           });

  op_output.emit(out_message.value(), "output");
}

}  // namespace holoscan::ops
