/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include "vis_orsi.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <npp.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"

#include "opengl_utils.hpp"

#define CUDA_TRY(stmt)                                                                          \
  ({                                                                                            \
    cudaError_t _holoscan_cuda_err = stmt;                                                      \
    if (cudaSuccess != _holoscan_cuda_err) {                                                    \
      HOLOSCAN_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", \
                         #stmt,                                                                 \
                         __LINE__,                                                              \
                         __FILE__,                                                              \
                         cudaGetErrorString(_holoscan_cuda_err),                                \
                         static_cast<int>(_holoscan_cuda_err));                                 \
    }                                                                                           \
    _holoscan_cuda_err;                                                                         \
  })

namespace holoscan::orsi {

// --------------------------------------------------------------------------------------------
//
// event handlers
//

void OrsiVis::onFramebufferSizeCallback(GLFWwindow* wnd, int width, int height) {
  vtk_view_.onSize(wnd, width, height);
}

void OrsiVis::onChar(GLFWwindow* wnd, unsigned int codepoint) {
  vtk_view_.onChar(wnd, codepoint);
}

void OrsiVis::onEnter(GLFWwindow* wnd, int entered) {
  vtk_view_.onEnter(wnd, entered);
}

void OrsiVis::onMouseMove(GLFWwindow* wnd, double x, double y) {
  if (!enable_model_manip_) return;
  vtk_view_.onMouseMove(wnd, x, y);
}

void OrsiVis::onMouseButtonCallback(GLFWwindow* wnd, int button, int action, int mods) {
  if (!enable_model_manip_) return;
  vtk_view_.onMouseButtonCallback(wnd, button, action, mods);
}

void OrsiVis::onScrollCallback(GLFWwindow* wnd, double xoffset, double yoffset) {
  if (!enable_model_manip_) return;
  vtk_view_.onScrollCallback(wnd, xoffset, yoffset);
}

void OrsiVis::onKeyCallback(GLFWwindow* wnd, int key, int scancode, int action, int mods) {
  if (action == GLFW_PRESS) {
    if (key == GLFW_KEY_O) {
      apply_tool_overlay_effect_ = !apply_tool_overlay_effect_;
      if (apply_tool_overlay_effect_) HOLOSCAN_LOG_INFO("Surgical Tool Overlay enabled");
      else
        HOLOSCAN_LOG_INFO("Surgical Tool Overlay disabled");
      return;
    }
    if (key == GLFW_KEY_T) {
      enable_model_manip_ = !enable_model_manip_;
      if (enable_model_manip_) {
        HOLOSCAN_LOG_INFO("3D Model transform enabled!");
      } else {
        HOLOSCAN_LOG_INFO("3D Model transform locked!");
      }

      return;
    }
    if (key == GLFW_KEY_B) {
      toggle_anonymization_ = !toggle_anonymization_;
      if (toggle_anonymization_) HOLOSCAN_LOG_INFO("Anonymization enabled!");
      else
        HOLOSCAN_LOG_INFO("Anonymization disabled!");

      return;
    }
  }

  vtk_view_.onKey(wnd, key, scancode, action, mods);
}

// --------------------------------------------------------------------------------------------
//
// Main GXF Codelet interface implementation
//
void OrsiVis::setup(OperatorSpec& spec) {
  spec.param(swizzle_video_,
             "swizzle_video",
             "Swizzle Video channels",
             "Set to true to swizzle input video from RGB to BGR",
             false);
  spec.param(stl_file_path_,
             "stl_file_path",
             "STL File Path",
             "Path to STL files used to construct the 3D model");
  spec.param(stl_names_,
             "stl_names",
             "STL Filenames",
             "Names of STL files used to construct the 3D model");
  spec.param(stl_colors_,
             "stl_colors",
             "STL Colors",
             "Colors of 3D structures corresponding to STL files");
  spec.param(stl_keys_,
             "stl_keys",
             "STL Keybindings",
             "Keybindings used to toggle on/off the corresponding STL file 3D structure");
  spec.param(registration_params_path_,
             "registration_params_path",
             "TF params",
             "VtkProp3DTransformParams for 3D structure",
             {});
}

void OrsiVis::initialize() {}

void OrsiVis::resizeVideoBufferResources(int width, int height, int channels) {
  // no need to re-allocate if nothing changed
  if (video_frame_width_ == width && video_frame_height_ == height &&
      video_frame_channels_ == channels) {
    return;
  }

  video_frame_width_ = width;
  video_frame_height_ = height;
  video_frame_channels_ = channels;

  // CUDA GL Interop not support with 3 channel formats, e.g. GL_RGB
  use_cuda_opengl_interop_ = video_frame_channels_ != 3;

  // Allocate host memory and OpenGL buffers, textures for video frame and inference results
  // ----------------------------------------------------------------------------------

  const size_t buffer_size = video_frame_width_ * video_frame_height_ * video_frame_channels_;
  video_frame_buffer_host_.resize(buffer_size, 0);

  glActiveTexture(GL_TEXTURE0);
  glGenTextures(1, &video_frame_tex_);
  glBindTexture(GL_TEXTURE_2D, video_frame_tex_);
  // allocate immutable texture storage ( if resize need to re-create texture object)
  GLenum format = (video_frame_channels_ == 4) ? GL_RGBA8 : GL_RGB8;
  glTexStorage2D(GL_TEXTURE_2D, 1, format, video_frame_width_, video_frame_height_);
  // set the texture wrapping parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);

  // register this texture with CUDA
  if (use_cuda_opengl_interop_) {
    cudaError_t cuda_status =
        CUDA_TRY(cudaGraphicsGLRegisterImage(&cuda_video_frame_tex_resource_,
                                             video_frame_tex_,
                                             GL_TEXTURE_2D,
                                             cudaGraphicsMapFlagsWriteDiscard));
    if (cuda_status) {
      HOLOSCAN_LOG_ERROR("Failed to register video frame texture for CUDA / OpenGL Interop");
      throw std::runtime_error("Failed to register video frame texture for CUDA / OpenGL Interop");
    }
  }
}

void OrsiVis::resizeSegmentationMaskResources(int width, int height) {
  // no need to re-allocate if nothing changed
  if (seg_mask_width_ == width && seg_mask_height_ == height) {
    return;
  }

  seg_mask_width_ = width;
  seg_mask_height_ = height;

  glGenTextures(1, &seg_mask_tex_);
  glBindTexture(GL_TEXTURE_2D, seg_mask_tex_);
  // allocate immutable texture storage ( if resize need to re-create texture object)
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8, seg_mask_width_, seg_mask_height_);
  // set the texture wrapping parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glBindTexture(GL_TEXTURE_2D, 0);

  // CUDA / GL interop for segmentation mask tex
  {
    cudaError_t cuda_status =
        CUDA_TRY(cudaGraphicsGLRegisterImage(&cuda_seg_mask_tex_resource_,
                                             seg_mask_tex_,
                                             GL_TEXTURE_2D,
                                             cudaGraphicsMapFlagsWriteDiscard));
    if (cuda_status) {
      HOLOSCAN_LOG_ERROR("Failed to register segmentation mask texture for CUDA / OpenGL Interop");
      throw std::runtime_error("Failed to register video frame texture for CUDA / OpenGL Interop");
    }
  }
}

void OrsiVis::start() {
  // Initialize helper class instancces
  // ----------------------------------------------------------------------------------
  if (swizzle_video_.get()) {
    HOLOSCAN_LOG_INFO("Surgical Video format will be change from RGB to BGR during vis");
  }

  video_frame_vis_.start(swizzle_video_.get());

  vtk_view_.setStlFilePath(stl_file_path_);
  vtk_view_.setTfParams(registration_params_path_);
  vtk_view_.setStlNames(stl_names_);
  vtk_view_.setStlColors(stl_colors_);
  vtk_view_.setStlKeys(stl_keys_);

  vtk_view_.start();
}

void OrsiVis::stop() {
  // Free mem allocated in utility classes.
  // ----------------------------------------------------------------------------------
  video_frame_vis_.stop();
  vtk_view_.stop();
}

void OrsiVis::compute(
    const std::unordered_map<std::string, holoscan::orsi::vis::BufferInfo>& input_buffers) {
  using holoscan::orsi::vis::BufferInfo;

  static const std::string surgical_video_buffer_key = "";
  static const std::string segmentation_tensor_key = "segmentation_postprocessed";
  static const std::string anon_key = "anonymization_infer";

  cudaError_t cuda_status = {};

#if 1
  // Update Surgical Video
  if (input_buffers.count(surgical_video_buffer_key)) {
    auto ibuffer = input_buffers.at(surgical_video_buffer_key);

    const nvidia::gxf::Shape& shape = ibuffer.shape;
    const int32_t columns = shape.dimension(1);
    const int32_t rows = shape.dimension(0);
    const int32_t channels = shape.dimension(2);
    uint8_t* in_tensor_ptr = nullptr;

    if (ibuffer.type == BufferInfo::VIDEO) {
      // NOTE: VideoBuffer::moveToTensor() converts VideoBuffer instance to the Tensor instance
      // with an unexpected shape:  [width, height] or [width, height, num_planes].
      // And, if we use moveToTensor() to convert VideoBuffer to Tensor, we may lose the the
      // original video buffer when the VideoBuffer instance is used in other places. For that
      // reason, we directly access internal data of VideoBuffer instance to access Tensor data.
      const auto& buffer_info = ibuffer.video_buffer->video_frame_info();
      switch (buffer_info.color_format) {
        case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA:
          break;
        default:
          HOLOSCAN_LOG_ERROR("Unsupported input format");
          throw std::runtime_error("Unsupported input format");
      }

      in_tensor_ptr = ibuffer.video_buffer->pointer();
    } else {
      // Get tensor attached to the message
      auto in_tensor = ibuffer.tensor;
      if (ibuffer.storage_type != nvidia::gxf::MemoryStorageType::kDevice) {
        HOLOSCAN_LOG_ERROR("Invalid tensor memory storage mode");
        throw std::runtime_error("Invalid tensor memory storage mode");
      }
      auto maybe_in_tensor_ptr = in_tensor->data<uint8_t>();
      if (!maybe_in_tensor_ptr) {
        HOLOSCAN_LOG_ERROR("Invalid tensor memory format");
        throw std::runtime_error("Invalid tensor memory format");
      }
      in_tensor_ptr = maybe_in_tensor_ptr.value();
    }

    resizeVideoBufferResources(columns, rows, channels);
    const size_t buffer_size = video_frame_width_ * video_frame_height_ * video_frame_channels_;

    if (in_tensor_ptr && buffer_size > 0) {
      //  Video Frame
      // --------------------------------------------------------------------------------------------
      if (use_cuda_opengl_interop_) {
        cuda_status = CUDA_TRY(cudaGraphicsMapResources(1, &cuda_video_frame_tex_resource_, 0));
        if (cuda_status) {
          HOLOSCAN_LOG_ERROR("Failed to map video frame texture via CUDA / OpenGL interop");
          throw std::runtime_error("Failed to map video frame texture via CUDA / OpenGL interop");
        }
        cudaArray* texture_ptr = nullptr;
        cuda_status = CUDA_TRY(cudaGraphicsSubResourceGetMappedArray(
            &texture_ptr, cuda_video_frame_tex_resource_, 0, 0));
        if (cuda_status) {
          HOLOSCAN_LOG_ERROR("Failed to get mapped array for video frame texture");
          throw std::runtime_error("Failed to get mapped array for video frame texture");
        }
        size_t spitch = video_frame_channels_ * video_frame_width_ * sizeof(GLubyte);
        cuda_status = CUDA_TRY(cudaMemcpy2DToArray(texture_ptr,
                                                   0,
                                                   0,
                                                   in_tensor_ptr,
                                                   spitch,
                                                   spitch,
                                                   video_frame_height_,
                                                   cudaMemcpyDeviceToDevice));
        if (cuda_status) {
          HOLOSCAN_LOG_ERROR(
              "Failed to copy video frame to OpenGL texture "
              "via CUDA / OpenGL interop");
          throw std::runtime_error(
              "Failed to copy video frame to OpenGL texture "
              "via CUDA / OpenGL interop");
        }
        cuda_status = CUDA_TRY(cudaGraphicsUnmapResources(1, &cuda_video_frame_tex_resource_, 0));
        if (cuda_status) {
          HOLOSCAN_LOG_ERROR("Failed to unmap video frame texture via CUDA / OpenGL interop");
          throw std::runtime_error("Failed to unmap video frame texture via CUDA / OpenGL interop");
        }
      } else {
        cuda_status = CUDA_TRY(cudaMemcpy(
            video_frame_buffer_host_.data(), in_tensor_ptr, buffer_size, cudaMemcpyDeviceToHost));
        if (cuda_status) {
          HOLOSCAN_LOG_ERROR("Failed to copy video frame texture from Device to Host");
          throw std::runtime_error("Failed to copy video frame texture from Device to Host");
        }
        // update data
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, video_frame_tex_);
        GLenum format = (video_frame_channels_ == 4) ? GL_RGBA : GL_RGB;
        glTexSubImage2D(GL_TEXTURE_2D,
                        0,
                        0,
                        0,
                        video_frame_width_,
                        video_frame_height_,
                        format,
                        GL_UNSIGNED_BYTE,
                        video_frame_buffer_host_.data());
      }
    }  // if (in_tensor_ptr && buffer_size > 0)
  }

#endif

  // ----------------------------------------------------------------------------------------------
  //
  // Update Non-Organic Structure segmentation
  //

#if 1
  if (input_buffers.count(segmentation_tensor_key)) {
    auto ibuffer = input_buffers.at(segmentation_tensor_key);
    auto seg_mask_tensor = ibuffer.tensor;

    const nvidia::gxf::Shape& shape = ibuffer.shape;
    const int32_t columns = shape.dimension(1);
    const int32_t rows = shape.dimension(0);
    const int32_t channels = shape.dimension(2);

    if (channels != 1) {
      HOLOSCAN_LOG_ERROR("Segmentation mask is not a single channel tensor. #%d channels!",
                         channels);
      throw std::runtime_error("Segmentation mask is not a single channel tensor");
    }

    resizeSegmentationMaskResources(columns, rows);

    auto ptr = seg_mask_tensor->data<uint8_t>().value();

    {
      cuda_status = CUDA_TRY(cudaGraphicsMapResources(1, &cuda_seg_mask_tex_resource_, 0));
      if (cuda_status) {
        HOLOSCAN_LOG_ERROR("Failed to map segmentation mask texture via CUDA / OpenGL interop");
        throw std::runtime_error(
            "Failed to map segmentation mask texture "
            "via CUDA / OpenGL interop");
      }

      cudaArray* texture_ptr = nullptr;
      cuda_status = CUDA_TRY(
          cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_seg_mask_tex_resource_, 0, 0));
      if (cuda_status) {
        HOLOSCAN_LOG_ERROR("Failed to get mapped array for segmentation mask texture");
        throw std::runtime_error("Failed to get mapped array for segmentation mask texture");
      }
      size_t spitch = seg_mask_width_ * sizeof(GLubyte);
      cuda_status = CUDA_TRY(cudaMemcpy2DToArray(
          texture_ptr, 0, 0, ptr, spitch, spitch, seg_mask_height_, cudaMemcpyDeviceToDevice));
      if (cuda_status) {
        HOLOSCAN_LOG_ERROR(
            "Failed to copy video frame to OpenGL texture "
            "via CUDA / OpenGL interop");
        throw std::runtime_error(
            "Failed to copy video frame to OpenGL texture "
            "via CUDA / OpenGL interop");
      }

      // unmap the graphics resource again
      cuda_status = CUDA_TRY(cudaGraphicsUnmapResources(1, &cuda_seg_mask_tex_resource_, 0));
      if (cuda_status) {
        HOLOSCAN_LOG_ERROR(
            "Failed to unmap segmentation mask texture "
            "via CUDA / OpenGL interop");
        throw std::runtime_error(
            "Failed to unmap segmentation mask texture "
            "via CUDA / OpenGL interop");
      }
    }
  } else {
    apply_tool_overlay_effect_ = false;
  }
#endif

  // ----------------------------------------------------------------------------------------------
  //
  // Update Anonymization
  //
  if (input_buffers.count(anon_key)) {
    auto ibuffer = input_buffers.at(anon_key);
    auto anonymization_mask_tensor = ibuffer.tensor;

    const nvidia::gxf::Shape& shape = ibuffer.shape;
    const int32_t columns = shape.dimension(1);
    const int32_t rows = shape.dimension(0);
    const int32_t channels = shape.dimension(2);

    if (channels != 1) {
      HOLOSCAN_LOG_ERROR("Anonymization tensor is not a single channel tensor. #%d channels!",
                         channels);
      throw std::runtime_error("Anonymization tensor is not a single channel tensor");
    }

    auto anonymization_ptr = anonymization_mask_tensor->data<float>().value();

    float anonymization_infer_value = 0;
    cuda_status = CUDA_TRY(cudaMemcpy(
        &anonymization_infer_value, anonymization_ptr, sizeof(float), cudaMemcpyDeviceToHost));
    // read anonymization value to host and pass as GLSL uniform

    const double sigmoid_value = 1.0 / (1.0 + exp(-anonymization_infer_value));
    const uint8_t sigmoid_result = sigmoid_value > 0.5 ? 1 : 0;

    apply_anonymization_effect_ = sigmoid_result;

    if (!toggle_anonymization_) {
      apply_anonymization_effect_ = false;
    }
  } else {
    apply_anonymization_effect_ = false;
  }

  // Draw VTK scene
  // ----------------------------------------------------------------------------------
#if 1
  GLuint preop_mesh_tex = vtk_view_.getTexture();
  // only re-render if manipulator is unlocked / active
  vtk_view_.render();
#endif

  // Draw Frame
  // TODO: Update with new shader taking VTK offscreen rendering result as argument
  // ----------------------------------------------------------------------------------
#if 1
  video_frame_vis_.tick(video_frame_tex_,
                        GL_LINEAR,
                        preop_mesh_tex,
                        GL_LINEAR,
                        seg_mask_tex_,
                        GL_NEAREST,
                        apply_tool_overlay_effect_,
                        apply_anonymization_effect_);

#endif
}

}  // namespace holoscan::orsi
