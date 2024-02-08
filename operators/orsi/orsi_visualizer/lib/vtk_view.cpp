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

#include "vtk_view.hpp"
#include <GL/glew.h>
#include <holoscan/logger/logger.hpp>

// VTK includes

#include <vtkAssembly.h>
#include <vtkNamedColors.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkSTLReader.h>

#include <GLFW/glfw3.h>  // NOLINT(build/include_order)

#include <iostream>

namespace holoscan::orsi {  // namespace visualizer_orsi

void convert(const double* src, std::vector<double>& dst) {
  for (int i = 0; i != dst.size(); ++i) { dst[i] = src[i]; }
}

struct VtkProp3DTransformParams {
  std::vector<double> origin_{0.0, 0.0, 0.0};       // Get / SetOrigin
  std::vector<double> position_{0.0, 0.0, 0.0};     // Get/SetPosition
  std::vector<double> orientation_{0.0, 0.0, 0.0};  // Get/SetOrientation
  std::vector<double> scale_{0.0, 0.0, 0.0};        // Get / SetScale
};

VtkProp3DTransformParams GetObjectTransform(vtkProp3D* p) {
  VtkProp3DTransformParams params;
  convert(p->GetOrigin(), params.origin_);
  convert(p->GetPosition(), params.position_);
  convert(p->GetOrientation(), params.orientation_);
  convert(p->GetScale(), params.scale_);
  return params;
}

void SetObjectTransform(vtkProp3D* p, VtkProp3DTransformParams const& params) {
  p->SetOrigin(params.origin_[0], params.origin_[1], params.origin_[2]);
  p->SetPosition(params.position_[0], params.position_[1], params.position_[2]);
  p->SetOrientation(params.orientation_[0], params.orientation_[1], params.orientation_[2]);
  p->SetScale(params.scale_[0], params.scale_[1], params.scale_[2]);
}

std::ostream& operator<<(std::ostream& os, const VtkProp3DTransformParams& p) {
  os << p.origin_[0] << " " << p.origin_[1] << " " << p.origin_[2] << std::endl;
  os << p.position_[0] << " " << p.position_[1] << " " << p.position_[2] << std::endl;
  os << p.orientation_[0] << " " << p.orientation_[1] << " " << p.orientation_[2] << std::endl;
  os << p.scale_[0] << " " << p.scale_[1] << " " << p.scale_[2] << std::endl;
  return os;
}

std::istream& operator>>(std::istream& is, VtkProp3DTransformParams& p) {
  is >> p.origin_[0] >> p.origin_[1] >> p.origin_[2];
  is >> p.position_[0] >> p.position_[1] >> p.position_[2];
  is >> p.orientation_[0] >> p.orientation_[1] >> p.orientation_[2];
  is >> p.scale_[0] >> p.scale_[1] >> p.scale_[2];
  return is;
}

// setters
void VtkView::setStlFilePath(std::string stl_file_path_) {
  this->stl_file_path_ = stl_file_path_;
}
void VtkView::setTfParams(std::string registration_params_path_) {
  this->registration_params_path_ = registration_params_path_;
}
void VtkView::setStlNames(std::vector<std::string> stl_names_) {
  this->stl_names_ = stl_names_;
}
void VtkView::setStlColors(std::vector<std::vector<int>> stl_colors_) {
  this->stl_colors_ = stl_colors_;
}
void VtkView::setStlKeys(std::vector<int> stl_keys_) {
  this->stl_keys_ = stl_keys_;
}

void VtkView::setRenderWindow(
    const vtkSmartPointer<vtkGenericOpenGLRenderWindow>& vtk_render_wnd_) {
  this->vtk_render_wnd_ = vtk_render_wnd_;
}

void VtkView::setInteractor(
    const vtkSmartPointer<vtkGenericRenderWindowInteractor>& vtk_interactor_) {
  this->vtk_interactor_ = vtk_interactor_;
}

void VtkView::setInteractorStyle(
    const vtkSmartPointer<vtkInteractorStyleTrackballActor>& vtk_interactor_style) {
  this->vtk_interactor_style_ = vtk_interactor_style;
}

void VtkView::setRenderer(const vtkSmartPointer<vtkRenderer>& vtk_renderer_) {
  this->vtk_renderer_ = vtk_renderer_;
}

vtkSmartPointer<vtkGenericOpenGLRenderWindow>& VtkView::getRenderWindow() {
  return vtk_render_wnd_;
}
vtkSmartPointer<vtkGenericRenderWindowInteractor>& VtkView::getInteractor() {
  return vtk_interactor_;
}
vtkSmartPointer<vtkInteractorStyleTrackballActor>& VtkView::getInteractorStyle() {
  return vtk_interactor_style_;
}

vtkSmartPointer<vtkRenderer>& VtkView::getRenderer() {
  return vtk_renderer_;
}

unsigned int VtkView::getTexture() const {
  return vtk_gl_color_tex_;
}

void VtkView::start() {
  vtkNew<vtkNamedColors> colors;

  const std::string case_directory = stl_file_path_;

  int idx = 0;
  for (const auto stl_model_name : stl_names_) {
    // Reader creation
    const std::string filename = case_directory + stl_model_name + ".stl";
    vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
    reader->SetFileName(filename.c_str());
    reader->Update();

    // Mapper creation
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());

    // Actor creation
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    std::array<unsigned char, 4> arr{{static_cast<unsigned char>(stl_colors_[idx][0]),
                                      static_cast<unsigned char>(stl_colors_[idx][1]),
                                      static_cast<unsigned char>(stl_colors_[idx][2]),
                                      static_cast<unsigned char>(stl_colors_[idx][3])}};
    colors->SetColor(stl_model_name + "_Color", arr.data());
    actor->GetProperty()->SetColor(colors->GetColor3d(stl_model_name + "_Color").GetData());
    actor->SetVisibility(true);
    actor->GetProperty()->SetOpacity(opacity_);
    m_[stl_model_name] = actor;
    part_visible_map_[stl_model_name] = true;

    actor->SetScale(1, 1, -1);
    // Putting together all the different parts
    assembly_->AddPart(actor);
    idx++;
  }

  // -------------------------------------------------------------------------------
  //
  // VTK renderer
  //

  vtk_renderer_ = vtkSmartPointer<vtkRenderer>::New();
  if (!vtk_renderer_) { HOLOSCAN_LOG_ERROR("Failed to initialize vtk renderer"); }
  vtk_renderer_->SetBackground(colors->GetColor3d("BkgColor").GetData());
  vtk_renderer_->SetBackgroundAlpha(0.0);
  vtk_renderer_->AddActor(assembly_);

  // -------------------------------------------------------------------------------
  //
  // VTK render window
  //
  vtk_render_wnd_ = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
  if (!vtk_render_wnd_) { HOLOSCAN_LOG_ERROR("Failed to initialize vtk render window"); }

  int viewportSize[2] = {static_cast<int>(vp_width_), static_cast<int>(vp_height_)};
  vtk_render_wnd_->SetSize(viewportSize);

  vtkSmartPointer<vtkCallbackCommand> isCurrentCallback =
      vtkSmartPointer<vtkCallbackCommand>::New();
  if (!isCurrentCallback) { HOLOSCAN_LOG_ERROR("Failed to initialize vtk callback"); }
  isCurrentCallback->SetCallback(&isCurrentCallbackFn);
  vtk_render_wnd_->AddObserver(vtkCommand::WindowIsCurrentEvent, isCurrentCallback);

  vtk_render_wnd_->SwapBuffersOn();
  vtk_render_wnd_->SetOffScreenRendering(true);
  vtk_render_wnd_->SetFrameBlitModeToNoBlit();
  vtk_render_wnd_->AddRenderer(vtk_renderer_);

  // OIT via Depth Deeling

  vtk_render_wnd_->SetAlphaBitPlanes(1);
  vtk_render_wnd_->SetMultiSamples(0);
  vtk_renderer_->SetUseDepthPeeling(1);
  vtk_renderer_->SetMaximumNumberOfPeels(4);
  vtk_renderer_->SetOcclusionRatio(0.1);

  // -------------------------------------------------------------------------------
  //
  // VTK render window interactor
  //

  vtk_interactor_ = vtkSmartPointer<vtkGenericRenderWindowInteractor>::New();
  if (!vtk_interactor_) { HOLOSCAN_LOG_ERROR("Failed to initialize vtk interactor"); }
  vtk_interactor_->SetRenderWindow(vtk_render_wnd_);
  vtk_interactor_->EnableRenderOff();

  // -------------------------------------------------------------------------------
  //
  // VTK interactor style
  //

  vtk_interactor_style_ = vtkSmartPointer<vtkInteractorStyleTrackballActor>::New();
  if (!vtk_interactor_style_) { HOLOSCAN_LOG_ERROR("Failed to initialize vtk interactor style"); }
  vtk_interactor_->SetInteractorStyle(vtk_interactor_style_);
  vtk_renderer_->ResetCamera();

  update();
}

void VtkView::render() {
  // skip frames with invalid dimensions
  if (vp_width_ <= 0 || vp_height_ <= 0) { return; }

  if (!update_frame_) {
    vtk_render_wnd_->WaitForCompletion();
    return;
  }

  if (realloc_texture) {
    // Free old buffers
    glDeleteTextures(1, &vtk_gl_color_tex_);

    glGenTextures(1, &vtk_gl_color_tex_);
    glBindTexture(GL_TEXTURE_2D, vtk_gl_color_tex_);
    //  To test if VTK access texture allocated with
    //    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, vp_width_, vp_height_, 0, GL_RGBA,
    //    GL_UNSIGNED_BYTE, 0);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, vp_width_, vp_height_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBindTexture(GL_TEXTURE_2D, 0);

    int32_t viewportSize[] = {vp_width_, vp_height_};
    vtk_render_wnd_->InitializeFromCurrentContext();
    vtk_render_wnd_->SetSize(viewportSize);
    vtk_interactor_->SetSize(viewportSize);

    auto vtkfbo = vtk_render_wnd_->GetDisplayFramebuffer();
    vtkfbo->Bind();
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, vtk_gl_color_tex_, 0);
    vtkfbo->UnBind();

    // TODO: check if this ok or to capture current FB from context
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    realloc_texture = false;
  }

  vtk_render_wnd_->Render();
  vtk_render_wnd_->WaitForCompletion();

  update_frame_ = false;
}

void VtkView::stop() {
  vtk_render_wnd_ = nullptr;
  vtk_interactor_ = nullptr;
  vtk_interactor_style_ = nullptr;
  vtk_renderer_ = nullptr;
  if (vtk_gl_color_tex_) {
    glDeleteTextures(1, &vtk_gl_color_tex_);
    vtk_gl_color_tex_ = 0;
  }
}

void VtkView::isCurrentCallbackFn(vtkObject* caller, long unsigned int eventId, void* clientData,
                                  void* callData) {
  bool* isCurrent = static_cast<bool*>(callData);
  *isCurrent = true;
}

int VtkView::onChar(GLFWwindow* wnd, unsigned int codepoint) {
  int alt = glfwGetKey(wnd, GLFW_MOD_ALT);
  int ctrl = glfwGetKey(wnd, GLFW_MOD_CONTROL);
  int shift = glfwGetKey(wnd, GLFW_MOD_SHIFT);
  vtk_interactor_->SetAltKey(alt);

  update();

  vtk_interactor_->SetKeyEventInformation(ctrl, shift, codepoint);
  return vtk_interactor_->InvokeEvent(vtkCommand::CharEvent, nullptr);
}

int VtkView::onEnter(GLFWwindow* wnd, int entered) {
  // TODO: fix  MouseInWindow
  // vtk_interactor_->MouseInWindow = entered;
  update();

  if (entered) return vtk_interactor_->InvokeEvent(vtkCommand::EnterEvent, nullptr);
  else
    return vtk_interactor_->InvokeEvent(vtkCommand::LeaveEvent, nullptr);
}

int VtkView::onMouseMove(GLFWwindow* wnd, double x, double y) {
  // TODO: fix  MouseInWindow
  // if (!vtk_interactor_->MouseInWindow) return 0;
  const int alt = glfwGetKey(wnd, GLFW_MOD_ALT);
  const int ctrl = glfwGetKey(wnd, GLFW_MOD_CONTROL);
  const int shift = glfwGetKey(wnd, GLFW_MOD_SHIFT);

  update();

  vtk_interactor_->SetAltKey(alt);
  vtk_interactor_->SetEventInformation(x, y, ctrl, shift);
  return vtk_interactor_->InvokeEvent(vtkCommand::MouseMoveEvent, nullptr);
}

int VtkView::onMouseButtonCallback(GLFWwindow* wnd, int button, int action, int mods) {
  const int alt = mods & GLFW_MOD_ALT;
  const int ctrl = mods & GLFW_MOD_CONTROL;
  const int shift = mods & GLFW_MOD_SHIFT;
  update();
  vtk_interactor_->SetAltKey(alt);

  double x(0), y(0);
  glfwGetCursorPos(wnd, &x, &y);
  vtk_interactor_->SetEventInformation(x, y, ctrl, shift);

  int retval(0);
  switch (action) {
    case GLFW_PRESS:
      switch (button) {
        case GLFW_MOUSE_BUTTON_LEFT:
          retval = vtk_interactor_->InvokeEvent(vtkCommand::LeftButtonPressEvent, nullptr);
        case GLFW_MOUSE_BUTTON_MIDDLE:
          retval = vtk_interactor_->InvokeEvent(vtkCommand::MiddleButtonPressEvent, nullptr);
        case GLFW_MOUSE_BUTTON_RIGHT:
          retval = vtk_interactor_->InvokeEvent(vtkCommand::RightButtonPressEvent, nullptr);
        default:
          break;
      }
      break;
    case GLFW_RELEASE:
      switch (button) {
        case GLFW_MOUSE_BUTTON_LEFT:
          retval = vtk_interactor_->InvokeEvent(vtkCommand::LeftButtonReleaseEvent, nullptr);
        case GLFW_MOUSE_BUTTON_MIDDLE:
          retval = vtk_interactor_->InvokeEvent(vtkCommand::MiddleButtonReleaseEvent, nullptr);
        case GLFW_MOUSE_BUTTON_RIGHT:
          retval = vtk_interactor_->InvokeEvent(vtkCommand::RightButtonReleaseEvent, nullptr);
        default:
          break;
      }
      break;
    default:
      break;
  }

  return retval;
}

int VtkView::onScrollCallback(GLFWwindow* wnd, double x, double y) {
  const int alt = glfwGetKey(wnd, GLFW_MOD_ALT);
  const int ctrl = glfwGetKey(wnd, GLFW_MOD_CONTROL);
  const int shift = glfwGetKey(wnd, GLFW_MOD_SHIFT);
  update();

  vtk_interactor_->SetAltKey(alt);
  vtk_interactor_->SetControlKey(ctrl);
  vtk_interactor_->SetShiftKey(shift);

  if (y > 0) {
    return vtk_interactor_->InvokeEvent(vtkCommand::MouseWheelForwardEvent, nullptr);
  } else {
    return vtk_interactor_->InvokeEvent(vtkCommand::MouseWheelBackwardEvent, nullptr);
  }
}

int VtkView::onKey(GLFWwindow* wnd, int key, int scancode, int action, int mods) {
  const int alt = mods & GLFW_MOD_ALT;
  const int ctrl = mods & GLFW_MOD_CONTROL;
  const int shift = mods & GLFW_MOD_SHIFT;

  update();

  const char* keysym = glfwGetKeyName(key, scancode);
  int repeat = (action == GLFW_REPEAT);
  int idx = 0;
  for (const auto stl_key : stl_keys_) {
    if ((key == stl_key) && (action == GLFW_RELEASE)) {
      if (part_visible_map_.find(stl_names_[idx]) != part_visible_map_.end()) {
        part_visible_map_[stl_names_[idx]] = !part_visible_map_[stl_names_[idx]];
        if (m_.find(stl_names_[idx]) != m_.end()) {
          m_[stl_names_[idx]]->SetVisibility(part_visible_map_[stl_names_[idx]]);
        }
      }
    }
    idx++;
  }

  if ((key == GLFW_KEY_E) && (action == GLFW_RELEASE)) {
    visibility_all_ = !visibility_all_;
    for (const auto stl_model_name : stl_names_) {
      if (m_.find(stl_model_name) != m_.end()) {
        m_[stl_model_name]->SetVisibility(visibility_all_);
      }
    }
  }

  if ((key == GLFW_KEY_KP_ADD) && (action == GLFW_RELEASE)) {
    opacity_ += 0.05;
    for (const auto stl_model_name : stl_names_) {
      if (m_.find(stl_model_name) != m_.end()) {
        m_[stl_model_name]->GetProperty()->SetOpacity(opacity_);
      }
    }
  }

  if ((key == GLFW_KEY_KP_SUBTRACT) && (action == GLFW_RELEASE)) {
    opacity_ -= 0.05;
    for (const auto stl_model_name : stl_names_) {
      if (m_.find(stl_model_name) != m_.end()) {
        m_[stl_model_name]->GetProperty()->SetOpacity(opacity_);
      }
    }
  }

  VtkProp3DTransformParams transform_params;

  if (ctrl && action == GLFW_RELEASE) {
    if (key == GLFW_KEY_S) {
      transform_params = GetObjectTransform(assembly_);
      std::ofstream myfile;
      myfile.open(registration_params_path_);
      myfile << transform_params << std::endl;
      myfile.close();
    } else if (key == GLFW_KEY_L) {
      std::ifstream t(registration_params_path_);
      std::string s;
      std::string ss;
      std::vector<double> params;
      std::istringstream iss;
      while (getline(t, s)) {
        iss.clear();
        iss.str(s);
        while (getline(iss, ss, ' ')) { params.push_back(std::stod(ss)); }
      }
      if (!params.empty()) {
        transform_params.origin_ = {params.begin(), params.begin() + 3};
        transform_params.position_ = {params.begin() + 3, params.begin() + 6};
        transform_params.orientation_ = {params.begin() + 6, params.begin() + 9};
        transform_params.scale_ = {params.begin() + 9, params.begin() + 12};
        SetObjectTransform(assembly_, transform_params);
      }
    }
  }

  vtk_interactor_->SetKeyEventInformation(ctrl, shift, scancode, repeat, keysym);

  if (action == GLFW_RELEASE)
    return vtk_interactor_->InvokeEvent(vtkCommand::KeyReleaseEvent, nullptr);
  else
    return vtk_interactor_->InvokeEvent(vtkCommand::KeyPressEvent, nullptr);
}

int VtkView::onSize(GLFWwindow* wnd, int w, int h) {
  if (vp_width_ != w || vp_height_ != h) {
    vp_width_ = w;
    vp_height_ = h;
    realloc_texture = true;
  }

  if (!vtk_interactor_) { return 0; }
  update();

  vtk_interactor_->UpdateSize(w, h);
  return vtk_interactor_->InvokeEvent(vtkCommand::ConfigureEvent, nullptr);
}

void VtkView::update() {
  update_frame_ = true;
}

}  // namespace holoscan::orsi
