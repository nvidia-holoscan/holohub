#pragma once

#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_ORSI_VTK_VIEW_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_ORSI_VTK_VIEW_HPP_

#include <vtkActor.h>
#include <vtkCallbackCommand.h>
#include <vtkCommand.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkGenericRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkOpenGLFramebufferObject.h>
#include <vtkProp.h>
#include <vtkPropCollection.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

struct GLFWwindow;

namespace holoscan::orsi {  // namespace visualizer_orsi

class VtkView {
 private:
  
  static void isCurrentCallbackFn(vtkObject* caller, long unsigned int eventId, void* clientData,
                                  void* callData);

  vtkSmartPointer<vtkGenericOpenGLRenderWindow> vtk_render_wnd_ = nullptr;
  vtkSmartPointer<vtkGenericRenderWindowInteractor> vtk_interactor_ = nullptr;
  vtkSmartPointer<vtkInteractorStyleTrackballCamera> vtk_interactor_style_ = nullptr;
  vtkSmartPointer<vtkRenderer> vtk_renderer_ = nullptr;

  uint32_t vtk_gl_color_tex_ = 0;

  int32_t vp_width_ = 0;
  int32_t vp_height_ = 0;
  bool realloc_texture = false;
  
  std::map<std::string,vtkSmartPointer<vtkActor>> m_;
  std::map<std::string, bool> part_visible_map_;

  bool visibility_all_ = true;

  float opacity_ = 0.5;
  std::string stl_file_path_;
  std::vector<std::string> stl_names_;
  std::vector<std::vector<int>> stl_colors_;
  std::vector<int> stl_keys_;

 public:
  
  VtkView() = default;
  ~VtkView() = default;

  VtkView(const VtkView& vtkView) = delete;
  VtkView(VtkView&& vtkViewer) = delete;
  VtkView& operator=(const VtkView& vtkView) = delete;

  // event handling via GLFW
  int onChar(GLFWwindow* wnd, unsigned int codepoint);
  int onEnter(GLFWwindow* wnd, int entered);
  int onMouseMove(GLFWwindow* wnd, double x, double y);
  int onMouseButtonCallback(GLFWwindow* wnd, int button, int action, int mods);
  int onScrollCallback(GLFWwindow* wnd, double x, double y);
  int onKey(GLFWwindow* wnd, int key, int scancode, int action, int mods);
  int onSize(GLFWwindow* wnd, int w, int h);

  void start();  // initialize object
  void render();
  void stop();

  void addActor(const vtkSmartPointer<vtkProp>& actor);
  void addActors(const vtkSmartPointer<vtkPropCollection>& actors);
  void removeActor(const vtkSmartPointer<vtkProp>& actor);

  void setStlFilePath(std::string stl_file_path);
  void setStlNames(std::vector<std::string> stl_names);
  void setStlColors(std::vector<std::vector<int>> stl_colors);
  void setStlKeys(std::vector<int> stl_keys);
  void setRenderWindow(const vtkSmartPointer<vtkGenericOpenGLRenderWindow>& renderWindow);
  void setInteractor(const vtkSmartPointer<vtkGenericRenderWindowInteractor>& interactor);
  void setInteractorStyle(
      const vtkSmartPointer<vtkInteractorStyleTrackballCamera>& interactorStyle);

  void setRenderer(const vtkSmartPointer<vtkRenderer>& renderer);

  vtkSmartPointer<vtkGenericOpenGLRenderWindow>& getRenderWindow();
  vtkSmartPointer<vtkGenericRenderWindowInteractor>& getInteractor();
  vtkSmartPointer<vtkInteractorStyleTrackballCamera>& getInteractorStyle();
  vtkSmartPointer<vtkRenderer>& getRenderer();

  unsigned int getTexture() const;
};

}
#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_ORSI_VTK_VIEW_HPP_
