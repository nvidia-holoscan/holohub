/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "orsi_visualizer.hpp"

#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"


#include "opengl_utils.hpp"
#include "vis_orsi.hpp"

namespace holoscan::ops::orsi {

// ---------------------------------------------------------------------------------------------
//
// GLFW callbacks
//
static void glfwPrintErrorCallback(int error, const char* msg) {
  if (error == 65539) {
    return;
  }

  HOLOSCAN_LOG_ERROR("GLFW ERROR [%d], msg:  %s\n", error, msg);
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react
// accordingly
// ---------------------------------------------------------------------------------------------
static void glfwProcessInput(GLFWwindow* window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);
}

// whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
static void glfwFramebufferSizeCallback(GLFWwindow* window, int width, int height) {
  OrsiVisualizationOp * op = static_cast<OrsiVisualizationOp*>(glfwGetWindowUserPointer(window));
  if (op) {
    op->onFramebufferSizeCallback(width, height);
  }
}

static void glfwSetWindowFocusCallback(GLFWwindow* window, int focused) {
  OrsiVisualizationOp * op = static_cast<OrsiVisualizationOp*>(glfwGetWindowUserPointer(window));
  if (op) {
    op->onWindowFocusCallback(focused);
  }
}

static void glfwCharCallback(GLFWwindow* window, unsigned int codepoint) {
  OrsiVisualizationOp * op = static_cast<OrsiVisualizationOp*>(glfwGetWindowUserPointer(window));
  if (op) {
    op->onChar(codepoint);
  }
}

static void glfwCursorPosCallback(GLFWwindow* window, double x, double y) {
  OrsiVisualizationOp * op = static_cast<OrsiVisualizationOp*>(glfwGetWindowUserPointer(window));
  if (op) {
    op->onMouseMove(x, y);
  }
}

static void glfwMouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  OrsiVisualizationOp * op = static_cast<OrsiVisualizationOp*>(glfwGetWindowUserPointer(window));
  if (op) {
    op->onMouseButtonCallback(button, action, mods);
  }
}

static void glfwScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
  OrsiVisualizationOp * op = static_cast<OrsiVisualizationOp*>(glfwGetWindowUserPointer(window));
  if (op) {
    op->onScrollCallback(xoffset, yoffset);
  }
}

// GLFW Key Events
static void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  OrsiVisualizationOp * op = static_cast<OrsiVisualizationOp*>(glfwGetWindowUserPointer(window));
  if (op) {
    op->onKeyCallback(key, scancode, action, mods);
  }
}

// --------------------------------------------------------------------------------------------
//
// event handlers
//
void OrsiVisualizationOp::onFramebufferSizeCallback(int width, int height) {
  vp_width_ = width;
  vp_height_ = height;

  pimpl_->onFramebufferSizeCallback(window_, width, height);
}

void OrsiVisualizationOp::onWindowFocusCallback(int focused) {
  pimpl_->onWindowFocusCallback(window_, focused);
}

void OrsiVisualizationOp::onChar(unsigned int codepoint) {
  pimpl_->onChar(window_, codepoint);
}

void OrsiVisualizationOp::onEnter(int entered) {
  pimpl_->onEnter(window_, entered);
}

void OrsiVisualizationOp::onMouseMove(double x, double y) {
  pimpl_->onMouseMove(window_, x, y);
}

void OrsiVisualizationOp::onMouseButtonCallback(int button, int action, int mods) {
  pimpl_->onMouseButtonCallback(window_, button, action, mods);
}

void OrsiVisualizationOp::onScrollCallback(double xoffset, double yoffset) {
  pimpl_->onScrollCallback(window_, xoffset, yoffset);
}

void OrsiVisualizationOp::onKeyCallback(int key, int scancode, int action, int mods) {
  pimpl_->onKeyCallback(window_, key, scancode, action, mods);
}


using holoscan::orsi::vis::VisIntf;
using holoscan::orsi::vis::BufferInfo;

void OrsiVisualizationOp::setup(OperatorSpec& spec) {
  pimpl_.reset(new holoscan::orsi::OrsiVis);

  spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});

  spec.param(
      window_close_scheduling_term_,
      "window_close_scheduling_term",
      "WindowCloseSchedulingTerm",
      "BooleanSchedulingTerm to stop the codelet from ticking after all messages are published.");

  pimpl_->setup(spec);

  cuda_stream_handler_.define_params(spec);
}

void OrsiVisualizationOp::initialize() {
  // Set up prerequisite parameters before calling GXFOperator::initialize()
  auto frag = fragment();

  // Find if there is an argument for 'window_close_scheduling_term'
  auto has_window_close_scheduling_term =
      std::find_if(args().begin(), args().end(), [](const auto& arg) {
        return (arg.name() == "window_close_scheduling_term");
      });
  // Create the BooleanCondition if there is no argument provided.
  if (has_window_close_scheduling_term == args().end()) {
    window_close_scheduling_term_ =
        frag->make_condition<holoscan::BooleanCondition>("window_close_scheduling_term");
    add_arg(window_close_scheduling_term_.get());
  }

  // parent class initialize() call must be after the argument additions above
  Operator::initialize();
}

void OrsiVisualizationOp::start() {
  glfwSetErrorCallback(glfwPrintErrorCallback);

  // Create window
  // -------------
  // initialize and configure
  if (!glfwInit()) {
    HOLOSCAN_LOG_ERROR("Failed to initialize GLFW");
    glfwTerminate();
    throw std::runtime_error("Failed to initialize GLFW");
  }

  constexpr int32_t wnd_width = 1920;
  constexpr int32_t wnd_height = 1080;

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  window_ = glfwCreateWindow(wnd_width, wnd_height, "GXF Video Stream", NULL, NULL);
  if (window_ == NULL) {
    HOLOSCAN_LOG_ERROR("Failed to create GLFW window");
    glfwTerminate();
    throw std::runtime_error("Failed to create GLFW window");
  }
  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, glfwFramebufferSizeCallback);

  glfwSetCharCallback(window_, glfwCharCallback);

  glfwSetCursorPosCallback(window_, glfwCursorPosCallback);
  glfwSetMouseButtonCallback(window_, glfwMouseButtonCallback);
  glfwSetScrollCallback(window_, glfwScrollCallback);

  glfwSetKeyCallback(window_, glfwKeyCallback);

  //  glfwSetWindowSizeCallback(window_, wnSizeCallback);

  glfwMakeContextCurrent(window_);

  // propagate width, height manually as first framebuffer resize callback is not triggered
  onFramebufferSizeCallback(wnd_width, wnd_height);

  // Load all OpenGL function pointers
  if (glewInit() != GLEW_OK) {
    HOLOSCAN_LOG_ERROR("Failed to initialize GLEW - OpenGL Extension Wrangler Library");
    throw std::runtime_error("Failed to initialize GLEW");
  }

  glEnable(GL_DEBUG_OUTPUT);
  // disable frequent GL API notification messages, e.g. buffer usage info, to avoid spamming log
  glDebugMessageControl(
      GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, GL_DEBUG_SEVERITY_NOTIFICATION, 0, 0, GL_FALSE);
  glDebugMessageCallback(OpenGLDebugMessageCallback, 0);
  glDisable(GL_BLEND);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Initialize helper class instancces
  // ----------------------------------------------------------------------------------
  pimpl_->start();

  // cast Condition to BooleanCondition
  auto bool_cond = window_close_scheduling_term_.get();
  bool_cond->enable_tick();
}

void OrsiVisualizationOp::compute(InputContext& op_input, OutputContext& op_output,
                        ExecutionContext& context) {
  std::vector<gxf::Entity> messages_h =
                       op_input.receive<std::vector<gxf::Entity>>("receivers").value();

  // create vector of nvidia::gxf::Entity as expected by the code below
  std::vector<nvidia::gxf::Entity> messages;
  messages.reserve(messages_h.size());
  for (auto& message_h : messages_h) {
    // cast each holoscan::gxf:Entity to its base class
    nvidia::gxf::Entity message = static_cast<nvidia::gxf::Entity>(message_h);
    messages.push_back(message);
  }

     // get the CUDA stream from the input message
  const gxf_result_t result = cuda_stream_handler_.from_messages(context.context(), messages);
  if (result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
  }

  // cast Condition to BooleanCondition
  auto bool_cond = window_close_scheduling_term_.get();
  glfwProcessInput(window_);
  if (glfwWindowShouldClose(window_)) {
    bool_cond->disable_tick();
    return;
  }

  // Set alignment requirement to 1 so that the tensor with any width can work.
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glViewport(0, 0, vp_width_, vp_height_);

  // -------------------------------------------------------------------------------
  //
  //

  std::unordered_map<std::string, holoscan::orsi::vis::BufferInfo> input_buffers;

  for (auto&& message : messages) {
    const auto tensors = message.findAll<nvidia::gxf::Tensor>();
    for (auto&& tensor : tensors.value()) {
      BufferInfo buffer_info;
      if (buffer_info.init(tensor.value()) != GXF_FAILURE) {
        input_buffers.emplace(std::make_pair(tensor->name(), buffer_info));
      }
    }
    const auto video_buffers = message.findAll<nvidia::gxf::VideoBuffer>();
    for (auto&& video_buffer : video_buffers.value()) {
      BufferInfo buffer_info;
      if (buffer_info.init(video_buffer.value()) != GXF_FAILURE) {
        input_buffers.emplace(std::make_pair(video_buffer->name(), buffer_info));
      }
    }
  }


  pimpl_->compute(input_buffers);

  // swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
  // -------------------------------------------------------------------------------
  glfwSwapBuffers(window_);
  glfwPollEvents();
}

void OrsiVisualizationOp::stop() {
    // Free mem allocated in utility classes.
  // ----------------------------------------------------------------------------------

  pimpl_->stop();

  // Free OpenGL buffer and texture memory
  // ----------------------------------------------------------------------------------

  // terminate, clearing all previously allocated GLFW resources.
  if (window_ != nullptr) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
  }
  glfwTerminate();
}

}  // namespace holoscan::ops::orsi
