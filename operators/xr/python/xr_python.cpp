/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include <xr_begin_frame_op.hpp>
#include <xr_begin_frame_pydoc.hpp>
#include <xr_composition_layers.hpp>
#include <xr_empty_composition_layer_op.hpp>
#include <xr_empty_composition_layer_pydoc.hpp>
#include <xr_end_frame_op.hpp>
#include <xr_end_frame_pydoc.hpp>
#include <xr_hand_tracker.hpp>
#include <xr_swapchain_cuda.hpp>
#include <../../operator_util.hpp>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/python/core/emitter_receiver_registry.hpp>

namespace py = pybind11;
using pybind11::literals::operator""_a;

namespace holoscan::ops {

class PyXrBeginFrameOp : public XrBeginFrameOp {
 public:
  /* Inherit the constructors */
  using XrBeginFrameOp::XrBeginFrameOp;

  // Define a constructor that fully initializes the object.
  PyXrBeginFrameOp(Fragment* fragment, const py::args& args,
                   std::shared_ptr<holoscan::XrSession> xr_session,
                   const std::string& name = "xr_begin_frame")
      : XrBeginFrameOp(ArgList{Arg{"xr_session", xr_session}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

class PyXrEndFrameOp : public XrEndFrameOp {
 public:
  /* Inherit the constructors */
  using XrEndFrameOp::XrEndFrameOp;

  // Define a constructor that fully initializes the object.
  PyXrEndFrameOp(Fragment* fragment, const py::args& args,
                 std::shared_ptr<holoscan::XrSession> xr_session,
                 const std::string& name = "xr_end_frame")
      : XrEndFrameOp(ArgList{Arg{"xr_session", xr_session}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

class PyXrEmptyCompositionLayerOp : public XrEmptyCompositionLayerOp {
 public:
  /* Inherit the constructors */
  using XrEmptyCompositionLayerOp::XrEmptyCompositionLayerOp;

  // Define a constructor that fully initializes the object.
  PyXrEmptyCompositionLayerOp(Fragment* fragment, const py::args& args,
                               const std::string& name = "xr_empty_composition_layer")
      : XrEmptyCompositionLayerOp() {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

}  // namespace holoscan::ops

namespace holoscan {

class PyXrSession : public XrSession {
 public:
  using XrSession::XrSession;

  PyXrSession(Fragment* fragment, const std::string& application_name = "Holoscan XR",
              uint32_t application_version = 0u, const std::string& name = "xr_session")
      : XrSession(ArgList{
            Arg{"application_name", application_name},
            Arg{"application_version", application_version},
        }) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
  }
};

class PyXrHandTracker : public XrHandTracker {
 public:
  using XrHandTracker::XrHandTracker;

  PyXrHandTracker(Fragment* fragment, std::shared_ptr<holoscan::XrSession> xr_session,
                  xr::HandEXT hand, const std::string& name = "xr_hand_tarcker")
      : XrHandTracker(xr_session, hand) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
  }
};

/*
 * Implements emit and receive for the xr::FrameState type.
 */
template <>
struct emitter_receiver<xr::FrameState> {
  static void emit(py::object& data, const std::string& name, PyOutputContext& op_output,
                   const int64_t acq_timestamp = -1) {
    auto frame_state = data.cast<xr::FrameState>();
    py::gil_scoped_release release;
    op_output.emit<xr::FrameState>(frame_state, name.c_str(), acq_timestamp);
    return;
  }

  static py::object receive(std::any result, const std::string& name, PyInputContext& op_input) {
    HOLOSCAN_LOG_DEBUG("py_receive: xr::FrameState case");
    auto frame_state = std::any_cast<xr::FrameState>(result);
    py::object py_frame_state = py::cast(frame_state);
    return py_frame_state;
  }
};

/*
/* Implements emit and receive for the XrCompositionLayerProjectionStorage type.
 */
template <>
struct emitter_receiver<std::shared_ptr<XrCompositionLayerProjectionStorage>> {
  static void emit(py::object& data, const std::string& name, PyOutputContext& op_output,
                   const int64_t acq_timestamp = -1) {
    auto composition_layer = data.cast<std::shared_ptr<XrCompositionLayerProjectionStorage>>();
    py::gil_scoped_release release;

    op_output.emit(std::static_pointer_cast<xr::CompositionLayerBaseHeader>(composition_layer),
                   name.c_str(),
                   acq_timestamp);
    return;
  }

  static py::object receive(std::any result, const std::string& name, PyInputContext& op_input) {
    HOLOSCAN_LOG_DEBUG("py_receive: XrCompositionLayerProjectionStorage case");
    auto composition_layer =
        std::any_cast<std::shared_ptr<XrCompositionLayerProjectionStorage>>(result);
    py::object py_composition_layer = py::cast(composition_layer);
    return py_composition_layer;
  }
};

PYBIND11_MODULE(_xr, m) {
  py::module_::import("holoscan.core");

  py::class_<ops::XrBeginFrameOp,
             ops::PyXrBeginFrameOp,
             Operator,
             std::shared_ptr<ops::XrBeginFrameOp>>(
      m, "XrBeginFrameOp", doc::XrBeginFrameOp::doc_XrBeginFrameOp_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<holoscan::XrSession>,
                    const std::string&>(),
           "fragment"_a,
           "xr_session"_a,
           "name"_a = "xr_begin_frame_op"s,
           doc::XrBeginFrameOp::doc_XrBeginFrameOp_python);
  py::class_<ops::XrEndFrameOp, ops::PyXrEndFrameOp, Operator, std::shared_ptr<ops::XrEndFrameOp>>(
      m, "XrEndFrameOp", doc::XrEndFrameOp::doc_XrEndFrameOp_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<holoscan::XrSession>,
                    const std::string&>(),
           "fragment"_a,
           "xr_session"_a,
           "name"_a = "xr_end_frame_op"s,
           doc::XrEndFrameOp::doc_XrEndFrameOp_python);
  py::class_<ops::XrEmptyCompositionLayerOp,
             ops::PyXrEmptyCompositionLayerOp,
             Operator,
             std::shared_ptr<ops::XrEmptyCompositionLayerOp>>(
      m,
      "XrEmptyCompositionLayerOp",
      doc::XrEmptyCompositionLayerOp::doc_XrEmptyCompositionLayerOp_python)
      .def(
          py::init<Fragment*, const py::args&, const std::string&>(),
          "fragment"_a,
          "name"_a = "xr_empty_composition_layer_op"s,
          doc::XrEmptyCompositionLayerOp::doc_XrEmptyCompositionLayerOp_python);

  py::class_<XrSession, PyXrSession, holoscan::Resource, std::shared_ptr<XrSession>>(m, "XrSession")
      .def(py::init<Fragment*, const std::string&, uint32_t, const std::string&>(),
           "fragment"_a,
           "application_name"_a = "Holoscan XR",
           "application_version"_a = 0u,
           "name"_a = "xr_session"s)
      .def("locate_view_space", &XrSession::locate_view_space)
      .def("view_configuration_depth_range", &XrSession::view_configuration_depth_range)
      .def("view_configurations", &XrSession::view_configurations);

  py::enum_<xr::HandEXT>(m, "XrHandEXT")
      .value("XR_HAND_LEFT_EXT", xr::HandEXT::Left)
      .value("XR_HAND_RIGHT_EXT", xr::HandEXT::Right);

  py::enum_<xr::HandJointEXT>(m, "HandJointEXT")
      .value("XR_HAND_JOINT_PALM_EXT", xr::HandJointEXT::Palm)
      .value("XR_HAND_JOINT_WRIST_EXT", xr::HandJointEXT::Wrist)
      .value("XR_HAND_JOINT_THUMB_METACARPAL_EXT", xr::HandJointEXT::ThumbMetacarpal)
      .value("XR_HAND_JOINT_THUMB_PROXIMAL_EXT", xr::HandJointEXT::ThumbProximal)
      .value("XR_HAND_JOINT_THUMB_DISTAL_EXT", xr::HandJointEXT::ThumbDistal)
      .value("XR_HAND_JOINT_THUMB_TIP_EXT", xr::HandJointEXT::ThumbTip)
      .value("XR_HAND_JOINT_INDEX_METACARPAL_EXT", xr::HandJointEXT::IndexMetacarpal)
      .value("XR_HAND_JOINT_INDEX_PROXIMAL_EXT", xr::HandJointEXT::IndexProximal)
      .value("XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT", xr::HandJointEXT::IndexIntermediate)
      .value("XR_HAND_JOINT_INDEX_DISTAL_EXT", xr::HandJointEXT::IndexDistal)
      .value("XR_HAND_JOINT_INDEX_TIP_EXT", xr::HandJointEXT::IndexTip)
      .value("XR_HAND_JOINT_MIDDLE_METACARPAL_EXT", xr::HandJointEXT::MiddleMetacarpal)
      .value("XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT", xr::HandJointEXT::MiddleProximal)
      .value("XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT", xr::HandJointEXT::MiddleIntermediate)
      .value("XR_HAND_JOINT_MIDDLE_DISTAL_EXT", xr::HandJointEXT::MiddleDistal)
      .value("XR_HAND_JOINT_MIDDLE_TIP_EXT", xr::HandJointEXT::MiddleTip)
      .value("XR_HAND_JOINT_RING_METACARPAL_EXT", xr::HandJointEXT::RingMetacarpal)
      .value("XR_HAND_JOINT_RING_PROXIMAL_EXT", xr::HandJointEXT::RingProximal)
      .value("XR_HAND_JOINT_RING_INTERMEDIATE_EXT", xr::HandJointEXT::RingIntermediate)
      .value("XR_HAND_JOINT_RING_DISTAL_EXT", xr::HandJointEXT::RingDistal)
      .value("XR_HAND_JOINT_RING_TIP_EXT", xr::HandJointEXT::RingTip)
      .value("XR_HAND_JOINT_LITTLE_METACARPAL_EXT", xr::HandJointEXT::LittleMetacarpal)
      .value("XR_HAND_JOINT_LITTLE_PROXIMAL_EXT", xr::HandJointEXT::LittleProximal)
      .value("XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT", xr::HandJointEXT::LittleIntermediate)
      .value("XR_HAND_JOINT_LITTLE_DISTAL_EXT", xr::HandJointEXT::LittleDistal)
      .value("XR_HAND_JOINT_LITTLE_TIP_EXT,", xr::HandJointEXT::LittleTip);

  py::class_<xr::Flags<xr::SpaceLocationFlagBits, uint64_t>>(m, "SpaceLocationFlagBits");

  py::class_<xr::Quaternionf>(m, "XrQuaternionf", py::buffer_protocol())
      .def_readwrite("x", &xr::Quaternionf::x)
      .def_readwrite("y", &xr::Quaternionf::y)
      .def_readwrite("z", &xr::Quaternionf::z)
      .def_readwrite("w", &xr::Quaternionf::w)
      .def_buffer([](xr::Quaternionf& quaternion) {
        return py::buffer_info(&quaternion.x,
                               sizeof(float),
                               py::format_descriptor<float>::format(),
                               4,
                               {4},
                               {sizeof(float)});
      });

  py::class_<xr::Vector3f>(m, "XrVector3f", py::buffer_protocol())
      .def_readwrite("x", &xr::Vector3f::x)
      .def_readwrite("y", &xr::Vector3f::y)
      .def_readwrite("z", &xr::Vector3f::z)
      .def_buffer([](xr::Vector3f& vector) {
        return py::buffer_info(&vector.x,
                               sizeof(float),
                               py::format_descriptor<float>::format(),
                               3,
                               {3},
                               {sizeof(float)});
      });

  py::class_<xr::Posef>(m, "XrPosef")
      .def_readwrite("position", &xr::Posef::position)
      .def_readwrite("orientation", &xr::Posef::orientation);

  py::class_<xr::Fovf>(m, "XrFovf")
      .def_readwrite("angleLeft", &xr::Fovf::angleLeft)
      .def_readwrite("angleRight", &xr::Fovf::angleRight)
      .def_readwrite("angleUp", &xr::Fovf::angleUp)
      .def_readwrite("angleDown", &xr::Fovf::angleDown);

  py::class_<xr::HandJointLocationEXT>(m, "XrHandJointLocationEXT")
      .def_readwrite("locationFlags", &xr::HandJointLocationEXT::locationFlags)
      .def_readwrite("pose", &xr::HandJointLocationEXT::pose)
      .def_readwrite("radius", &xr::HandJointLocationEXT::radius);

  py::class_<xr::SpaceLocation>(m, "XrSpaceLocation")
      .def_readwrite("locationFlags", &xr::SpaceLocation::locationFlags)
      .def_readwrite("pose", &xr::SpaceLocation::pose);

  py::class_<xr::Rect2Di>(m, "XrRect2Di")
      .def_readwrite("offset", &xr::Rect2Di::offset)
      .def_readwrite("extent", &xr::Rect2Di::extent);

  py::class_<xr::Offset2Di>(m, "XrOffset2Di")
      .def_readwrite("x", &xr::Offset2Di::x)
      .def_readwrite("y", &xr::Offset2Di::y);

  py::class_<xr::Extent2Di>(m, "XrExtent2Di")
      .def_readwrite("width", &xr::Extent2Di::width)
      .def_readwrite("height", &xr::Extent2Di::height);

  py::class_<xr::SwapchainSubImage>(m, "XrSwapchainSubImage")
      .def_readwrite("swapchain", &xr::SwapchainSubImage::swapchain)
      .def_readwrite("imageRect", &xr::SwapchainSubImage::imageRect)
      .def_readwrite("imageArrayIndex", &xr::SwapchainSubImage::imageArrayIndex);

  py::class_<xr::FrameState>(m, "XrFrameState")
      .def_readwrite("predictedDisplayTime", &xr::FrameState::predictedDisplayTime)
      .def_readwrite("predictedDisplayPeriod", &xr::FrameState::predictedDisplayPeriod)
      .def_readwrite("shouldRender", &xr::FrameState::shouldRender);

  py::class_<xr::ViewConfigurationDepthRangeEXT>(m, "ViewConfigurationDepthRangeEXT")
      .def_readwrite("recommendedNearZ", &xr::ViewConfigurationDepthRangeEXT::recommendedNearZ)
      .def_readwrite("minNearZ", &xr::ViewConfigurationDepthRangeEXT::minNearZ)
      .def_readwrite("recommendedFarZ", &xr::ViewConfigurationDepthRangeEXT::recommendedFarZ)
      .def_readwrite("maxFarZ", &xr::ViewConfigurationDepthRangeEXT::maxFarZ);

  py::class_<xr::ViewConfigurationView>(m, "XrViewConfigurationView")
      .def_readwrite("recommendedImageRectWidth",
                     &xr::ViewConfigurationView::recommendedImageRectWidth)
      .def_readwrite("recommendedImageRectHeight",
                     &xr::ViewConfigurationView::recommendedImageRectHeight);

  py::class_<xr::CompositionLayerProjectionView>(m, "XrCompositionLayerProjectionView")
      .def_readwrite("pose", &xr::CompositionLayerProjectionView::pose)
      .def_readwrite("fov", &xr::CompositionLayerProjectionView::fov)
      .def_readwrite("subImage", &xr::CompositionLayerProjectionView::subImage);

  py::class_<xr::CompositionLayerDepthInfoKHR>(m, "XrCompositionLayerDepthInfoKHR")
      .def_readwrite("subImage", &xr::CompositionLayerDepthInfoKHR::subImage)
      .def_readwrite("minDepth", &xr::CompositionLayerDepthInfoKHR::minDepth)
      .def_readwrite("maxDepth", &xr::CompositionLayerDepthInfoKHR::maxDepth)
      .def_readwrite("nearZ", &xr::CompositionLayerDepthInfoKHR::nearZ)
      .def_readwrite("farZ", &xr::CompositionLayerDepthInfoKHR::farZ);

  py::class_<XrHandTracker, PyXrHandTracker, holoscan::Resource, std::shared_ptr<XrHandTracker>>(
      m, "XrHandTracker")
      .def(py::init<Fragment*,
                    std::shared_ptr<holoscan::XrSession>,
                    xr::HandEXT,
                    const std::string&>(),
           "fragment"_a,
           "xr_session"_a,
           "hand"_a,
           "name"_a = "xr_hand_tracker"s)
      .def("locate_hand_joints", &XrHandTracker::locate_hand_joints);

  py::enum_<XrSwapchainCuda::Format>(m, "XrSwapchainCudaFormat")
      .value("R8G8B8A8_SRGB", XrSwapchainCuda::Format::R8G8B8A8_SRGB)
      .value("R8G8B8A8_UNORM", XrSwapchainCuda::Format::R8G8B8A8_UNORM)
      .value("D16_UNORM", XrSwapchainCuda::Format::D16_UNORM)
      .value("D32_SFLOAT", XrSwapchainCuda::Format::D32_SFLOAT);

  py::class_<XrSwapchainCuda, std::shared_ptr<XrSwapchainCuda>>(m, "XrSwapchainCuda")
      .def(py::init<XrSession&, XrSwapchainCuda::Format, uint32_t, uint32_t>(),
           "session"_a,
           "format"_a,
           "width"_a,
           "height"_a)
      .def("acquire", &XrSwapchainCuda::acquire)
      .def("release",
           [](XrSwapchainCuda& self, intptr_t cuda_stream) {
             self.release(reinterpret_cast<cudaStream_t>(cuda_stream));
           })
      .def("width", &XrSwapchainCuda::width)
      .def("height", &XrSwapchainCuda::height);

  py::class_<XrCompositionLayerProjectionStorage,
             std::shared_ptr<XrCompositionLayerProjectionStorage>>(
      m, "XrCompositionLayerProjectionStorage")
      .def_static("create_for_frame",
                  &XrCompositionLayerProjectionStorage::create_for_frame,
                  "xr_frame_state"_a,
                  "xr_session"_a,
                  "color_swapchain"_a,
                  "depth_swapchain"_a)
      .def_readwrite("views", &XrCompositionLayerProjectionStorage::views)
      .def_readwrite("depth_info", &XrCompositionLayerProjectionStorage::depth_info);

  // Register custom type emitter-receivers with the EmitterReceiverRegistry.
  m.def("register_types", [](EmitterReceiverRegistry& registry) {
    registry.add_emitter_receiver<xr::FrameState>("xr::FrameState"s);
    registry.add_emitter_receiver<std::shared_ptr<XrCompositionLayerProjectionStorage>>(
        "XrCompositionLayerProjectionStorage"s);
  });
}  // PYBIND11_MODULE NOLINT

}  // namespace holoscan
