#include <pybind11/pybind11.h>

#include "../streaming_client.hpp"
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <pybind11/stl.h>

using std::string_literals::operator""s;
namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline class for handling Python kwargs */
class PyStreamingClientOp : public StreamingClientOp {
 public:
  /* Inherit the constructors */
  using StreamingClientOp::StreamingClientOp;

  /* For handling kwargs in Python */
  PyStreamingClientOp(const py::kwargs& kwargs) : StreamingClientOp() {
    // Set the name from Python if provided
    if (kwargs.contains("name"s)) {
      name_ = kwargs["name"s].cast<std::string>();
    }
  }
};

PYBIND11_MODULE(streaming_client, m) {
  py::class_<StreamingClientOp, PyStreamingClientOp, Operator, std::shared_ptr<StreamingClientOp>>(
      m, "StreamingClientOp")
      .def(py::init<>())
      .def(py::init<const py::kwargs&>());
}

}  // namespace holoscan::ops 