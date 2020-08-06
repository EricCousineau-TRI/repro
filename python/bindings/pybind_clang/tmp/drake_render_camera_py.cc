// Generated from `scan_drake.py` @ da8d779.
#include <pybind11/pybind11.h>

#include "drake/geometry/render/render_engine.h"

namespace py = pybind11;
void auto_bind_ClippingRange(py::module& m) {
  using Class = ClippingRange;
  py::class_<Class>(
      m, "ClippingRange")
      .def(py::init<Class const&>())
      .def(py::init<double, double>())
      .def("far",
          static_cast<double (Class::*)()
                  const>(&Class::far))
      .def("near",
          static_cast<double (Class::*)()
                  const>(&Class::near))

      ;
}
#include <pybind11/pybind11.h>

#include "drake/geometry/render/render_engine.h"

namespace py = pybind11;
void auto_bind_ColorRenderCamera(py::module& m) {
  using Class = ColorRenderCamera;
  py::class_<Class>(
      m, "ColorRenderCamera")
      .def(py::init<Class const&>())
      .def(py::init<RenderCameraCore, bool>())
      .def("core",
          static_cast<RenderCameraCore const& (
              Class::*)() const>(
              &Class::core))
      .def("show_window",
          static_cast<bool (Class::*)()
                  const>(
              &Class::show_window))

      ;
}
#include <pybind11/pybind11.h>

#include "drake/geometry/render/render_engine.h"

namespace py = pybind11;
void auto_bind_DepthRange(py::module& m) {
  using Class = DepthRange;
  py::class_<Class>(
      m, "DepthRange")
      .def(py::init<Class const&>())
      .def(py::init<double, double>())
      .def("max_depth",
          static_cast<double (Class::*)()
                  const>(&Class::max_depth))
      .def("min_depth",
          static_cast<double (Class::*)()
                  const>(&Class::min_depth))

      ;
}
#include <pybind11/pybind11.h>

#include "drake/geometry/render/render_engine.h"

namespace py = pybind11;
void auto_bind_DepthRenderCamera(py::module& m) {
  using Class = DepthRenderCamera;
  py::class_<Class>(
      m, "DepthRenderCamera")
      .def(py::init<Class const&>())
      .def(py::init<RenderCameraCore,
          DepthRange>())
      .def("core",
          static_cast<RenderCameraCore const& (
              Class::*)() const>(
              &Class::core))
      .def("depth_range",
          static_cast<DepthRange const& (
              Class::*)() const>(
              &Class::depth_range))

      ;
}
#include <pybind11/pybind11.h>

#include "drake/geometry/render/render_engine.h"

namespace py = pybind11;
void auto_bind_RenderCameraCore(py::module& m) {
  using Class = RenderCameraCore;
  py::class_<Class>(
      m, "RenderCameraCore")
      .def(py::init<Class const&>())
      .def(py::init<::std::string, CameraInfo,
          ClippingRange,
          RigidTransformd>())
      .def("clipping",
          static_cast<ClippingRange const& (
              Class::*)() const>(
              &Class::clipping))
      .def("intrinsics",
          static_cast<CameraInfo const& (
              Class::*)() const>(
              &Class::intrinsics))
      .def("renderer_name",
          static_cast<::std::string const& (
              Class::*)() const>(
              &Class::renderer_name))
      .def("sensor_pose_in_camera_body",
          static_cast<RigidTransformd const& (
              Class::*)() const>(
              &Class::
                  sensor_pose_in_camera_body))

      ;
}
