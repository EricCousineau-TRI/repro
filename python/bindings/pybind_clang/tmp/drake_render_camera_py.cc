// Generated from `scan_drake.py` @ da8d779.
#include <pybind11/pybind11.h>

#include "drake/geometry/render/render_engine.h"

namespace py = pybind11;
void auto_bind_ClippingRange(py::module& m) {
  py::class_<::drake::geometry::render::ClippingRange>(
      m, "ClippingRange", py::module_local())
      .def(py::init<::drake::geometry::render::ClippingRange const&>())
      .def(py::init<double, double>())
      .def_static("DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE",
          static_cast<void (*)()>(&::drake::geometry::render::ClippingRange::
                                      DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE))
      .def("far",
          static_cast<double (::drake::geometry::render::ClippingRange::*)()
                  const>(&::drake::geometry::render::ClippingRange::far))
      .def("near",
          static_cast<double (::drake::geometry::render::ClippingRange::*)()
                  const>(&::drake::geometry::render::ClippingRange::near))

      ;
}
#include <pybind11/pybind11.h>

#include "drake/geometry/render/render_engine.h"

namespace py = pybind11;
void auto_bind_ColorRenderCamera(py::module& m) {
  py::class_<::drake::geometry::render::ColorRenderCamera>(
      m, "ColorRenderCamera", py::module_local())
      .def(py::init<::drake::geometry::render::ColorRenderCamera const&>())
      .def(py::init<::drake::geometry::render::RenderCameraCore, bool>())
      .def_static("DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE",
          static_cast<void (*)()>(
              &::drake::geometry::render::ColorRenderCamera::
                  DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE))
      .def("core",
          static_cast<::drake::geometry::render::RenderCameraCore const& (
              ::drake::geometry::render::ColorRenderCamera::*)() const>(
              &::drake::geometry::render::ColorRenderCamera::core))
      .def("show_window",
          static_cast<bool (::drake::geometry::render::ColorRenderCamera::*)()
                  const>(
              &::drake::geometry::render::ColorRenderCamera::show_window))

      ;
}
#include <pybind11/pybind11.h>

#include "drake/geometry/render/render_engine.h"

namespace py = pybind11;
void auto_bind_DepthRange(py::module& m) {
  py::class_<::drake::geometry::render::DepthRange>(
      m, "DepthRange", py::module_local())
      .def(py::init<::drake::geometry::render::DepthRange const&>())
      .def(py::init<double, double>())
      .def_static("DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE",
          static_cast<void (*)()>(&::drake::geometry::render::DepthRange::
                                      DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE))
      .def("max_depth",
          static_cast<double (::drake::geometry::render::DepthRange::*)()
                  const>(&::drake::geometry::render::DepthRange::max_depth))
      .def("min_depth",
          static_cast<double (::drake::geometry::render::DepthRange::*)()
                  const>(&::drake::geometry::render::DepthRange::min_depth))

      ;
}
#include <pybind11/pybind11.h>

#include "drake/geometry/render/render_engine.h"

namespace py = pybind11;
void auto_bind_DepthRenderCamera(py::module& m) {
  py::class_<::drake::geometry::render::DepthRenderCamera>(
      m, "DepthRenderCamera", py::module_local())
      .def(py::init<::drake::geometry::render::DepthRenderCamera const&>())
      .def(py::init<::drake::geometry::render::RenderCameraCore,
          ::drake::geometry::render::DepthRange>())
      .def_static("DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE",
          static_cast<void (*)()>(
              &::drake::geometry::render::DepthRenderCamera::
                  DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE))
      .def("core",
          static_cast<::drake::geometry::render::RenderCameraCore const& (
              ::drake::geometry::render::DepthRenderCamera::*)() const>(
              &::drake::geometry::render::DepthRenderCamera::core))
      .def("depth_range",
          static_cast<::drake::geometry::render::DepthRange const& (
              ::drake::geometry::render::DepthRenderCamera::*)() const>(
              &::drake::geometry::render::DepthRenderCamera::depth_range))

      ;
}
#include <pybind11/pybind11.h>

#include "drake/geometry/render/render_engine.h"

namespace py = pybind11;
void auto_bind_RenderCameraCore(py::module& m) {
  py::class_<::drake::geometry::render::RenderCameraCore>(
      m, "RenderCameraCore", py::module_local())
      .def(py::init<::drake::geometry::render::RenderCameraCore const&>())
      .def(py::init<::std::string, ::drake::systems::sensors::CameraInfo,
          ::drake::geometry::render::ClippingRange,
          ::drake::math::RigidTransformd>())
      .def_static("DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE",
          static_cast<void (*)()>(&::drake::geometry::render::RenderCameraCore::
                                      DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE))
      .def("clipping",
          static_cast<::drake::geometry::render::ClippingRange const& (
              ::drake::geometry::render::RenderCameraCore::*)() const>(
              &::drake::geometry::render::RenderCameraCore::clipping))
      .def("intrinsics",
          static_cast<::drake::systems::sensors::CameraInfo const& (
              ::drake::geometry::render::RenderCameraCore::*)() const>(
              &::drake::geometry::render::RenderCameraCore::intrinsics))
      .def("renderer_name",
          static_cast<::std::string const& (
              ::drake::geometry::render::RenderCameraCore::*)() const>(
              &::drake::geometry::render::RenderCameraCore::renderer_name))
      .def("sensor_pose_in_camera_body",
          static_cast<::drake::math::RigidTransformd const& (
              ::drake::geometry::render::RenderCameraCore::*)() const>(
              &::drake::geometry::render::RenderCameraCore::
                  sensor_pose_in_camera_body))

      ;
}
