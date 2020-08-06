// Generated using `scane_drake.py` @ 6c0c76f.

#include <pybind11/pybind11.h>

#include "drake/geometry/render/render_engine.h"

class PyRenderEngine : public RenderEngine {
 public:
  typedef RenderEngine Base;
  using Base::Base;

  void UpdateViewpoint(RigidTransformd const& X_WR) override {
    PYBIND11_OVERLOAD_PURE(void, Base, UpdateViewpoint, X_WR);
  }

  bool DoRegisterVisual(GeometryId id,
      Shape const& shape,
      PerceptionProperties const& properties,
      RigidTransformd const& X_WG) override {
    PYBIND11_OVERLOAD_PURE(bool, Base, DoRegisterVisual, id,
        shape, properties, X_WG);
  }

  void DoUpdateVisualPose(GeometryId id,
      RigidTransformd const& X_WG) override {
    PYBIND11_OVERLOAD_PURE(
        void, Base, DoUpdateVisualPose, id, X_WG);
  }

  bool DoRemoveGeometry(GeometryId id) override {
    PYBIND11_OVERLOAD_PURE(bool, Base, DoRemoveGeometry, id);
  }

  std::unique_ptr<RenderEngine>
  DoClone() const override {
    PYBIND11_OVERLOAD_PURE(
        std::unique_ptr<RenderEngine>,
        Base, DoClone);
  }

  void DoRenderColorImage(
      ColorRenderCamera const& camera,
      ImageRgba8U* color_image_out) const override {
    PYBIND11_OVERLOAD(
        void, Base, DoRenderColorImage, camera, color_image_out);
  }

  void DoRenderDepthImage(
      DepthRenderCamera const& camera,
      ImageDepth32F* depth_image_out) const override {
    PYBIND11_OVERLOAD(
        void, Base, DoRenderDepthImage, camera, depth_image_out);
  }

  void DoRenderLabelImage(
      ColorRenderCamera const& camera,
      ImageLabel16I* label_image_out) const override {
    PYBIND11_OVERLOAD(
        void, Base, DoRenderLabelImage, camera, label_image_out);
  }

  void SetDefaultLightPosition(
      Eigen::Matrix<double, 3, 1, 0, 3, 1> const& X_DL) override {
    PYBIND11_OVERLOAD(void, Base, SetDefaultLightPosition, X_DL);
  }

  using Base::DoClone;
  using Base::DoRegisterVisual;
  using Base::DoRemoveGeometry;
  using Base::DoRenderColorImage;
  using Base::DoRenderDepthImage;
  using Base::DoRenderLabelImage;
  using Base::DoUpdateVisualPose;
  using Base::GetColorDFromLabel;
  using Base::GetColorIFromLabel;
  using Base::GetRenderLabelOrThrow;
  using Base::LabelFromColor;
  using Base::SetDefaultLightPosition;
  using Base::ThrowIfInvalid;
};

namespace py = pybind11;
void auto_bind_RenderEngine(py::module& m) {
  py::class_<RenderEngine, PyRenderEngine>(m, "RenderEngine")
      .def(py::init<>())
      .def("Clone",
          static_cast<::std::unique_ptr<RenderEngine> (
              RenderEngine::*)() const>(
              &RenderEngine::Clone))
      .def("RegisterVisual",
          static_cast<bool (RenderEngine::*)(
              GeometryId, Shape const&,
              PerceptionProperties const&,
              RigidTransformd const&, bool)>(
              &RenderEngine::RegisterVisual),
          py::arg("id"), py::arg("shape"), py::arg("properties"),
          py::arg("X_WG"), py::arg("needs_updates") = true)
      .def("RemoveGeometry",
          static_cast<bool (RenderEngine::*)(
              GeometryId)>(
              &RenderEngine::RemoveGeometry),
          py::arg("id"))
      .def("has_geometry",
          static_cast<bool (RenderEngine::*)(
              GeometryId) const>(
              &RenderEngine::has_geometry),
          py::arg("id"))
      .def("UpdateViewpoint",
          static_cast<void (RenderEngine::*)(
              RigidTransformd const&)>(
              &RenderEngine::UpdateViewpoint),
          py::arg("X_WR"))
      .def("RenderColorImage",
          static_cast<void (RenderEngine::*)(
              CameraProperties const&, bool,
              ImageRgba8U*) const>(
              &RenderEngine::RenderColorImage),
          py::arg("camera"), py::arg("show_window"), py::arg("color_image_out"))
      .def("RenderDepthImage",
          static_cast<void (RenderEngine::*)(
              DepthCameraProperties const&,
              ImageDepth32F*) const>(
              &RenderEngine::RenderDepthImage),
          py::arg("camera"), py::arg("depth_image_out"))
      .def("RenderLabelImage",
          static_cast<void (RenderEngine::*)(
              CameraProperties const&, bool,
              ImageLabel16I*) const>(
              &RenderEngine::RenderLabelImage),
          py::arg("camera"), py::arg("show_window"), py::arg("label_image_out"))
      .def("RenderColorImage",
          static_cast<void (RenderEngine::*)(
              ColorRenderCamera const&,
              ImageRgba8U*) const>(
              &RenderEngine::RenderColorImage),
          py::arg("camera"), py::arg("color_image_out"))
      .def("RenderDepthImage",
          static_cast<void (RenderEngine::*)(
              DepthRenderCamera const&,
              ImageDepth32F*) const>(
              &RenderEngine::RenderDepthImage),
          py::arg("camera"), py::arg("depth_image_out"))
      .def("RenderLabelImage",
          static_cast<void (RenderEngine::*)(
              ColorRenderCamera const&,
              ImageLabel16I*) const>(
              &RenderEngine::RenderLabelImage),
          py::arg("camera"), py::arg("label_image_out"))
      .def("default_render_label",
          static_cast<RenderLabel (
              RenderEngine::*)() const>(
              &RenderEngine::default_render_label))
      .def_static("DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE",
          static_cast<void (*)()>(
              &PyRenderEngine::DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE))
      .def("DoRegisterVisual",
          static_cast<bool (RenderEngine::*)(
              GeometryId, Shape const&,
              PerceptionProperties const&,
              RigidTransformd const&)>(
              &PyRenderEngine::DoRegisterVisual),
          py::arg("id"), py::arg("shape"), py::arg("properties"),
          py::arg("X_WG"))
      .def("DoUpdateVisualPose",
          static_cast<void (RenderEngine::*)(
              GeometryId,
              RigidTransformd const&)>(
              &PyRenderEngine::DoUpdateVisualPose),
          py::arg("id"), py::arg("X_WG"))
      .def("DoRemoveGeometry",
          static_cast<bool (RenderEngine::*)(
              GeometryId)>(
              &PyRenderEngine::DoRemoveGeometry),
          py::arg("id"))
      .def("DoClone",
          static_cast<::std::unique_ptr<RenderEngine> (
              RenderEngine::*)() const>(
              &PyRenderEngine::DoClone))
      .def("DoRenderColorImage",
          static_cast<void (RenderEngine::*)(
              ColorRenderCamera const&,
              ImageRgba8U*) const>(
              &PyRenderEngine::DoRenderColorImage),
          py::arg("camera"), py::arg("color_image_out"))
      .def("DoRenderDepthImage",
          static_cast<void (RenderEngine::*)(
              DepthRenderCamera const&,
              ImageDepth32F*) const>(
              &PyRenderEngine::DoRenderDepthImage),
          py::arg("camera"), py::arg("depth_image_out"))
      .def("DoRenderLabelImage",
          static_cast<void (RenderEngine::*)(
              ColorRenderCamera const&,
              ImageLabel16I*) const>(
              &PyRenderEngine::DoRenderLabelImage),
          py::arg("camera"), py::arg("label_image_out"))
      .def("GetRenderLabelOrThrow",
          static_cast<RenderLabel (
              RenderEngine::*)(
              PerceptionProperties const&) const>(
              &PyRenderEngine::GetRenderLabelOrThrow),
          py::arg("properties"))
      .def_static("LabelFromColor",
          static_cast<RenderLabel (*)(
              ColorI const&)>(
              &PyRenderEngine::LabelFromColor),
          py::arg("color"))
      .def_static("GetColorIFromLabel",
          static_cast<ColorI (*)(
              RenderLabel const&)>(
              &PyRenderEngine::GetColorIFromLabel),
          py::arg("label"))
      .def_static("GetColorDFromLabel",
          static_cast<ColorD (*)(
              RenderLabel const&)>(
              &PyRenderEngine::GetColorDFromLabel),
          py::arg("label"))
      .def("SetDefaultLightPosition",
          static_cast<void (RenderEngine::*)(
              ::Eigen::Matrix<double, 3, 1, 0, 3, 1> const&)>(
              &PyRenderEngine::SetDefaultLightPosition),
          py::arg("X_DL"))
      .def_static("ThrowIfInvalid",
          static_cast<void (*)(CameraInfo const&,
              Image<PixelType::kRgba8U> const*,
              char const*)>(&PyRenderEngine::ThrowIfInvalid),
          py::arg("intrinsics"), py::arg("image"), py::arg("image_type"))
      .def_static("ThrowIfInvalid",
          static_cast<void (*)(CameraInfo const&,
              Image<PixelType::kDepth32F> const*,
              char const*)>(&PyRenderEngine::ThrowIfInvalid),
          py::arg("intrinsics"), py::arg("image"), py::arg("image_type"))
      .def_static("ThrowIfInvalid",
          static_cast<void (*)(CameraInfo const&,
              Image<PixelType::kLabel16I> const*,
              char const*)>(&PyRenderEngine::ThrowIfInvalid),
          py::arg("intrinsics"), py::arg("image"), py::arg("image_type"))

      ;
}
