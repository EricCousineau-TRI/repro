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
      Vector3d const& X_DL) override {
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
  using Class = RenderEngine;
  py::class_<Class, PyRenderEngine>(m, "RenderEngine")
      .def(py::init<>())
      .def("Clone",
          static_cast<::std::unique_ptr<Class> (
              Class::*)() const>(
              &Class::Clone))
      .def("RegisterVisual",
          static_cast<bool (Class::*)(
              GeometryId, Shape const&,
              PerceptionProperties const&,
              RigidTransformd const&, bool)>(
              &Class::RegisterVisual),
          py::arg("id"), py::arg("shape"), py::arg("properties"),
          py::arg("X_WG"), py::arg("needs_updates") = true)
      .def("RemoveGeometry",
          static_cast<bool (Class::*)(
              GeometryId)>(
              &Class::RemoveGeometry),
          py::arg("id"))
      .def("has_geometry",
          static_cast<bool (Class::*)(
              GeometryId) const>(
              &Class::has_geometry),
          py::arg("id"))
      .def("UpdateViewpoint",
          static_cast<void (Class::*)(
              RigidTransformd const&)>(
              &Class::UpdateViewpoint),
          py::arg("X_WR"))
      .def("RenderColorImage",
          static_cast<void (Class::*)(
              ColorRenderCamera const&,
              ImageRgba8U*) const>(
              &Class::RenderColorImage),
          py::arg("camera"), py::arg("color_image_out"))
      .def("RenderDepthImage",
          static_cast<void (Class::*)(
              DepthRenderCamera const&,
              ImageDepth32F*) const>(
              &Class::RenderDepthImage),
          py::arg("camera"), py::arg("depth_image_out"))
      .def("RenderLabelImage",
          static_cast<void (Class::*)(
              ColorRenderCamera const&,
              ImageLabel16I*) const>(
              &Class::RenderLabelImage),
          py::arg("camera"), py::arg("label_image_out"))
      .def("default_render_label",
          static_cast<RenderLabel (
              Class::*)() const>(
              &Class::default_render_label))
      .def("DoRegisterVisual",
          static_cast<bool (Class::*)(
              GeometryId, Shape const&,
              PerceptionProperties const&,
              RigidTransformd const&)>(
              &PyRenderEngine::DoRegisterVisual),
          py::arg("id"), py::arg("shape"), py::arg("properties"),
          py::arg("X_WG"))
      .def("DoUpdateVisualPose",
          static_cast<void (Class::*)(
              GeometryId,
              RigidTransformd const&)>(
              &PyRenderEngine::DoUpdateVisualPose),
          py::arg("id"), py::arg("X_WG"))
      .def("DoRemoveGeometry",
          static_cast<bool (Class::*)(
              GeometryId)>(
              &PyRenderEngine::DoRemoveGeometry),
          py::arg("id"))
      .def("DoClone",
          static_cast<::std::unique_ptr<Class> (
              Class::*)() const>(
              &PyRenderEngine::DoClone))
      .def("DoRenderColorImage",
          static_cast<void (Class::*)(
              ColorRenderCamera const&,
              ImageRgba8U*) const>(
              &PyRenderEngine::DoRenderColorImage),
          py::arg("camera"), py::arg("color_image_out"))
      .def("DoRenderDepthImage",
          static_cast<void (Class::*)(
              DepthRenderCamera const&,
              ImageDepth32F*) const>(
              &PyRenderEngine::DoRenderDepthImage),
          py::arg("camera"), py::arg("depth_image_out"))
      .def("DoRenderLabelImage",
          static_cast<void (Class::*)(
              ColorRenderCamera const&,
              ImageLabel16I*) const>(
              &PyRenderEngine::DoRenderLabelImage),
          py::arg("camera"), py::arg("label_image_out"))
      .def("GetRenderLabelOrThrow",
          static_cast<RenderLabel (
              Class::*)(
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
          static_cast<void (Class::*)(
              Vector3d const&)>(
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
          py::arg("intrinsics"), py::arg("image"), py::arg("image_type"));
}
