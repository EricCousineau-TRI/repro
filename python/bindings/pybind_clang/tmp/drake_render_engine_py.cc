// Generated using `scane_drake.py` @ 6c0c76f.

#include <pybind11/pybind11.h>

#include "drake/geometry/render/render_engine.h"

class RenderEngine_trampoline : public ::drake::geometry::render::RenderEngine {
 public:
  typedef ::drake::geometry::render::RenderEngine RenderEngine_alias;
  using RenderEngine_alias::RenderEngine;

  void UpdateViewpoint(drake::math::RigidTransformd const& X_WR) override {
    PYBIND11_OVERLOAD_PURE(void, RenderEngine_alias, UpdateViewpoint, X_WR);
  }

  void RenderColorImage(drake::geometry::render::CameraProperties const& camera,
      bool show_window,
      drake::systems::sensors::ImageRgba8U* color_image_out) const override {
    PYBIND11_OVERLOAD_PURE(void, RenderEngine_alias, RenderColorImage, camera,
        show_window, color_image_out);
  }

  void RenderDepthImage(
      drake::geometry::render::DepthCameraProperties const& camera,
      drake::systems::sensors::ImageDepth32F* depth_image_out) const override {
    PYBIND11_OVERLOAD_PURE(
        void, RenderEngine_alias, RenderDepthImage, camera, depth_image_out);
  }

  void RenderLabelImage(drake::geometry::render::CameraProperties const& camera,
      bool show_window,
      drake::systems::sensors::ImageLabel16I* label_image_out) const override {
    PYBIND11_OVERLOAD_PURE(void, RenderEngine_alias, RenderLabelImage, camera,
        show_window, label_image_out);
  }

  bool DoRegisterVisual(drake::geometry::GeometryId id,
      drake::geometry::Shape const& shape,
      drake::geometry::PerceptionProperties const& properties,
      drake::math::RigidTransformd const& X_WG) override {
    PYBIND11_OVERLOAD_PURE(bool, RenderEngine_alias, DoRegisterVisual, id,
        shape, properties, X_WG);
  }

  void DoUpdateVisualPose(drake::geometry::GeometryId id,
      drake::math::RigidTransformd const& X_WG) override {
    PYBIND11_OVERLOAD_PURE(
        void, RenderEngine_alias, DoUpdateVisualPose, id, X_WG);
  }

  bool DoRemoveGeometry(drake::geometry::GeometryId id) override {
    PYBIND11_OVERLOAD_PURE(bool, RenderEngine_alias, DoRemoveGeometry, id);
  }

  std::unique_ptr<drake::geometry::render::RenderEngine,
      std::default_delete<drake::geometry::render::RenderEngine>>
  DoClone() const override {
    PYBIND11_OVERLOAD_PURE(
        std::unique_ptr<drake::geometry::render::RenderEngine,
            std::default_delete<drake::geometry::render::RenderEngine>>,
        RenderEngine_alias, DoClone, );
  }

  void DoRenderColorImage(
      drake::geometry::render::ColorRenderCamera const& camera,
      drake::systems::sensors::ImageRgba8U* color_image_out) const override {
    PYBIND11_OVERLOAD(
        void, RenderEngine_alias, DoRenderColorImage, camera, color_image_out);
  }

  void DoRenderDepthImage(
      drake::geometry::render::DepthRenderCamera const& camera,
      drake::systems::sensors::ImageDepth32F* depth_image_out) const override {
    PYBIND11_OVERLOAD(
        void, RenderEngine_alias, DoRenderDepthImage, camera, depth_image_out);
  }

  void DoRenderLabelImage(
      drake::geometry::render::ColorRenderCamera const& camera,
      drake::systems::sensors::ImageLabel16I* label_image_out) const override {
    PYBIND11_OVERLOAD(
        void, RenderEngine_alias, DoRenderLabelImage, camera, label_image_out);
  }

  void SetDefaultLightPosition(
      Eigen::Matrix<double, 3, 1, 0, 3, 1> const& X_DL) override {
    PYBIND11_OVERLOAD(void, RenderEngine_alias, SetDefaultLightPosition, X_DL);
  }

  void ImplementGeometry(
      drake::geometry::Sphere const& sphere, void* user_data) override {
    PYBIND11_OVERLOAD(
        void, RenderEngine_alias, ImplementGeometry, sphere, user_data);
  }

  void ImplementGeometry(
      drake::geometry::Cylinder const& cylinder, void* user_data) override {
    PYBIND11_OVERLOAD(
        void, RenderEngine_alias, ImplementGeometry, cylinder, user_data);
  }

  void ImplementGeometry(
      drake::geometry::HalfSpace const& half_space, void* user_data) override {
    PYBIND11_OVERLOAD(
        void, RenderEngine_alias, ImplementGeometry, half_space, user_data);
  }

  void ImplementGeometry(
      drake::geometry::Box const& box, void* user_data) override {
    PYBIND11_OVERLOAD(
        void, RenderEngine_alias, ImplementGeometry, box, user_data);
  }

  void ImplementGeometry(
      drake::geometry::Capsule const& capsule, void* user_data) override {
    PYBIND11_OVERLOAD(
        void, RenderEngine_alias, ImplementGeometry, capsule, user_data);
  }

  void ImplementGeometry(
      drake::geometry::Ellipsoid const& ellipsoid, void* user_data) override {
    PYBIND11_OVERLOAD(
        void, RenderEngine_alias, ImplementGeometry, ellipsoid, user_data);
  }

  void ImplementGeometry(
      drake::geometry::Mesh const& mesh, void* user_data) override {
    PYBIND11_OVERLOAD(
        void, RenderEngine_alias, ImplementGeometry, mesh, user_data);
  }

  void ImplementGeometry(
      drake::geometry::Convex const& convex, void* user_data) override {
    PYBIND11_OVERLOAD(
        void, RenderEngine_alias, ImplementGeometry, convex, user_data);
  }

  void ThrowUnsupportedGeometry(std::string const& shape_name) override {
    PYBIND11_OVERLOAD(
        void, RenderEngine_alias, ThrowUnsupportedGeometry, shape_name);
  }
};

class RenderEngine_publicist : public ::drake::geometry::render::RenderEngine {
 public:
  using ::drake::geometry::render::RenderEngine::DoClone;
  using ::drake::geometry::render::RenderEngine::DoRegisterVisual;
  using ::drake::geometry::render::RenderEngine::DoRemoveGeometry;
  using ::drake::geometry::render::RenderEngine::DoRenderColorImage;
  using ::drake::geometry::render::RenderEngine::DoRenderDepthImage;
  using ::drake::geometry::render::RenderEngine::DoRenderLabelImage;
  using ::drake::geometry::render::RenderEngine::DoUpdateVisualPose;
  using ::drake::geometry::render::RenderEngine::
      DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE;
  using ::drake::geometry::render::RenderEngine::GetColorDFromLabel;
  using ::drake::geometry::render::RenderEngine::GetColorIFromLabel;
  using ::drake::geometry::render::RenderEngine::GetRenderLabelOrThrow;
  using ::drake::geometry::render::RenderEngine::LabelFromColor;
  using ::drake::geometry::render::RenderEngine::SetDefaultLightPosition;
  using ::drake::geometry::render::RenderEngine::ThrowIfInvalid;
};

namespace py = pybind11;
void auto_bind_RenderEngine(py::module& m) {
  py::class_<::drake::geometry::render::RenderEngine,
      ::drake::geometry::ShapeReifier, RenderEngine_trampoline>(
      m, "RenderEngine", py::module_local())
      .def(py::init<>())
      .def("Clone",
          static_cast<::std::unique_ptr<drake::geometry::render::RenderEngine,
              std::default_delete<drake::geometry::render::RenderEngine>> (
              ::drake::geometry::render::RenderEngine::*)() const>(
              &::drake::geometry::render::RenderEngine::Clone))
      .def("RegisterVisual",
          static_cast<bool (::drake::geometry::render::RenderEngine::*)(
              ::drake::geometry::GeometryId, ::drake::geometry::Shape const&,
              ::drake::geometry::PerceptionProperties const&,
              ::drake::math::RigidTransformd const&, bool)>(
              &::drake::geometry::render::RenderEngine::RegisterVisual),
          py::arg("id"), py::arg("shape"), py::arg("properties"),
          py::arg("X_WG"), py::arg("needs_updates") = true)
      .def("RemoveGeometry",
          static_cast<bool (::drake::geometry::render::RenderEngine::*)(
              ::drake::geometry::GeometryId)>(
              &::drake::geometry::render::RenderEngine::RemoveGeometry),
          py::arg("id"))
      .def("has_geometry",
          static_cast<bool (::drake::geometry::render::RenderEngine::*)(
              ::drake::geometry::GeometryId) const>(
              &::drake::geometry::render::RenderEngine::has_geometry),
          py::arg("id"))
      .def("UpdateViewpoint",
          static_cast<void (::drake::geometry::render::RenderEngine::*)(
              ::drake::math::RigidTransformd const&)>(
              &::drake::geometry::render::RenderEngine::UpdateViewpoint),
          py::arg("X_WR"))
      .def("RenderColorImage",
          static_cast<void (::drake::geometry::render::RenderEngine::*)(
              ::drake::geometry::render::CameraProperties const&, bool,
              ::drake::systems::sensors::ImageRgba8U*) const>(
              &::drake::geometry::render::RenderEngine::RenderColorImage),
          py::arg("camera"), py::arg("show_window"), py::arg("color_image_out"))
      .def("RenderDepthImage",
          static_cast<void (::drake::geometry::render::RenderEngine::*)(
              ::drake::geometry::render::DepthCameraProperties const&,
              ::drake::systems::sensors::ImageDepth32F*) const>(
              &::drake::geometry::render::RenderEngine::RenderDepthImage),
          py::arg("camera"), py::arg("depth_image_out"))
      .def("RenderLabelImage",
          static_cast<void (::drake::geometry::render::RenderEngine::*)(
              ::drake::geometry::render::CameraProperties const&, bool,
              ::drake::systems::sensors::ImageLabel16I*) const>(
              &::drake::geometry::render::RenderEngine::RenderLabelImage),
          py::arg("camera"), py::arg("show_window"), py::arg("label_image_out"))
      .def("RenderColorImage",
          static_cast<void (::drake::geometry::render::RenderEngine::*)(
              ::drake::geometry::render::ColorRenderCamera const&,
              ::drake::systems::sensors::ImageRgba8U*) const>(
              &::drake::geometry::render::RenderEngine::RenderColorImage),
          py::arg("camera"), py::arg("color_image_out"))
      .def("RenderDepthImage",
          static_cast<void (::drake::geometry::render::RenderEngine::*)(
              ::drake::geometry::render::DepthRenderCamera const&,
              ::drake::systems::sensors::ImageDepth32F*) const>(
              &::drake::geometry::render::RenderEngine::RenderDepthImage),
          py::arg("camera"), py::arg("depth_image_out"))
      .def("RenderLabelImage",
          static_cast<void (::drake::geometry::render::RenderEngine::*)(
              ::drake::geometry::render::ColorRenderCamera const&,
              ::drake::systems::sensors::ImageLabel16I*) const>(
              &::drake::geometry::render::RenderEngine::RenderLabelImage),
          py::arg("camera"), py::arg("label_image_out"))
      .def("default_render_label",
          static_cast<::drake::geometry::render::RenderLabel (
              ::drake::geometry::render::RenderEngine::*)() const>(
              &::drake::geometry::render::RenderEngine::default_render_label))
      .def_static("DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE",
          static_cast<void (*)()>(
              &RenderEngine_publicist::DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE))
      .def("DoRegisterVisual",
          static_cast<bool (::drake::geometry::render::RenderEngine::*)(
              ::drake::geometry::GeometryId, ::drake::geometry::Shape const&,
              ::drake::geometry::PerceptionProperties const&,
              ::drake::math::RigidTransformd const&)>(
              &RenderEngine_publicist::DoRegisterVisual),
          py::arg("id"), py::arg("shape"), py::arg("properties"),
          py::arg("X_WG"))
      .def("DoUpdateVisualPose",
          static_cast<void (::drake::geometry::render::RenderEngine::*)(
              ::drake::geometry::GeometryId,
              ::drake::math::RigidTransformd const&)>(
              &RenderEngine_publicist::DoUpdateVisualPose),
          py::arg("id"), py::arg("X_WG"))
      .def("DoRemoveGeometry",
          static_cast<bool (::drake::geometry::render::RenderEngine::*)(
              ::drake::geometry::GeometryId)>(
              &RenderEngine_publicist::DoRemoveGeometry),
          py::arg("id"))
      .def("DoClone",
          static_cast<::std::unique_ptr<drake::geometry::render::RenderEngine,
              std::default_delete<drake::geometry::render::RenderEngine>> (
              ::drake::geometry::render::RenderEngine::*)() const>(
              &RenderEngine_publicist::DoClone))
      .def("DoRenderColorImage",
          static_cast<void (::drake::geometry::render::RenderEngine::*)(
              ::drake::geometry::render::ColorRenderCamera const&,
              ::drake::systems::sensors::ImageRgba8U*) const>(
              &RenderEngine_publicist::DoRenderColorImage),
          py::arg("camera"), py::arg("color_image_out"))
      .def("DoRenderDepthImage",
          static_cast<void (::drake::geometry::render::RenderEngine::*)(
              ::drake::geometry::render::DepthRenderCamera const&,
              ::drake::systems::sensors::ImageDepth32F*) const>(
              &RenderEngine_publicist::DoRenderDepthImage),
          py::arg("camera"), py::arg("depth_image_out"))
      .def("DoRenderLabelImage",
          static_cast<void (::drake::geometry::render::RenderEngine::*)(
              ::drake::geometry::render::ColorRenderCamera const&,
              ::drake::systems::sensors::ImageLabel16I*) const>(
              &RenderEngine_publicist::DoRenderLabelImage),
          py::arg("camera"), py::arg("label_image_out"))
      .def("GetRenderLabelOrThrow",
          static_cast<::drake::geometry::render::RenderLabel (
              ::drake::geometry::render::RenderEngine::*)(
              ::drake::geometry::PerceptionProperties const&) const>(
              &RenderEngine_publicist::GetRenderLabelOrThrow),
          py::arg("properties"))
      .def_static("LabelFromColor",
          static_cast<::drake::geometry::render::RenderLabel (*)(
              ::drake::systems::sensors::ColorI const&)>(
              &RenderEngine_publicist::LabelFromColor),
          py::arg("color"))
      .def_static("GetColorIFromLabel",
          static_cast<::drake::systems::sensors::ColorI (*)(
              ::drake::geometry::render::RenderLabel const&)>(
              &RenderEngine_publicist::GetColorIFromLabel),
          py::arg("label"))
      .def_static("GetColorDFromLabel",
          static_cast<::drake::systems::sensors::ColorD (*)(
              ::drake::geometry::render::RenderLabel const&)>(
              &RenderEngine_publicist::GetColorDFromLabel),
          py::arg("label"))
      .def("SetDefaultLightPosition",
          static_cast<void (::drake::geometry::render::RenderEngine::*)(
              ::Eigen::Matrix<double, 3, 1, 0, 3, 1> const&)>(
              &RenderEngine_publicist::SetDefaultLightPosition),
          py::arg("X_DL"))
      .def_static("ThrowIfInvalid",
          static_cast<void (*)(::drake::systems::sensors::CameraInfo const&,
              ::drake::systems::sensors::Image<
                  drake::systems::sensors::PixelType::kRgba8U> const*,
              char const*)>(&RenderEngine_publicist::ThrowIfInvalid),
          py::arg("intrinsics"), py::arg("image"), py::arg("image_type"))
      .def_static("ThrowIfInvalid",
          static_cast<void (*)(::drake::systems::sensors::CameraInfo const&,
              ::drake::systems::sensors::Image<
                  drake::systems::sensors::PixelType::kDepth32F> const*,
              char const*)>(&RenderEngine_publicist::ThrowIfInvalid),
          py::arg("intrinsics"), py::arg("image"), py::arg("image_type"))
      .def_static("ThrowIfInvalid",
          static_cast<void (*)(::drake::systems::sensors::CameraInfo const&,
              ::drake::systems::sensors::Image<
                  drake::systems::sensors::PixelType::kLabel16I> const*,
              char const*)>(&RenderEngine_publicist::ThrowIfInvalid),
          py::arg("intrinsics"), py::arg("image"), py::arg("image_type"))

      ;
}
