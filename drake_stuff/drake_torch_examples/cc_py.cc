#include <pybind11/pybind11.h>

#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "anzu/common/schema/anzu_serialize_pybind.h"
#include "anzu/.../sys_id_param.h"

namespace py = pybind11;

namespace anzu {
namespace intuitive {

using common::schema::BindSchema;

void BindSysId(py::module m) {
  py::module::import("pydrake.multibody.plant");
  const auto add_str = std::false_type{};
  BindSchema<JointDryFriction>(m, "JointDryFriction", add_str);
  BindSchema<JointViscousFriction>(m, "JointViscousFriction", add_str);
  BindSchema<JointFriction>(m, "JointFriction", add_str);
  BindSchema<JointParam>(m, "JointParam", add_str);
  BindSchema<InertiaParam>(m, "InertiaParam", add_str);
  BindSchema<BodyParam>(m, "BodyParam", add_str);
  BindSchema<ModelParam>(m, "ModelParam", add_str);

  m.def(
      "ApplyJointParam", &ApplyJointParam,
      py::arg("param"), py::arg("context"), py::arg("joint"),
      py::arg("include_damping") = false);
  m.def(
      "ExtractJointParam", &ExtractJointParam,
      py::arg("context"), py::arg("joint"),
      py::arg("include_damping") = false);
  m.def(
      "ApplyBodyParam", &ApplyBodyParam,
      py::arg("param"), py::arg("context"), py::arg("body"));
  m.def(
      "ExtractBodyParam", &ExtractBodyParam,
      py::arg("context"), py::arg("body"));
}

PYBIND11_MODULE(cc, m) {
  BindSysId(m);
}

}  // namespace intuitive
}  // namespace anzu
