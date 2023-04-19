#include "intuitive/controllers/sys_id_param.h"

#include "drake/multibody/tree/revolute_joint.h"

namespace anzu {
namespace intuitive {

void ApplyJointParam(
    const JointParam& param,
    drake::systems::Context<double>*,
    drake::multibody::Joint<double>* joint,
    bool include_damping) {
  auto* revolute =
      dynamic_cast<drake::multibody::RevoluteJoint<double>*>(joint);
  DRAKE_DEMAND(revolute != nullptr);
  if (include_damping) {
    // TODO(eric.cousineau): This should be a parameter.
    revolute->set_default_damping(param.friction.viscous.damping);
  } else {
    revolute->set_default_damping(0.0);
  }
}

JointParam ExtractJointParam(
    drake::systems::Context<double>* context,
    const drake::multibody::Joint<double>* joint,
    bool include_damping) {
  // TODO(eric.cousineau): Ensure these are MbP param.
  const auto* revolute =
      dynamic_cast<const drake::multibody::RevoluteJoint<double>*>(joint);
  DRAKE_DEMAND(revolute != nullptr);
  JointParam param = {
      .name = joint->name(),
      .reflected_rotor_inertia = 0.0,
      .friction = {
          .dry = {
              .max_generalized_force = 0.0,
              .v0 = 1.0
          },
          .viscous = {.damping = 0.0}
      }
  };
  if (include_damping) {
    param.friction.viscous.damping = revolute->damping();
  }
  return param;
}

drake::multibody::RotationalInertia<double>
InertiaParamToRotationalInertia(const InertiaParam& param) {
  return drake::multibody::RotationalInertia<double>(
      param.Ixx, param.Iyy, param.Izz, param.Ixy, param.Ixz, param.Iyz);
}

InertiaParam RotationalInertiaToInertiaParam(
    const drake::multibody::RotationalInertia<double>& inertia) {
  const auto moments = inertia.get_moments();
  const auto products = inertia.get_products();
  return InertiaParam{
      .Ixx = moments(0), .Iyy = moments(1), .Izz = moments(2),
      .Ixy = products(0), .Ixz = products(1), .Iyz = products(2)};
}

drake::multibody::SpatialInertia<double>
BodyParamToSpatialInertia(const BodyParam& param) {
  const auto I_BBo_B = InertiaParamToRotationalInertia(param.I_BBo_B);
  const drake::multibody::UnitInertia<double> G_BBo_B(I_BBo_B / param.mass);
  const auto M_BBo_B = drake::multibody::SpatialInertia<double>(
      param.mass, param.p_BBcm, G_BBo_B);
  return M_BBo_B;
}

BodyParam SpatialInertiaToBodyParam(
    const drake::multibody::SpatialInertia<double>& M_BBo_B) {
  const auto I_BBo_B = RotationalInertiaToInertiaParam(
      M_BBo_B.CalcRotationalInertia());
  return BodyParam{
      .mass = M_BBo_B.get_mass(),
      .p_BBcm = M_BBo_B.get_com(),
      .I_BBo_B = I_BBo_B};
}

void ApplyBodyParam(
    const BodyParam& param,
    drake::systems::Context<double>* context,
    drake::multibody::RigidBody<double>* body) {
  const auto M_BBo_B = BodyParamToSpatialInertia(param);
  body->SetSpatialInertiaInBodyFrame(context, M_BBo_B);
}

BodyParam ExtractBodyParam(
    const drake::systems::Context<double>& context,
    const drake::multibody::RigidBody<double>& body) {
  const auto M_BBo_B = body.CalcSpatialInertiaInBodyFrame(context);
  return SpatialInertiaToBodyParam(M_BBo_B);
}

}  // namespace intuitive
}  // namespace anzu
