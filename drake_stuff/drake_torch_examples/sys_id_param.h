/*
Provides structures for serializable model parameter modifications, and
functions to apply and/or extract modifications.
*/

#pragma once

#include <map>
#include <string>

#include <Eigen/Dense>

#include "drake/common/name_value.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/rotational_inertia.h"
#include "drake/multibody/tree/spatial_inertia.h"

namespace anzu {
namespace intuitive {

struct JointDryFriction {
  double max_generalized_force{};
  double v0{};

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(max_generalized_force));
    a->Visit(DRAKE_NVP(v0));
  }
};

struct JointViscousFriction {
  double damping{};

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(damping));
  }
};

struct JointFriction {
  JointDryFriction dry;
  JointViscousFriction viscous;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(dry));
    a->Visit(DRAKE_NVP(viscous));
  }
};

struct JointParam {
  std::string name;
  double reflected_rotor_inertia{};
  JointFriction friction{};

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(reflected_rotor_inertia));
    a->Visit(DRAKE_NVP(friction));
  }
};

void ApplyJointParam(
    const JointParam& param,
    drake::systems::Context<double>*,
    drake::multibody::Joint<double>* joint,
    bool include_damping = false);

JointParam ExtractJointParam(
    drake::systems::Context<double>* context,
    const drake::multibody::Joint<double>* joint,
    bool include_damping = false);

struct InertiaParam {
  double Ixx{};
  double Iyy{};
  double Izz{};
  double Ixy{};
  double Ixz{};
  double Iyz{};

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(Ixx));
    a->Visit(DRAKE_NVP(Iyy));
    a->Visit(DRAKE_NVP(Izz));
    a->Visit(DRAKE_NVP(Ixy));
    a->Visit(DRAKE_NVP(Ixz));
    a->Visit(DRAKE_NVP(Iyz));
  }
};

drake::multibody::RotationalInertia<double>
InertiaParamToRotationalInertia(const InertiaParam& param);

InertiaParam RotationalInertiaToInertiaParam(
    const drake::multibody::RotationalInertia<double>& inertia);

struct BodyParam {
  double mass{};
  Eigen::Vector3d p_BBcm;
  InertiaParam I_BBo_B;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(mass));
    a->Visit(DRAKE_NVP(p_BBcm));
    a->Visit(DRAKE_NVP(I_BBo_B));
  }
};

drake::multibody::SpatialInertia<double>
BodyParamToSpatialInertia(const BodyParam& param);

BodyParam SpatialInertiaToBodyParam(
    const drake::multibody::SpatialInertia<double>& M_BBo_B);

void ApplyBodyParam(
    const BodyParam& param,
    drake::systems::Context<double>* context,
    drake::multibody::RigidBody<double>* body);

BodyParam ExtractBodyParam(
    const drake::systems::Context<double>& context,
    const drake::multibody::RigidBody<double>& body);

struct ModelParam {
  std::map<std::string, JointParam> joint_param;
  std::map<std::string, BodyParam> body_param;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(joint_param));
    a->Visit(DRAKE_NVP(body_param));
  }
};

}  // namespace intuitive
}  // namespace anzu
