#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "drake_all.h"

namespace py = pybind11;
using py_rvp = py::return_value_policy;

using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::AngleAxisd;

// WARNING: Hacky setup here, only use for rapid prototyping.
using namespace drake::drake_all;
using Contextd = Context<double>;
using DiagramBuilderd = DiagramBuilder<double>;
using ExternallyAppliedSpatialForced = ExternallyAppliedSpatialForce<double>;
using Framed = Frame<double>;
using InputPortd = InputPort<double>;
using LeafSystemd = LeafSystem<double>;
using MultibodyPlantd = MultibodyPlant<double>;
using OutputPortd = OutputPort<double>;
using RotationalInertiad = RotationalInertia<double>;
using SpatialForced = SpatialForce<double>;
using SpatialInertiad = SpatialInertia<double>;
using SpatialVelocityd = SpatialVelocity<double>;

namespace anzu {

Vector3d rotation_matrix_to_axang3(const RotationMatrixd& R) {
  AngleAxisd axang(R.matrix());
  Vector3d axang3 = axang.angle() * axang.axis();
  return axang3;
}

Matrix3d reexpress_to_matrix(
    const RotationMatrixd& R_AE, const RotationalInertiad& I_BP_E) {
  const auto I_BP_A = I_BP_E.ReExpress(R_AE);
  return I_BP_A.CopyToFullMatrix3();
}

ExternallyAppliedSpatialForced make_force_for_frame(
    const Framed& frame_P, const SpatialForced& F_P_W) {
  return {
    .body_index = frame_P.body().index(),
    .p_BoBq_B = frame_P.GetFixedPoseInBodyFrame().translation(),
    .F_Bq_W = F_P_W
  };
}

struct FloatingBodyFeedback {
  using RotationFunc = std::function<Matrix3d (const RotationMatrixd&)>;

  Matrix3d Kp_xyz;
  Matrix3d Kd_xyz;
  RotationFunc Kp_rot;
  RotationFunc Kd_rot;

  SpatialForced operator()(
      const RigidTransformd& X_WP,
      const SpatialVelocityd& V_WP,
      const RigidTransformd& X_WPdes,
      const SpatialVelocityd& V_WPdes) const {
    // Transform to "negative error": desired w.r.t. actual,
    // expressed in world frame (for applying the force).
    Vector3d p_PPdes_W = X_WPdes.translation() - X_WP.translation();
    const auto R_WP = X_WP.rotation();
    const auto R_PPdes = R_WP.inverse() * X_WPdes.rotation();
    const Vector3d axang3_PPdes = rotation_matrix_to_axang3(R_PPdes);
    const Vector3d axang3_PPdes_W = R_WP * axang3_PPdes;
    const auto V_PPdes_W = V_WPdes - V_WP;
    // Compute wrench components.
    const Vector3d f_P_W = (
        Kp_xyz * p_PPdes_W + Kd_xyz * V_PPdes_W.translational()
    );
    const Vector3d tau_P_W = (
        Kp_rot(R_WP) * axang3_PPdes_W
        + Kd_rot(R_WP) * V_PPdes_W.rotational()
    );
    const SpatialForced F_P_W_feedback(tau_P_W, f_P_W);
    return F_P_W_feedback;
  }
};

struct FloatingBodyFeedforward {
  Vector3d g_W;
  SpatialInertiad M_PPo_P;

  SpatialForced operator()(const RotationMatrixd& R_WP) const {
    const SpatialForced F_Pcm_W(
        Vector3d::Zero(),
        -g_W * M_PPo_P.get_mass()
    );
    const Vector3d p_PoPcm_W = R_WP * M_PPo_P.get_com();
    const Vector3d p_PcmP_W = -p_PoPcm_W;
    const SpatialForced F_P_W_feedforward = F_Pcm_W.Shift(p_PcmP_W);
    return F_P_W_feedforward;
  }
};

bool isfinite(const Eigen::Ref<const Eigen::MatrixXd>& x) {
  return x.array().isFinite().all();
}

/*
Returns:
    SpatialVelocity of frame F's origin w.r.t. frame T, expressed in E
    (which is frame T if unspecified).
*/
SpatialVelocityd get_frame_spatial_velocity(
    const MultibodyPlantd& plant,
    const Contextd& context,
    const Framed& frame_T,
    const Framed& frame_F,
    const Framed* frame_E = nullptr) {  
  if (!frame_E) {
    frame_E = &frame_T;
  };
  MatrixXd Jv_TF_E(6, plant.num_velocities());
  plant.CalcJacobianSpatialVelocity(
      context,
      JacobianWrtVariable::kV,
      frame_F,
      Vector3d::Zero(),
      frame_T,
      *frame_E,
      &Jv_TF_E);
  const auto v = plant.GetVelocities(context);
  const SpatialVelocityd V_TF_E(Jv_TF_E * v);
  return V_TF_E;
}

template <typename T>
auto MakeAlloc() {
  return []() { return AbstractValue::Make<T>(); };
}

template <typename T, typename Func>
auto WrapCalc(Func func) {
  return [func](const Contextd& context, AbstractValue* abstract_value) {
    T& value = abstract_value->get_mutable_value<T>();
    func(context, &value);
  };
}

/**
Controls for the pose of a single-body floating model using frame P w.r.t.
inertial frame T.

Inputs:
    X_TPdes: Desired pose.
    V_TPdes: Desired velocity.
Outputs:
    forces:
        Spatial forces to apply to body to track reference
        trajectories.
*/
class FloatingBodyPoseController : public LeafSystemd {
 public:
  FloatingBodyPoseController(
      const MultibodyPlantd& plant,
      ModelInstanceIndex model_instance,
      const Framed& frame_T,
      const Framed& frame_P,
      bool add_centering = false) {
    this->frame_P = &frame_P;
    auto& frame_W = plant.world_frame();

    const int nx = plant.num_positions(model_instance) + plant.num_velocities(
        model_instance
    );
    // N.B. This will be in the `controller` closure, and thus kept alive.
    context_ = plant.CreateDefaultContext();

    const auto M_PPo_P = plant.CalcSpatialInertia(
        *context_, frame_P, plant.GetBodyIndices(model_instance)
    );
    const auto mass = M_PPo_P.get_mass();
    const auto M = mass * Matrix3d::Identity();
    const auto I_PPo_P = M_PPo_P.CalcRotationalInertia();

    // TODO(eric.cousineau): Hoist these parameters somewhere.
    const double scale = 0.2;
    const double kp = scale * 500;
    const double kd = 2 * sqrt(kp * mass);

    FloatingBodyFeedback feedback{
        .Kp_xyz = kp * M,
        .Kd_xyz = kd * M,
        .Kp_rot = [=](const RotationMatrixd& R_WP) {
          return kp * reexpress_to_matrix(R_WP, I_PPo_P);
        },
        .Kd_rot = [=](const RotationMatrixd& R_WP) {
          return kd * reexpress_to_matrix(R_WP, I_PPo_P);
        }
    };
    FloatingBodyFeedforward feedforward{
        .g_W = plant.gravity_field().gravity_vector(),
        .M_PPo_P = M_PPo_P,
    };

    const double center_kp = 0.2 * kp;
    const double center_kd = 2 * sqrt(center_kp * mass);
    FloatingBodyFeedback centering_feedback{
        .Kp_xyz = center_kp * M,
        .Kd_xyz = center_kd * M,
        .Kp_rot = [=](const RotationMatrixd& R_WP) {
          return center_kp * reexpress_to_matrix(R_WP, I_PPo_P);
        },
        .Kd_rot = [=](const RotationMatrixd& R_WP) {
          return center_kd * reexpress_to_matrix(R_WP, I_PPo_P);
        }
    };
    const auto X_WP_init =
        plant.CalcRelativeTransform(*context_, frame_W, frame_P);

    using Forces = std::vector<ExternallyAppliedSpatialForced>;

    auto control_math = [=, &plant, &frame_W, &frame_T, &frame_P](
        const VectorXd& x,
        const RigidTransformd& X_TPdes,
        const SpatialVelocityd& V_TPdes) {
      plant.SetPositionsAndVelocities(context_.get(), model_instance, x);

      const auto X_WT =
          plant.CalcRelativeTransform(*context_, frame_W, frame_T);
      const auto V_WT = get_frame_spatial_velocity(
          plant, *context_, frame_W, frame_T
      );
      const auto X_WPdes = X_WT * X_TPdes;
      const auto V_WPdes = V_WT.ComposeWithMovingFrameVelocity(
          X_WT.translation(), X_WT.rotation() * V_TPdes
      );

      const auto X_WP =
          plant.CalcRelativeTransform(*context_, frame_W, frame_P);
      const auto V_WP = get_frame_spatial_velocity(
          plant, *context_, frame_W, frame_P
      );

      auto F_P_W_feedback = feedback(X_WP, V_WP, X_WPdes, V_WPdes);
      if (add_centering) {
        const SpatialForced F_P_W_centering = centering_feedback(
            X_WP, V_WP, X_WP_init, SpatialVelocityd::Zero()
        );
        F_P_W_feedback += F_P_W_centering;
      }

      const auto F_P_W_feedforward = feedforward(X_WP.rotation());

      // Package it up.
      const auto F_P_W = F_P_W_feedback + F_P_W_feedforward;
      DRAKE_DEMAND(isfinite(F_P_W.get_coeffs()));
      const ExternallyAppliedSpatialForced external_force =
          make_force_for_frame(frame_P, F_P_W);
      return external_force;
    };

    plant_state_input = &DeclareVectorInputPort("plant_state", nx);
    X_TPdes_input = &DeclareAbstractInputPort(
        "X_TPdes", Value<RigidTransformd>());
    V_TPdes_input = &DeclareAbstractInputPort(
        "V_TPdes", Value<SpatialVelocityd>());

    auto control_calc = [=](
        const Contextd& sys_context,
        Forces* output) {
      const VectorXd x = plant_state_input->Eval(sys_context);
      const auto X_TPdes = X_TPdes_input->Eval<RigidTransformd>(sys_context);
      const auto V_TPdes = V_TPdes_input->Eval<SpatialVelocityd>(sys_context);
      const ExternallyAppliedSpatialForced external_force =
          control_math(x, X_TPdes, V_TPdes);
      output->clear();
      output->push_back(external_force);
    };

    forces_output = &DeclareAbstractOutputPort(
        "forces_output", MakeAlloc<Forces>(), WrapCalc<Forces>(control_calc));
  }

  static FloatingBodyPoseController* AddToBuilder(
      DiagramBuilderd* builder,
      const MultibodyPlantd& plant,
      ModelInstanceIndex model_instance,
      const Framed& frame_T,
      const Framed& frame_P,
      bool add_centering = false,
      bool connect_to_plant = true,
      const std::string& name = "controller") {
    auto owned = std::make_unique<FloatingBodyPoseController>(
        plant,
        model_instance,
        frame_T,
        frame_P,
        add_centering);
    owned->set_name(name);
    auto* controller = owned.get();

    builder->AddSystem(std::move(owned));
    builder->Connect(
        plant.get_state_output_port(model_instance),
        *controller->plant_state_input);
    if (connect_to_plant) {
      builder->Connect(
          *controller->forces_output,
          plant.get_applied_spatial_force_input_port());
    }
    return controller;
  }

  const Framed* frame_P{};

  InputPortd* plant_state_input{};
  InputPortd* X_TPdes_input{};
  InputPortd* V_TPdes_input{};
  OutputPortd* forces_output{};

 private:
  std::unique_ptr<Contextd> context_;
};

class PositionReferenceCoupler : public LeafSystemd {
 public:
  PositionReferenceCoupler(
      const MultibodyPlantd& plant,
      const Framed& frame_T,
      const Framed& frame_A,
      const Framed& frame_B) {
    context_ = plant.CreateDefaultContext();
    const int nx = plant.num_positions() + plant.num_velocities();

    // TODO(eric.cousineau): This is fragile b/c it relies on default
    // context. Should instead do an initialization step / event.
    const auto X_TAinit =
        plant.CalcRelativeTransform(*context_, frame_T, frame_A);
    const auto R_TAinit = X_TAinit.rotation();
    const auto X_TBinit =
        plant.CalcRelativeTransform(*context_, frame_T, frame_B);
    const auto R_TBinit = X_TBinit.rotation();

    struct Coupling {
      struct PoseAndVelocity {
        RigidTransformd X;
        SpatialVelocityd V;
      };

      PoseAndVelocity TAdes;
      PoseAndVelocity TBdes;
    };

    auto compute_coupling = [=, &plant, &frame_T, &frame_A, &frame_B](
          const VectorXd& x) {
      // Compute actual values.
      plant.SetPositionsAndVelocities(context_.get(), x);
      const auto X_TA =
          plant.CalcRelativeTransform(*context_, frame_T, frame_A);
      const auto V_TA =
          get_frame_spatial_velocity(plant, *context_, frame_T, frame_A);
      const auto X_TB =
          plant.CalcRelativeTransform(*context_, frame_T, frame_B);
      const auto V_TB =
          get_frame_spatial_velocity(plant, *context_, frame_T, frame_B);

      // Compute deltas from initial pose.
      const auto X_AinitA = X_TAinit.inverse() * X_TA;
      const auto V_TA_Ainit = R_TAinit.inverse() * V_TA;
      const auto X_BinitB = X_TBinit.inverse() * X_TB;
      const auto V_TB_Binit = R_TBinit.inverse() * V_TB;

      // Compute setpoints via coupling off of deltas.
      const auto X_TAdes = X_TAinit * X_BinitB;
      const auto V_TAdes = R_TAinit * V_TB_Binit;
      const auto X_TBdes = X_TBinit * X_AinitA;
      const auto V_TBdes = R_TBinit * V_TA_Ainit;

      return Coupling{
          .TAdes = {.X = X_TAdes, .V = V_TAdes},
          .TBdes = {.X = X_TBdes, .V = V_TBdes}
      };
    };

    plant_state_input = &DeclareVectorInputPort("plant_state", nx);

    auto calc_value = [=](const Contextd& sys_context) {
      // TODO(eric.cousineau): Place this in cache entry.
      auto x = plant_state_input->Eval(sys_context);
      return compute_coupling(x);
    };

    X_TAdes_output = &DeclareAbstractOutputPort(
        "X_TAdes",
        MakeAlloc<RigidTransformd>(),
        [=](const Contextd& sys_context, AbstractValue* output) {
          output->get_mutable_value<RigidTransformd>() =
              calc_value(sys_context).TAdes.X;
        });
    V_TAdes_output = &DeclareAbstractOutputPort(
        "V_TAdes",
        MakeAlloc<SpatialVelocityd>(),
        [=](const Contextd& sys_context, AbstractValue* output) {
          output->get_mutable_value<SpatialVelocityd>() =
              calc_value(sys_context).TAdes.V;
        });

    X_TBdes_output = &DeclareAbstractOutputPort(
        "X_TBdes",
        MakeAlloc<RigidTransformd>(),
        [=](const Contextd& sys_context, AbstractValue* output) {
          output->get_mutable_value<RigidTransformd>() =
              calc_value(sys_context).TBdes.X;
        });
    V_TBdes_output = &DeclareAbstractOutputPort(
        "V_TBdes",
        MakeAlloc<SpatialVelocityd>(),
        [=](const Contextd& sys_context, AbstractValue* output) {
          output->get_mutable_value<SpatialVelocityd>() =
              calc_value(sys_context).TBdes.V;
        });
  }

  static PositionReferenceCoupler* AddToBuilder(
      DiagramBuilderd* builder,
      const MultibodyPlantd& plant,
      const Framed& frame_T,
      const FloatingBodyPoseController& controller_A,
      const FloatingBodyPoseController& controller_B) {
    auto owned = std::make_unique<PositionReferenceCoupler>(
        plant,
        frame_T,
        *controller_A.frame_P,
        *controller_B.frame_P);
    auto* coupler = owned.get();
    builder->AddSystem(std::move(owned));
    builder->Connect(
        plant.get_state_output_port(), *coupler->plant_state_input);

    builder->Connect(
        *coupler->X_TAdes_output, *controller_A.X_TPdes_input);
    builder->Connect(
        *coupler->V_TAdes_output, *controller_A.V_TPdes_input);

    builder->Connect(
        *coupler->X_TBdes_output, *controller_B.X_TPdes_input);
    builder->Connect(
        *coupler->V_TBdes_output, *controller_B.V_TPdes_input);

    return coupler;
  }

  InputPortd* plant_state_input{};
  OutputPortd* X_TAdes_output{};
  OutputPortd* V_TAdes_output{};
  OutputPortd* X_TBdes_output{};
  OutputPortd* V_TBdes_output{};

 private:
  std::unique_ptr<Contextd> context_;
};


PYBIND11_MODULE(components_cc, m) {
  m.doc() = "Very low effort, hacky transcription of Python code to C++.";

  py::module::import("pydrake.systems.framework");

  {
    using Class = FloatingBodyPoseController;
    py::class_<Class, LeafSystemd>(m, "FloatingBodyPoseController")
        .def(
            py::init<
                const MultibodyPlantd&,
                ModelInstanceIndex,
                const Framed&,
                const Framed&,
                bool>(),
            py::arg("plant"), py::arg("model_instance"),
            py::arg("frame_T"), py::arg("frame_P"),
            py::arg("add_centering") =  false)
        .def_readonly("frame_P", &Class::frame_P)
        .def_readonly("plant_state_input", &Class::plant_state_input)
        .def_readonly("X_TPdes_input", &Class::X_TPdes_input)
        .def_readonly("V_TPdes_input", &Class::V_TPdes_input)
        .def_readonly("forces_output", &Class::forces_output)
        .def_static(
            "AddToBuilder",
            &Class::AddToBuilder,
            py_rvp::reference,
            // Keep liave, ownership: `return` keeps `builder` alive.
            py::keep_alive<0, 1>(),
            py::arg("builder"), py::arg("plant"), py::arg("model_instance"),
            py::arg("frame_T"), py::arg("frame_P"),
            py::arg("add_centering") = false,
            py::arg("connect_to_plant") = true,
            py::arg("name") = std::string("controller"));
  }

  {
    using Class = PositionReferenceCoupler;
    py::class_<Class, LeafSystemd>(m, "PositionReferenceCoupler")
        .def(
            py::init<
              const MultibodyPlantd&,
              const Framed&,
              const Framed&,
              const Framed&>(),
            py::arg("plant"), py::arg("frame_T"),
            py::arg("frame_A"), py::arg("frame_B"))
        .def_readonly("plant_state_input", &Class::plant_state_input)
        .def_readonly("X_TAdes_output", &Class::X_TAdes_output)
        .def_readonly("V_TAdes_output", &Class::V_TAdes_output)
        .def_readonly("X_TBdes_output", &Class::X_TBdes_output)
        .def_readonly("V_TBdes_output", &Class::V_TBdes_output)
        .def_static(
            "AddToBuilder",
            &Class::AddToBuilder,
            py_rvp::reference,
            // Keep liave, ownership: `return` keeps `builder` alive.
            py::keep_alive<0, 1>(),
            py::arg("builder"), py::arg("plant"), py::arg("frame_T"),
            py::arg("controller_A"), py::arg("controller_B"));
  }
}

}  // namespace anzu
