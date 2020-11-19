# Cribbed from Anzu.

import numpy as np

from pydrake.common.eigen_geometry import AngleAxis
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.common.cpp_param import List
from pydrake.common.value import Value
from pydrake.multibody.math import SpatialForce, SpatialVelocity
from pydrake.multibody.plant import ExternallyAppliedSpatialForce

# `somewhere` doesn't actually exist.
from somewhere.cc import ComputeCompositeInertia
import somewhere.multibody_extras as me
from somewhere.function_system import FunctionSystem, VectorArg


def reexpress(R_AE, I_BP_E):
    # TODO(eric.cousineau): Use bindings.
    R_AE = R_AE.matrix()
    I_BP_E = I_BP_E.CopyToFullMatrix3()
    I_BP_A = R_AE @ I_BP_E @ R_AE.T
    return I_BP_A


def rotation_matrix_to_axang3(R):
    assert isinstance(R, RotationMatrix), repr(type(R))
    axang = AngleAxis(R.matrix())
    axang3 = axang.angle() * axang.axis()
    return axang3


def build_se3_pd_controller(
    builder, plant, model_instance, frame_T, frame_P,
):
    nx = plant.num_positions(model_instance) + plant.num_velocities(
        model_instance
    )
    # N.B. This will be in `controllers` closure, and thus kept alive.
    context = plant.CreateDefaultContext()
    ForceList = Value[List[ExternallyAppliedSpatialForce]]

    M_PPo_P = ComputeCompositeInertia(
        plant, context, frame_P, me.get_bodies(plant, {model_instance})
    )

    M = M_PPo_P.get_mass() * np.eye(3)
    I_PPo_P = M_PPo_P.CalcRotationalInertia()
    # TODO(eric.cousineau): Force-based or velocity-based control?
    # TODO(eric.cousineau): Hoist these parameters somewhere.
    Kp_xyz = 100.0 * M
    Kd_xyz = 20.0 * M
    Kp_rot = lambda R_WP: 100.0 * reexpress(R_WP, I_PPo_P)
    Kd_rot = lambda R_WP: 20.0 * reexpress(R_WP, I_PPo_P)

    def controller(
        X_TPdes: RigidTransform, V_TPdes: SpatialVelocity, x: VectorArg(nx),
    ) -> ForceList:
        # Cribbed from:
        # https://github.com/gizatt/drake_hydra_interact/tree/cce3ecbb
        frame_W = plant.world_frame()
        plant.SetPositionsAndVelocities(context, model_instance, x)

        X_WT = plant.CalcRelativeTransform(context, frame_W, frame_T)
        V_WT = me.get_frame_spatial_velocity(plant, context, frame_W, frame_T)
        X_WPdes = X_WT @ X_TPdes
        V_WPdes = V_WT.ComposeWithMovingFrameVelocity(
            X_WT.translation(), V_TPdes.Rotate(X_WT.rotation())
        )

        X_WP = plant.CalcRelativeTransform(context, frame_W, frame_P)
        R_WP = X_WP.rotation()
        V_WP = me.get_frame_spatial_velocity(plant, context, frame_W, frame_P)
        # Transform to "negative error": desired w.r.t. actual,
        # expressed in world frame (for applying the force).
        p_PPdes_W = X_WPdes.translation() - X_WP.translation()
        # N.B. We don't project away symmetry here because we're expecting
        # smooth trajectories (for now).
        R_PPdes = R_WP.inverse() @ X_WPdes.rotation()
        axang3_PPdes = rotation_matrix_to_axang3(R_PPdes)
        axang3_PPdes_W = R_WP @ axang3_PPdes
        V_PPdes_W = V_WPdes - V_WP
        # Compute wrench components.
        f_P_W = Kp_xyz @ p_PPdes_W + Kd_xyz @ V_PPdes_W.translational()
        tau_P_W = (
            Kp_rot(R_WP) @ axang3_PPdes_W
            + Kd_rot(R_WP) @ V_PPdes_W.rotational()
        )
        F_P_W_feedback = SpatialForce(tau=tau_P_W, f=f_P_W)
        # Add gravity-compensation term.
        g_W = plant.gravity_field().gravity_vector()
        F_Pcm_W = SpatialForce(tau=[0, 0, 0], f=-g_W * M_PPo_P.get_mass())
        p_PoPcm_W = R_WP @ M_PPo_P.get_com()
        F_P_W_feedforward = F_Pcm_W.Shift(-p_PoPcm_W)
        # Package it up.
        F_P_W = F_P_W_feedback + F_P_W_feedforward
        external_force = ExternallyAppliedSpatialForce()
        external_force.body_index = frame_P.body().index()
        external_force.F_Bq_W = F_P_W
        external_force.p_BoBq_B = (
            frame_P.GetFixedPoseInBodyFrame().translation()
        )
        return ForceList([external_force])

    controller_sys = builder.AddSystem(FunctionSystem(controller))
    builder.Connect(
        plant.get_state_output_port(model_instance),
        controller_sys.GetInputPort("x"),
    )
    builder.Connect(
        controller_sys.get_output_port(),
        plant.get_applied_spatial_force_input_port(),
    )
    X_TPdes_input = controller_sys.GetInputPort("X_TPdes")
    V_TPdes_input = controller_sys.GetInputPort("V_TPdes")
    return X_TPdes_input, V_TPdes_input
