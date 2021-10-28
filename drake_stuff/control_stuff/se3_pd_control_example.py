# Cribbed from Anzu.
# Also derived from
# https://github.com/gizatt/drake_hydra_interact/tree/cce3ecbb

from enum import Enum

import numpy as np

from pydrake.common.eigen_geometry import AngleAxis
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.common.cpp_param import List
from pydrake.common.value import Value
from pydrake.multibody.math import SpatialForce, SpatialVelocity
from pydrake.multibody.plant import ExternallyAppliedSpatialForce

SpatialForceList = Value[List[ExternallyAppliedSpatialForce]]


def rotation_matrix_to_axang3(R):
    assert isinstance(R, RotationMatrix), repr(type(R))
    axang = AngleAxis(R.matrix())
    axang3 = axang.angle() * axang.axis()
    return axang3


class RotationMode(Enum):
    SwapMatrices = 0
    ReExpressAxang3 = 1


def calc_se3_error(X_WP, X_WPdes, rotation_mode=RotationMode.ReExpressAxang3):
    # Option 2: Re-express so(3) error.
    R_WP = X_WP.rotation()
    R_WPdes = X_WPdes.rotation()
    p_PPdes_W = X_WPdes.translation() - X_WP.translation()

    if rotation_mode == RotationMode.ReExpressAxang3:
        R_PPdes = R_WP.inverse() @ R_WPdes
        axang3_PPdes = rotation_matrix_to_axang3(R_PPdes)
        # Re-express so(3) error.
        axang3_PPdes_W = R_WP @ axang3_PPdes
    elif rotation_mode == RotationMode.SwapMatrices:
        # Same as ComputePoseDiffInCommonFrame
        dR_PPdes_W = R_WPdes @ R_WP.inverse()
        axang3_PPdes_W = rotation_matrix_to_axang3(dR_PPdes_W)

    dX_PPdes_W = SpatialVelocity(w=axang3_PPdes_W, v=p_PPdes_W)
    return dX_PPdes_W


def reexpress_and_shift(M_BFo_F, X_GF):
    M_BFo_G = M_BFo_F.ReExpress(X_GF.rotation())
    p_FoGo_G = -X_GF.translation()
    M_BGo_G = M_BFo_G.Shift(p_FoGo_G)
    return M_BGo_G


def compute_composite_inertia(plant, context, frame_F, bodies):
    M_CFo_F = SpatialInertia.Zero()
    for body in bodies:
        X_FB = plant.CalcRelativeTransform(context, frame_F, body.body_frame())
        M_BBo_B = body.CalcSpatialInertiaInBodyFrame(context)
        M_BFo_F = reexpress_and_shift(M_BBo_B, X_FB)
        M_CFo_F += M_BFo_F
    return M_CFo_F


class Se3Controller(LeafSystem):
    def __init__(self, plant, model_instance, frame_T, frame_P):
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._frame_T = frame_T
        self._frame_P = frame_P

        nx = (
            plant.num_positions(model_instance)
            + plant.num_velocities(model_instance)
        )

    def _calc_desired_force(self, context, output):
        plant_context = self._plant_context
        X_TPdes = self._X_TPdes.Eval(context)
        V_TPdes = self._V_TPdes.Eval(context)
        x = self._x.Eval(context)
        M = M_PPo_P.get_mass() * np.eye(3)
        I_PPo_P = M_PPo_P.CalcRotationalInertia()

        frame_W = plant.world_frame()
        plant.SetPositionsAndVelocities(plant_context, model_instance, x)

        bodies = [plant.get_body(x) for x in plant.GetBodyIndices(model_instance)]
        M_PPo_P = compute_composite_inertia(plant, context, frame_P, bodies)

        X_WT = plant.CalcRelativeTransform(plant_context, frame_W, frame_T)
        V_WT = me.get_frame_spatial_velocity(plant, plant_context, frame_W, frame_T)
        X_WPdes = X_WT @ X_TPdes
        V_WPdes = V_WT.ComposeWithMovingFrameVelocity(
            X_WT.translation(), V_TPdes.Rotate(X_WT.rotation())
        )

        X_WP = plant.CalcRelativeTransform(plant_context, frame_W, frame_P)
        V_WP = me.get_frame_spatial_velocity(plant, plant_context, frame_W, frame_P)

        # Compute error terms.
        # N.B. These are "desired - actual", *not* "actual - desired".
        dX_PPdes_W = calc_se3_error(X_WP, X_WPdes)
        V_PPdes_W = V_WPdes - V_WP

        R_WP = X_WP.rotation()
        I_PPo_W = I_PPo_P.ReExpress(R_WP)

        # TODO(eric.cousineau): Hoist gains somewhere.
        Kp_xyz = 100.0
        Kp_rot = 100.0
        Kd_xyz = 20.0
        Kd_rot = 20.0

        # Compute feedback terms.
        tau_P_W = (
            (Kp_rot * I_PPo_W) @ dX_PPdes_W.rotational()
            + (Kd_rot * I_PPo_W) @ V_PPdes_W.rotational()
        )
        f_P_W = (
            (Kp_xyz * M) @ dX_PPdes_W.translational()
            + (Kd_xyz * M) @ V_PPdes_W.translational()
        )
        F_P_W_feedback = SpatialForce(tau=tau_P_W, f=f_P_W)

        # Compute feedforward term (gravity compensation).
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
        return SpatialForceList([external_force])


def build_simulation():
    builder = DiagramBuilder()

    plant = MultibodyPlant(time_step=0.0)
    body = plant.AddRigidBody(...)

    frame_W = plant.world_frame()
    frame_B = body.body_frame()

    controller = Se3Controller(
        plant, default_model_instance(), frame_W, frame_B
    )
    builder.AddSystem(controller)
    builder.Connect(
        plant.get_state_output_port(model_instance),
        controller.GetInputPort("x"),
    )
    builder.Connect(
        controller.get_output_port(),
        plant.get_applied_spatial_force_input_port(),
    )

    diagram = builder.Build()
    return diagram, plant


def run_simulations():
    initial_conditions = [
    ]
