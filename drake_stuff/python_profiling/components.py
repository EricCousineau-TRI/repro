import dataclasses as dc

import numpy as np

# Prototyping and hacking.
from pydrake.all import *


def rotation_matrix_to_axang3(R):
    assert isinstance(R, RotationMatrix), repr(type(R))
    axang = AngleAxis(R.matrix())
    axang3 = axang.angle() * axang.axis()
    return axang3


def reexpress_to_matrix(R_AE, I_BP_E):
    I_BP_A = I_BP_E.ReExpress(R_AE)
    return I_BP_A.CopyToFullMatrix3()


def make_force_for_frame(frame_P, F_P_W):
    external_force = ExternallyAppliedSpatialForce()
    external_force.body_index = frame_P.body().index()
    external_force.F_Bq_W = F_P_W
    external_force.p_BoBq_B = frame_P.GetFixedPoseInBodyFrame().translation()
    return external_force


@dc.dataclass
class FloatingBodyFeedback:
    Kp_xyz: np.ndarray
    Kd_xyz: np.ndarray
    Kp_rot: callable
    Kd_rot: callable

    def __call__(self, X_WP, V_WP, X_WPdes, V_WPdes):
        # Transform to "negative error": desired w.r.t. actual,
        # expressed in world frame (for applying the force).
        p_PPdes_W = X_WPdes.translation() - X_WP.translation()
        R_WP = X_WP.rotation()
        R_PPdes = R_WP.inverse() @ X_WPdes.rotation()
        axang3_PPdes = rotation_matrix_to_axang3(R_PPdes)
        axang3_PPdes_W = R_WP @ axang3_PPdes
        V_PPdes_W = V_WPdes - V_WP
        # Compute wrench components.
        f_P_W = (
            self.Kp_xyz @ p_PPdes_W + self.Kd_xyz @ V_PPdes_W.translational()
        )
        tau_P_W = (
            self.Kp_rot(R_WP) @ axang3_PPdes_W
            + self.Kd_rot(R_WP) @ V_PPdes_W.rotational()
        )
        F_P_W_feedback = SpatialForce(tau=tau_P_W, f=f_P_W)
        return F_P_W_feedback


@dc.dataclass
class FloatingBodyFeedforward:
    g_W: np.ndarray
    M_PPo_P: SpatialInertia

    def __call__(self, R_WP):
        F_Pcm_W = SpatialForce(
            tau=[0, 0, 0], f=-self.g_W * self.M_PPo_P.get_mass(),
        )
        p_PoPcm_W = R_WP @ self.M_PPo_P.get_com()
        p_PcmP_W = -p_PoPcm_W
        F_P_W_feedforward = F_Pcm_W.Shift(p_PcmP_W)
        return F_P_W_feedforward


def get_frame_spatial_velocity(plant, context, frame_T, frame_F, frame_E=None):
    """
    Returns:
        SpatialVelocity of frame F's origin w.r.t. frame T, expressed in E
        (which is frame T if unspecified).
    """
    if frame_E is None:
        frame_E = frame_T
    Jv_TF_E = plant.CalcJacobianSpatialVelocity(
        context,
        with_respect_to=JacobianWrtVariable.kV,
        frame_B=frame_F,
        p_BP=[0, 0, 0],
        frame_A=frame_T,
        frame_E=frame_E,
    )
    v = plant.GetVelocities(context)
    V_TF_E = SpatialVelocity(Jv_TF_E @ v)
    return V_TF_E


def set_default_frame_pose(plant, frame_F, X_WF):
    assert frame_F.body().is_floating()
    X_FB = frame_F.GetFixedPoseInBodyFrame().inverse()
    X_WB = X_WF @ X_FB
    plant.SetDefaultFreeBodyPose(frame_F.body(), X_WB)


class FloatingBodyPoseController(LeafSystem):
    """
    Controls for the pose of a single-body floating model using frame P w.r.t.
    inertial frame T.

    Inputs:
        X_TPdes: Desired pose.
        V_TPdes: Desired velocity.
    Outputs:
        forces:
            Spatial forces to apply to body to track reference
            trajectories.
    """

    def __init__(
        self, plant, model_instance, frame_T, frame_P, *, add_centering=False
    ):
        super().__init__()
        self.frame_P = frame_P
        frame_W = plant.world_frame()

        nx = plant.num_positions(model_instance) + plant.num_velocities(
            model_instance
        )
        # N.B. This will be in the `controller` closure, and thus kept alive.
        context = plant.CreateDefaultContext()

        M_PPo_P = plant.CalcSpatialInertia(
            context, frame_P, plant.GetBodyIndices(model_instance)
        )
        mass = M_PPo_P.get_mass()
        M = mass * np.eye(3)
        I_PPo_P = M_PPo_P.CalcRotationalInertia()

        # TODO(eric.cousineau): Hoist these parameters somewhere.
        scale = 0.2
        kp = scale * 500
        kd = 2 * np.sqrt(kp * mass)

        feedback = FloatingBodyFeedback(
            Kp_xyz=kp * M,
            Kd_xyz=kd * M,
            Kp_rot=lambda R_WP: kp * reexpress_to_matrix(R_WP, I_PPo_P),
            Kd_rot=lambda R_WP: kd * reexpress_to_matrix(R_WP, I_PPo_P),
        )
        feedforward = FloatingBodyFeedforward(
            g_W=plant.gravity_field().gravity_vector(), M_PPo_P=M_PPo_P,
        )

        if add_centering:
            center_kp = 0.2 * kp
            center_kd = 2 * np.sqrt(center_kp * mass)
            centering_feedback = FloatingBodyFeedback(
                Kp_xyz=center_kp * M,
                Kd_xyz=center_kd * M,
                Kp_rot=lambda R_WP: center_kp
                * reexpress_to_matrix(R_WP, I_PPo_P),
                Kd_rot=lambda R_WP: center_kd
                * reexpress_to_matrix(R_WP, I_PPo_P),
            )
            X_WP_init = plant.CalcRelativeTransform(context, frame_W, frame_P)

        def control_math(x, X_TPdes, V_TPdes):
            # Adapted from:
            # https://github.com/gizatt/drake_hydra_interact/tree/cce3ecbb
            plant.SetPositionsAndVelocities(context, model_instance, x)

            X_WT = plant.CalcRelativeTransform(context, frame_W, frame_T)
            V_WT = get_frame_spatial_velocity(
                plant, context, frame_W, frame_T
            )
            X_WPdes = X_WT @ X_TPdes
            V_WPdes = V_WT.ComposeWithMovingFrameVelocity(
                X_WT.translation(), V_TPdes.Rotate(X_WT.rotation())
            )

            X_WP = plant.CalcRelativeTransform(context, frame_W, frame_P)
            V_WP = get_frame_spatial_velocity(
                plant, context, frame_W, frame_P
            )

            F_P_W_feedback = feedback(X_WP, V_WP, X_WPdes, V_WPdes)
            if add_centering:
                F_P_W_centering = centering_feedback(
                    X_WP, V_WP, X_WP_init, SpatialVelocity.Zero(),
                )
                F_P_W_feedback += F_P_W_centering

            F_P_W_feedforward = feedforward(X_WP.rotation())

            # Package it up.
            F_P_W = F_P_W_feedback + F_P_W_feedforward
            assert np.all(np.isfinite(F_P_W.get_coeffs()))
            external_force = make_force_for_frame(frame_P, F_P_W)
            return external_force

        self.plant_state_input = self.DeclareVectorInputPort("plant_state", nx)
        self.X_TPdes_input = self.DeclareAbstractInputPort(
            "X_TPdes", Value[RigidTransform]()
        )
        self.V_TPdes_input = self.DeclareAbstractInputPort(
            "V_TPdes", Value[SpatialVelocity]()
        )

        def control_calc(sys_context, output):
            plant_state = self.plant_state_input.Eval(sys_context)
            X_TPdes = self.X_TPdes_input.Eval(sys_context)
            V_TPdes = self.V_TPdes_input.Eval(sys_context)
            external_force = control_math(plant_state, X_TPdes, V_TPdes)
            output.set_value([external_force])

        forces_cls = Value[List[ExternallyAppliedSpatialForce]]
        self.forces_output = self.DeclareAbstractOutputPort(
            "forces_output", alloc=forces_cls, calc=control_calc,
        )

    @staticmethod
    def AddToBuilder(
        builder,
        plant,
        model_instance,
        frame_T,
        frame_P,
        *,
        add_centering=False,
        connect_to_plant=True,
        name="controller",
    ):
        controller = FloatingBodyPoseController(
            plant,
            model_instance,
            frame_T,
            frame_P,
            add_centering=add_centering,
        )
        controller.set_name(name)
        builder.AddSystem(controller)
        builder.Connect(
            plant.get_state_output_port(model_instance),
            controller.plant_state_input,
        )
        if connect_to_plant:
            builder.Connect(
                controller.forces_output,
                plant.get_applied_spatial_force_input_port(),
            )
        return controller
