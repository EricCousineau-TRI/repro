import numpy as np

from pydrake.common.eigen_geometry import AngleAxis, Quaternion
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix


def rotation_matrix_to_axang3(R):
    assert isinstance(R, RotationMatrix), repr(type(R))
    axang = AngleAxis(R.matrix())
    axang3 = axang.angle() * axang.axis()
    return axang3


def so3_vector_minus(R_WA, R_WD):
    """
    Provides an error δR_DA_W on the tangent bundle of SE(3) for an actual
    frame A and desired frame D, whose poses are expressed in common frame W.

    For a trivial first-order system with coordinates of R_WA, whose dynamics
    are defined via angular velocity:
        Ṙ_WA = ω_WA × R_WA
    We can use this error to exponentially drive A to D (when D is stationary):
        δR_DA_W = R_WA ⊖ R_WD  (what this function provides)
        ω_WA = -δR_DA_W
    Note that this is restricted to angular error magnitude within interval
    [0, π).

    The analogous error and feedback on a Euclidean quantity x ∈ Rⁿ with actual
    value xₐ and desired value xₜ:
        eₓ = xₐ - xₜ
        ẋₐ = -eₓ
    """
    # TODO(eric.cousineau): Add citations.
    # Wa is W from "actual perspective", Wd is W from "desired perspetive".
    # TODO(eric.cousineau): Better notation for this difference?
    R_WaWd = R_WA @ R_WD.inverse()
    dR_DA_W = rotation_matrix_to_axang3(R_WaWd)
    return dR_DA_W


def se3_vector_minus(X_WA, X_WD):
    """
    Extension of so3_vector_minus() from SO(3) to SE(3). Returns as a 6d
    vector that can be used for feedback via spatial velocity / acceleration:
        δX_DA_W = X_WA ⊖ X_WD
                = [δR_DA_W, p_DA_W]
    """
    # TODO(eric.cousineau): Hoist comments to Drake's
    # ComputePoseDiffInCommonFrame and rename that function.
    dX_DA_W = np.zeros(6)
    dX_DA_W[:3] = so3_vector_minus(X_WA.rotation(), X_WD.rotation())
    dX_DA_W[3:] = X_WA.translation() - X_WD.translation()
    return dX_DA_W


def xyz_rpy(xyz, rpy):
    """Shorthand to create an isometry from XYZ and RPY."""
    return RigidTransform(R=RotationMatrix(rpy=RollPitchYaw(rpy)), p=xyz)


def xyz_rpy_deg(xyz, rpy_deg):
    return xyz_rpy(xyz, np.deg2rad(rpy_deg))
