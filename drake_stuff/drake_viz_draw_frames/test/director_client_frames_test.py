import numpy as np

from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix

from director_client_frames import (
    draw_frames,
    draw_frames_args,
    draw_frames_dict,
)


def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand to create a rigid transform from XYZ and RPY (degrees)."""
    rpy = np.deg2rad(rpy_deg)
    return RigidTransform(
        R=RotationMatrix(
            rpy=RollPitchYaw(rpy),
        ),
        p=xyz,
    )


def main():
    X_WA = xyz_rpy_deg([0.1, 0.2, 0.3], [0.0, 0.0, 0.0])
    X_WB = xyz_rpy_deg([0.5, 0.0, 0.0], [15.0, 30.0, 45.0])
    X_WC = xyz_rpy_deg([-0.5, 0.0, 0.5], [0.0, 0.0, 90.0])

    # Show each flavor. "suffix" could be anything you want.
    draw_frames(["X_WA"], [X_WA], suffix="_draw_frames")
    draw_frames_dict({"X_WB": X_WB}, suffix="_draw_frames_dict")
    draw_frames_args(X_WC=X_WC, suffix="_draw_frames_args")


assert __name__ == "__main__"
main()
