"""
Hacks relevant to:
https://stackoverflow.com/questions/61841013/how-to-dampen-multibodyplants-compliant-contact-model-in-a-simulation
"""
import numpy as np

from pydrake.multibody.inverse_kinematics import MinimumDistanceConstraint
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve


def set_frame_pose(plant, context, F, X_WF):
    """Sets the pose of a frame, assuming it is connecting to a free body."""
    B = F.body()
    X_FB = F.GetFixedPoseInBodyFrame().inverse()
    X_WB = X_WF @ X_FB
    plant.SetFreeBodyPose(context, B, X_WB)


def set_frame_heights(plant, context, frame_list, z):
    """Sets the height of a list of frames."""
    W = plant.world_frame()
    for F, zi in zip(frame_list, z):
        # Handle when F's scalar type differs from the plant.
        F = plant.get_frame(F.index())
        X_WF = plant.CalcRelativeTransform(context, W, F)
        p_WF = X_WF.translation().copy()
        p_WF[2] = zi
        X_WF.set_translation(p_WF)
        set_frame_pose(plant, context, F, X_WF)


def get_frame_heights(plant, context, frame_list):
    """Gets the height from a list of frames."""
    W = plant.world_frame()
    z = []
    for F in frame_list:
        # Handle when F's scalar type differs from the plant.
        F = plant.get_frame(F.index())
        X_WF = plant.CalcRelativeTransform(context, W, F)
        z_WF = X_WF.translation()[2]
        z.append(z_WF)
    return np.array(z)


def minimize_height(
        diagram_f, plant_f, d_context_f, frame_list):
    """Fragile and slow :("""
    context_f = plant_f.GetMyContextFromRoot(d_context_f)
    diagram_ad = diagram_f.ToAutoDiffXd()
    plant_ad = diagram_ad.GetSubsystemByName(plant_f.get_name())
    d_context_ad = diagram_ad.CreateDefaultContext()
    d_context_ad.SetTimeStateAndParametersFrom(d_context_f)
    context_ad = plant_ad.GetMyContextFromRoot(d_context_ad)

    def prepare_plant_and_context(z):
        if z.dtype == float:
            plant, context = plant_f, context_f
        else:
            plant, context = plant_ad, context_ad
        set_frame_heights(plant, context, frame_list, z)
        return plant, context

    prog = MathematicalProgram()
    num_z = len(frame_list)
    z_vars = prog.NewContinuousVariables(num_z, "z")
    q0 = plant_f.GetPositions(context_f)
    z0 = get_frame_heights(plant_f, context_f, frame_list)
    cost = prog.AddCost(np.sum(z_vars))
    prog.AddBoundingBoxConstraint([0.01]*num_z, [5.]*num_z, z_vars)
    # # N.B. Cannot use AutoDiffXd due to cylinders.
    distance = MinimumDistanceConstraint(
        plant=plant_f, plant_context=context_f,
        minimum_distance=0.05)

    def distance_with_z(z):
        plant, context = prepare_plant_and_context(z)
        q = plant.GetPositions(context)
        return distance.Eval(q)

    prog.AddConstraint(
        distance_with_z,
        lb=distance.lower_bound(),
        ub=distance.upper_bound(),
        vars=z_vars)

    result = Solve(prog, initial_guess=z0)
    assert result.is_success()
    z = result.GetSolution(z_vars)
    set_frame_heights(plant_f, context_f, frame_list, z)
    q = plant_f.GetPositions(context_f)
    return q
