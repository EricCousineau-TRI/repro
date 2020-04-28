#!/usr/bin/env python3

"""
Simple demo using rviz.
"""

import time

import numpy as np

import rospy

from pydrake.common import FindResourceOrThrow
from pydrake.geometry import ConnectDrakeVisualizer
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource

from drake_ros1_hacks.rviz_visualizer import ConnectRvizVisualizer


def no_control(plant, builder, model):
    nu = plant.num_actuated_dofs(model)
    u0 = np.zeros(nu)
    constant = builder.AddSystem(ConstantVectorSource(u0))
    builder.Connect(
        constant.get_output_port(0),
        plant.get_actuation_input_port(model))


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--single_shot", action="store_true",
        help="")
    sdf_file = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/iiwa7/"
        "iiwa7_no_collision.sdf")
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.01)
    parser = Parser(plant)
    models = []
    model_count = 3
    for i in range(model_count):
        model_name = f"iiwa{i}"
        # TODO: Test multiple IIWAs.
        model = parser.AddModelFromFile(sdf_file, model_name)
        models.append(model)
        base_frame = plant.GetFrameByName("iiwa_link_0", model)
        # Translate along x-axis by 1m to separate.
        X_WB = RigidTransform([i, 0, 0])
        plant.WeldFrames(plant.world_frame(), base_frame, X_WB)
    plant.Finalize()
    for model in models:
        no_control(plant, builder, model)

    ConnectDrakeVisualizer(builder, scene_graph)
    ConnectRvizVisualizer(builder, scene_graph)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    simulator.set_target_realtime_rate(1.)

    # Wait for ROS publishers to wake up :(
    time.sleep(0.3)
    single_shot = True

    if single_shot:
        # To see what 'preview' scripts look like.
        # TODO(eric.cousineau): Make this work *robustly* with Rviz. Markers
        # still don't always show up :(
        simulator.Initialize()
        diagram.Publish(context)
    else:
        for _ in range(1000):
            simulator.AdvanceTo(context.get_time() + 0.1)


if __name__ == "__main__":
    rospy.init_node("demo", disable_signals=True)
    main()
