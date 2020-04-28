#!/usr/bin/env python3

"""
Simple demo using rviz.

Known issues:

* Markers do not always show up in Rviz on the first time. I (Eric) have to
relaunch this multiple times.
* Colors from SceneGraph cannot be retrieved in Python (due to binding of
Value[Vector*]`.
* Changes in geometry are explicitly disabled here.
"""

import argparse
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
        "--simulation_sec", type=float, default=np.inf)
    args_parser.add_argument(
        "--sim_dt", type=float, default=0.1)
    args_parser.add_argument(
        "--single_shot", action="store_true",
        help="Test workflow of visulaization through Simulator.Initialize")
    args_parser.add_argument(
        "--realtime_rate", type=float, default=1.)
    args_parser.add_argument(
        "--num_models", type=int, default=3)
    args = args_parser.parse_args()

    sdf_file = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/iiwa7/"
        "iiwa7_no_collision.sdf")
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.01)
    parser = Parser(plant)
    models = []
    for i in range(args.num_models):
        model_name = f"iiwa{i}"
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
    simulator.set_target_realtime_rate(args.realtime_rate)

    # Wait for ROS publishers to wake up :(
    time.sleep(0.3)

    if args.single_shot:
        # To see what 'preview' scripts look like.
        # TODO(eric.cousineau): Make this work *robustly* with Rviz. Markers
        # still don't always show up :(
        simulator.Initialize()
        diagram.Publish(context)
    else:
        while context.get_time() < args.simulation_sec:
            # Use increments to permit Ctrl+C to be caught.
            simulator.AdvanceTo(context.get_time() + args.sim_dt)


if __name__ == "__main__":
    rospy.init_node("demo", disable_signals=True)
    main()
