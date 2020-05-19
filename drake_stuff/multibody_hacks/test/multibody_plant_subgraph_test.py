"""
Tests MultibodyPlantSubgraph setup.

To visualize:

    bazel build //tools:drake_visualizer //common:multibody_plant_subgraph_test

    # Terminal 1
    ./run //tools:drake_visualizer

    # Terminal 2
    ./run //common:multibody_plant_subgraph_test --visualize

"""
import anzu.common.multibody_plant_subgraph as mut

import sys
import unittest

import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.geometry import ConnectDrakeVisualizer
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource

VISUALIZE = False

# TODO(eric.cousineau): Port this to pure Drake resource / API.
# TODO(eric.cousineau): Add a test that showed the sim contact failure (from
# clutter gen).
# TODO(eric.cousineau): Add test showing that a purely copied plant has the
# same position ordering? (fingers crossed)


def compare_frames(
        plant, context, sub_plant, sub_context, base_frame, test_frame,
        **kwargs):
    X_BT_sub = sub_plant.CalcRelativeTransform(
        sub_context,
        sub_plant.GetFrameByName(base_frame),
        sub_plant.GetFrameByName(test_frame))
    X_BT = plant.CalcRelativeTransform(
        context,
        plant.GetFrameByName(base_frame, **kwargs),
        plant.GetFrameByName(test_frame, **kwargs))
    np.testing.assert_allclose(
        X_BT_sub.GetAsMatrix4(),
        X_BT.GetAsMatrix4(), rtol=0., atol=1e-10)


class TestMultibodyPlantSubgraph(unittest.TestCase):

    def test_composition_array_without_scene_graph(self):
        """Tests subgraphs (post-finalize) without a scene graph."""
        iiwa_plant = MultibodyPlant(time_step=0.01)
        iiwa_file = FindResourceOrThrow(
            "drake/manipulation/models/iiwa_description/urdf/"
            "iiwa14_spheres_dense_elbow_collision.urdf")
        iiwa_model = Parser(iiwa_plant).AddModelFromFile(iiwa_file, "iiwa")
        iiwa_plant.Finalize()
        iiwa_context = iiwa_plant.CreateDefaultContext()
        base_frame_sub = iiwa_plant.GetFrameByName("base")

        # N.B. Because the model is not welded, we do not an additional policy
        # to "disconnect" it.
        iiwa_subgraph = mut.MultibodyPlantSubgraph.from_plant(iiwa_plant)

        # Make 10 copies of the IIWA in a line.
        plant = MultibodyPlant(time_step=0.01)
        models = []
        for i in range(10):
            sub_to_full = iiwa_subgraph.add_to(
                plant, model_instance_remap=f"iiwa_{i}")
            X_WB = RigidTransform(p=[i * 0.5, 0, 0])
            base_frame = sub_to_full.frames[base_frame_sub]
            plant.WeldFrames(plant.world_frame(), base_frame, X_WB)
            model = sub_to_full.model_instances[iiwa_model]
            models.append(model)

        plant.Finalize()
        context = plant.CreateDefaultContext()
        for i, model in enumerate(models):
            self.assertEqual(
                plant.GetModelInstanceName(model), f"iiwa_{i}")
            compare_frames(
                plant, context, iiwa_plant, iiwa_context,
                "base", "iiwa_link_7", model_instance=model)

    def test_composition_array_with_scene_graph(self):
        """Tests subgraphs (post-finalize) with a scene graph. This time, using
        sugar from parse_as_multibody_plant_subgraph."""
        # Create IIWA.
        iiwa_file = FindResourceOrThrow(
            "drake/manipulation/models/iiwa_description/urdf/"
            "iiwa14_spheres_dense_elbow_collision.urdf")
        iiwa_subgraph, iiwa_model = mut.parse_as_multibody_plant_subgraph(
            iiwa_file)
        # Make 10 copies of the IIWA in a line.
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=0.01)
        models = []
        for i in range(10):
            sub_to_full = iiwa_subgraph.add_to(
                plant, scene_graph, model_instance_remap=f"iiwa_{i}")
            X_WB = RigidTransform(p=[i * 0.5, 0, 0])
            model = sub_to_full.model_instances[iiwa_model]
            base_frame = plant.GetFrameByName("base", model)
            plant.WeldFrames(plant.world_frame(), base_frame, X_WB)
            models.append(model)

        plant.Finalize()
        if VISUALIZE:
            print("test_composition_array_with_scene_graph")
            ConnectDrakeVisualizer(builder, scene_graph)
        diagram = builder.Build()
        d_context = diagram.CreateDefaultContext()

        for i, model in enumerate(models):
            self.assertEqual(
                plant.GetModelInstanceName(model), f"iiwa_{i}")

        if VISUALIZE:
            print("  Visualize composite")
            Simulator(diagram, d_context.Clone()).Initialize()
            input("    Press enter...")

    def test_composition_gripper_workflow(self):
        """Tests subgraphs (pre-finalize) for composition, with a scene graph,
        welding bodies together across different subgraphs."""

        # Create IIWA.
        iiwa_builder = DiagramBuilder()
        iiwa_plant, iiwa_scene_graph = AddMultibodyPlantSceneGraph(
            iiwa_builder, time_step=0.)
        iiwa_file = FindResourceOrThrow(
            "drake/manipulation/models/iiwa_description/urdf/"
            "iiwa14_spheres_dense_elbow_collision.urdf")
        iiwa_model = Parser(iiwa_plant).AddModelFromFile(iiwa_file, "iiwa")
        iiwa_plant.WeldFrames(
            iiwa_plant.world_frame(),
            iiwa_plant.GetFrameByName("base"))

        iiwa_subgraph = mut.MultibodyPlantSubgraph.from_plant(
            iiwa_plant, iiwa_scene_graph)
        iiwa_subgraph.add_policy(mut.DisconnectFromWorldSubgraphPolicy())

        # Create WSG.
        wsg_builder = DiagramBuilder()
        wsg_plant, wsg_scene_graph = AddMultibodyPlantSceneGraph(
            wsg_builder, time_step=0.)
        wsg_file = FindResourceOrThrow(
            "drake/manipulation/models/wsg_50_description/sdf/"
            "schunk_wsg_50.sdf")
        wsg_model = Parser(wsg_plant).AddModelFromFile(
            wsg_file, "gripper_model")
        wsg_plant.WeldFrames(
            wsg_plant.world_frame(),
            wsg_plant.GetFrameByName("__model__"))

        wsg_subgraph = mut.MultibodyPlantSubgraph.from_plant(
            wsg_plant, wsg_scene_graph)
        wsg_subgraph.add_policy(mut.DisconnectFromWorldSubgraphPolicy())

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=1e-3)

        iiwa_to_plant = iiwa_subgraph.add_to(plant, scene_graph)
        iiwa = iiwa_to_plant.model_instances[iiwa_model]
        wsg_to_plant = wsg_subgraph.add_to(plant, scene_graph)
        wsg = iiwa_to_plant.model_instances[wsg_model]

        if VISUALIZE:
            print("test_composition")
            ConnectDrakeVisualizer(iiwa_builder, iiwa_scene_graph)
            ConnectDrakeVisualizer(wsg_builder, wsg_scene_graph)
            ConnectDrakeVisualizer(builder, scene_graph)

        rpy_deg = np.array([90., 0., 90])
        X_7G = RigidTransform(
            p=[0, 0, 0.114],
            rpy=RollPitchYaw(rpy_deg * np.pi / 180))
        frame_7 = plant.GetFrameByName("iiwa_link_7")
        frame_G = plant.GetFrameByName("body")
        plant.WeldFrames(frame_7, frame_G, X_7G)
        plant.Finalize()

        q_iiwa = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        q_wsg = [-0.03, 0.03]

        iiwa_plant.Finalize()
        iiwa_diagram = iiwa_builder.Build()
        iiwa_d_context = iiwa_diagram.CreateDefaultContext()
        iiwa_context = iiwa_plant.GetMyContextFromRoot(iiwa_d_context)
        iiwa_plant.SetPositions(iiwa_context, iiwa_model, q_iiwa)

        wsg_plant.Finalize()
        wsg_diagram = wsg_builder.Build()
        wsg_d_context = wsg_diagram.CreateDefaultContext()
        wsg_context = wsg_plant.GetMyContextFromRoot(wsg_d_context)
        wsg_plant.SetPositions(wsg_context, wsg_model, q_wsg)

        # Transfer state and briefly compare.
        diagram = builder.Build()
        d_context = diagram.CreateDefaultContext()
        context = plant.GetMyContextFromRoot(d_context)

        iiwa_to_plant.copy_state(iiwa_context, context)
        wsg_to_plant.copy_state(wsg_context, context)

        # Compare frames from sub-plants.
        compare_frames(
            plant, context, iiwa_plant, iiwa_context, "base", "iiwa_link_7")
        compare_frames(
            plant, context, wsg_plant, wsg_context, "body", "left_finger")

        # Visualize.
        if VISUALIZE:
            print("  Visualize IIWA")
            Simulator(iiwa_diagram, iiwa_d_context.Clone()).Initialize()
            input("    Press enter...")
            print("  Visualize WSG")
            Simulator(wsg_diagram, wsg_d_context.Clone()).Initialize()
            input("    Press enter...")
            print("  Visualize Composite")
            Simulator(diagram, d_context.Clone()).Initialize()
            input("    Press enter...")

    def test_decomposition_controller_like_workflow(self):
        """Tests subgraphs (post-finalize) for decomposition, with a
        scene-graph. Also shows a workflow of replacing joints / welding
        joints."""
        builder = DiagramBuilder()
        # N.B. I (Eric) am using ManipulationStation because it's currently
        # the simplest way to get a commplex scene in pydrake.
        station = ManipulationStation(time_step=0.001)
        station.SetupManipulationClassStation()
        station.Finalize()
        builder.AddSystem(station)
        plant = station.get_multibody_plant()
        scene_graph = station.get_scene_graph()
        iiwa = plant.GetModelInstanceByName("iiwa")
        wsg = plant.GetModelInstanceByName("gripper")

        if VISUALIZE:
            print("test_decomposition")
            ConnectDrakeVisualizer(
                builder, scene_graph, station.GetOutputPort("pose_bundle"))
        diagram = builder.Build()

        # Set the context with which things should be computed.
        d_context = diagram.CreateDefaultContext()
        context = plant.GetMyContextFromRoot(d_context)
        q_iiwa = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        q_wsg = [-0.03, 0.03]
        plant.SetPositions(context, iiwa, q_iiwa)
        plant.SetPositions(context, wsg, q_wsg)

        # Make a simple subgraph with just IIWA and gripper.
        subgraph = mut.MultibodyPlantSubgraph.from_plant(
            plant, scene_graph, model_instances=[iiwa, wsg])
        # Remove all articulation from the gripper.
        subgraph.add_policy(
            mut.FreezeJointSubgraphPolicy.from_plant(
                plant, context, model_instances=[wsg]))

        # Build and visualize.
        sub_builder = DiagramBuilder()
        sub_plant, sub_scene_graph = AddMultibodyPlantSceneGraph(
            sub_builder, time_step=0.)
        if VISUALIZE:
            ConnectDrakeVisualizer(sub_builder, sub_scene_graph)
        original_to_sub = subgraph.add_to(
            sub_plant, sub_scene_graph,
            model_instance_remap=mut.model_instance_remap_same_name,
        )
        sub_iiwa = original_to_sub.model_instances[iiwa]
        sub_plant.WeldFrames(
            sub_plant.world_frame(),
            sub_plant.GetFrameByName("iiwa_link_0", sub_iiwa))

        sub_plant.Finalize()
        self.assertEqual(
            sub_plant.num_positions(), plant.num_positions(iiwa))

        sub_diagram = sub_builder.Build()

        sub_d_context = sub_diagram.CreateDefaultContext()
        sub_context = sub_plant.GetMyContextFromRoot(sub_d_context)

        original_to_sub.copy_state(context, sub_context)
        compare_frames(
            plant, context, sub_plant, sub_context,
            "iiwa_link_0", "iiwa_link_7")
        compare_frames(
            plant, context, sub_plant, sub_context,
            "body", "left_finger")

        # Visualize.
        if VISUALIZE:
            print("  Visualize composite")
            Simulator(diagram, d_context.Clone()).Initialize()
            input("    Press enter...")
            print("  Visualize sub")
            Simulator(sub_diagram, sub_d_context.Clone()).Initialize()
            input("    Press enter...")

        # For grins, ensure that we can copy everything, including world weld.
        sub_plant_copy = MultibodyPlant(time_step=0.)
        mut.MultibodyPlantSubgraph.from_plant(sub_plant).add_to(sub_plant_copy)
        sub_plant_copy.Finalize()
        self.assertEqual(
            sub_plant_copy.num_positions(), sub_plant.num_positions())


if __name__ == "__main__":
    if "--visualize" in sys.argv:
        sys.argv.remove("--visualize")
        VISUALIZE = True
    unittest.main()
