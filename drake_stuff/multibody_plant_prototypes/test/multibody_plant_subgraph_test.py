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
from pydrake.geometry import ConnectDrakeVisualizer, HalfSpace, Role
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.math import SpatialVelocity
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    ConnectContactResultsToDrakeVisualizer,
    CoulombFriction,
    MultibodyPlant,
)
from pydrake.multibody.tree import ModelInstanceIndex, RevoluteJoint
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, EventStatus
from pydrake.systems.primitives import ConstantVectorSource

VISUALIZE = False

# TODO(eric.cousineau): Port this to pure Drake resource / API.
# TODO(eric.cousineau): Add a test that showed the sim contact failure (from
# clutter gen).
# TODO(eric.cousineau): Add test showing that a purely copied plant has the
# same position ordering? (fingers crossed)


def no_control(builder, plant, model):
    nu = plant.num_actuated_dofs(model)
    constant = builder.AddSystem(ConstantVectorSource(np.zeros(nu)))
    builder.Connect(
        constant.get_output_port(0),
        plant.get_actuation_input_port(model))


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
        self.assertIsInstance(iiwa_subgraph, mut.MultibodyPlantSubgraph)

        # Make 10 copies of the IIWA in a line.
        plant = MultibodyPlant(time_step=0.01)
        models = []
        for i in range(10):
            sub_to_full = iiwa_subgraph.add_to(
                plant, model_instance_remap=f"iiwa_{i}")
            self.assertIsInstance(
                sub_to_full, mut.MultibodyPlantAssociations)
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
        self.assertIsInstance(iiwa_subgraph, mut.MultibodyPlantSubgraph)
        self.assertIsInstance(iiwa_model, ModelInstanceIndex)
        # Make 10 copies of the IIWA in a line.
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=0.01)
        models = []
        for i in range(10):
            sub_to_full = iiwa_subgraph.add_to(
                plant, scene_graph, model_instance_remap=f"iiwa_{i}")
            self.assertIsInstance(
                sub_to_full, mut.MultibodyPlantAssociations)
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
        self.assertIsInstance(iiwa_subgraph, mut.MultibodyPlantSubgraph)
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
        self.assertIsInstance(wsg_subgraph, mut.MultibodyPlantSubgraph)
        wsg_subgraph.add_policy(mut.DisconnectFromWorldSubgraphPolicy())

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=1e-3)

        iiwa_to_plant = iiwa_subgraph.add_to(plant, scene_graph)
        self.assertIsInstance(iiwa_to_plant, mut.MultibodyPlantAssociations)
        iiwa = iiwa_to_plant.model_instances[iiwa_model]
        wsg_to_plant = wsg_subgraph.add_to(plant, scene_graph)
        self.assertIsInstance(wsg_to_plant, mut.MultibodyPlantAssociations)
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
        if False: #VISUALIZE:
            print("  Visualize IIWA")
            Simulator(iiwa_diagram, iiwa_d_context.Clone()).Initialize()
            input("    Press enter...")
            print("  Visualize WSG")
            Simulator(wsg_diagram, wsg_d_context.Clone()).Initialize()
            input("    Press enter...")
            print("  Visualize Composite")
            Simulator(diagram, d_context.Clone()).Initialize()
            input("    Press enter...")

        self.do_exploding_iiwa_sim(plant, scene_graph, context)

    def do_exploding_iiwa_sim(self, plant_src, scene_graph_src, context_src):
        role = Role.kIllustration
        # role = Role.kProximity
        # Show a simulation which "changes state" by being rebuilt.
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=1e-3)
        subgraph_src = mut.MultibodyPlantSubgraph.from_plant(
            plant_src, scene_graph_src)
        to_plant = subgraph_src.add_to(plant, scene_graph)
        # Add ground plane.
        X_FH = HalfSpace.MakePose([0, 0, 1], [0, 0, 0])
        plant.RegisterCollisionGeometry(
            plant.world_body(), X_FH, HalfSpace(), "ground_plane_collision",
            CoulombFriction(0.8, 0.3))
        plant.Finalize()
        # Loosey-goosey.
        for model in mut.get_model_instances(plant):
            no_control(builder, plant, model)
        if VISUALIZE:
            ConnectDrakeVisualizer(builder, scene_graph, role=role)
            ConnectContactResultsToDrakeVisualizer(builder, plant)
        diagram = builder.Build()
        # Set up context.
        d_context = diagram.CreateDefaultContext()
        context = plant.GetMyContextFromRoot(d_context)
        to_plant.copy_state(context_src, context)
        # - Hoist IIWA up in the air.
        plant.SetFreeBodyPose(
            context, plant.GetBodyByName("base"), RigidTransform([0, 0, 1.]))
        # - Set joint velocities to "spin" it in the air.
        for joint in mut.get_joints(plant):
            if isinstance(joint, RevoluteJoint):
                mut.set_joint_positions(plant, context, joint, 0.7)
                mut.set_joint_velocities(plant, context, joint, -5.)

        def monitor(d_context):
            context = plant.GetMyContextFromRoot(d_context)
            # If any contact?
            query_object = plant.get_geometry_query_input_port().Eval(context)
            if query_object.HasCollisions():
                return EventStatus.ReachedTermination(plant, "Collision")
            else:
                return EventStatus.DidNothing()

        # Forward simulate.
        simulator = Simulator(diagram, d_context)
        simulator.Initialize()
        simulator.set_monitor(monitor)
        simulator.set_target_realtime_rate(1.)
        simulator.AdvanceTo(2.)
        # Try to push a bit further?
        simulator.clear_monitor()
        simulator.AdvanceTo(d_context.get_time() + 0.05)
        diagram.Publish(d_context)

        # Recreate simulator.
        builder_new = DiagramBuilder()
        plant_new, scene_graph_new = AddMultibodyPlantSceneGraph(
            builder_new, time_step=plant.time_step())

        subgraph = mut.MultibodyPlantSubgraph.from_plant(plant, scene_graph)
        # Remove all joints; make them floating bodies.
        for joint in mut.get_joints(plant):
            subgraph.remove_joint(joint)
        # Remove massless / low inertia bodies?
        for body in mut.get_bodies(plant):
            if body is plant.world_body():
                continue
            if body.default_mass() == 0.:
                subgraph.remove_body(body)

        to_new = subgraph.add_to(plant_new, scene_graph_new)
        plant_new.Finalize()

        if VISUALIZE:
            ConnectDrakeVisualizer(builder_new, scene_graph_new, role=role)
            ConnectContactResultsToDrakeVisualizer(builder_new, plant_new)
        diagram_new = builder_new.Build()

        # Remap state.
        d_context_new = diagram_new.CreateDefaultContext()
        d_context_new.SetTime(d_context.get_time())
        context_new = plant_new.GetMyContextFromRoot(d_context_new)
        to_new.copy_state(context, context_new)

        # Simulate.
        simulator_new = Simulator(diagram_new, d_context_new)
        simulator_new.Initialize()
        diagram_new.Publish(d_context_new)
        simulator_new.set_target_realtime_rate(1.)
        simulator_new.AdvanceTo(context_new.get_time() + 2)

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
            print("test_decomposition_controller_like_workflow")
            ConnectDrakeVisualizer(
                builder, scene_graph, station.GetOutputPort("pose_bundle"))
        diagram = builder.Build()

        # Set the context with which things should be computed.
        d_context = diagram.CreateDefaultContext()
        context = plant.GetMyContextFromRoot(d_context)
        q_iiwa = [0.3, 0.7, 0.3, 0.6, 0.5, 0.6, 0.7]
        q_wsg = [-0.03, 0.03]
        plant.SetPositions(context, iiwa, q_iiwa)
        plant.SetPositions(context, wsg, q_wsg)

        # Build and visualize.
        control_builder = DiagramBuilder()
        control_plant, control_scene_graph = AddMultibodyPlantSceneGraph(
            control_builder, time_step=0.)
        if VISUALIZE:
            ConnectDrakeVisualizer(control_builder, control_scene_graph)

        # N.B. This has the same scene, but with all joints outside of the
        # IIWA frozen.
        to_control = mut.add_plant_with_articulated_subset_to(
            plant_src=plant, scene_graph_src=scene_graph,
            articulated_models_src=[iiwa], context_src=context,
            plant_dest=control_plant, scene_graph_dest=control_scene_graph)
        self.assertIsInstance(to_control, mut.MultibodyPlantAssociations)
        control_iiwa = to_control.model_instances[iiwa]
        control_plant.Finalize()

        self.assertEqual(
            control_plant.num_positions(), plant.num_positions(iiwa))

        control_diagram = control_builder.Build()

        control_d_context = control_diagram.CreateDefaultContext()
        control_context = control_plant.GetMyContextFromRoot(control_d_context)

        to_control.copy_state(context, control_context)
        compare_frames(
            plant, context, control_plant, control_context,
            "iiwa_link_0", "iiwa_link_7")
        compare_frames(
            plant, context, control_plant, control_context,
            "body", "left_finger")

        # Visualize.
        if VISUALIZE:
            print("  Visualize original plant")
            Simulator(diagram, d_context.Clone()).Initialize()
            input("    Press enter...")
            print("  Visualize control plant")
            Simulator(control_diagram, control_d_context.Clone()).Initialize()
            input("    Press enter...")

        # For grins, ensure that we can copy everything, including world weld.
        control_plant_copy = MultibodyPlant(time_step=0.)
        mut.MultibodyPlantSubgraph.from_plant(
            control_plant).add_to(control_plant_copy)
        control_plant_copy.Finalize()
        self.assertEqual(
            control_plant_copy.num_positions(), control_plant.num_positions())


if __name__ == "__main__":
    if "--visualize" in sys.argv:
        sys.argv.remove("--visualize")
        VISUALIZE = True
    unittest.main()
