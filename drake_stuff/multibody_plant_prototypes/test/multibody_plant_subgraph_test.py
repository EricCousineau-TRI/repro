"""
Tests MultibodyPlantSubgraph setup.

To visualize:

    bazel build //tools:drake_visualizer //common:multibody_plant_subgraph_test

    # Terminal 1
    ./run //tools:drake_visualizer

    # Terminal 2
    ./run //common:multibody_plant_subgraph_test --visualize

"""
from contextlib import contextmanager
import copy
import sys
import unittest

import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.geometry import (
    DrakeVisualizer,
    DrakeVisualizerParams,
    HalfSpace,
    Role,
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    ConnectContactResultsToDrakeVisualizer,
    CoulombFriction,
    MultibodyPlant,
)
from pydrake.multibody.tree import (
    ModelInstanceIndex,
    RevoluteJoint,
    world_model_instance,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.framework import DiagramBuilder, EventStatus

import multibody_plant_prototypes.multibody_extras as me
import multibody_plant_prototypes.multibody_plant_subgraph as mut
import multibody_plant_prototypes.test.multibody_plant_subgraph_test_helpers as util
from multibody_plant_prototypes.function_system import (
    ContextTimeArg,
    FunctionSystem,
    VectorArg,
)

VISUALIZE = False

# TODO(eric.cousineau): Add a test that showed the sim contact failure (from
# clutter gen).


class MultibodyPlantElementsCount(mut._MultibodyPlantElementsBase):
    def __init__(self):
        super().__init__(int)

    """Takes a count of elements."""
    @staticmethod
    def of(x):
        new = MultibodyPlantElementsCount()
        new.model_instances = len(x.model_instances)
        new.bodies = len(x.bodies)
        new.frames = len(x.frames)
        new.joints = len(x.joints)
        new.joint_actuators = len(x.joint_actuators)
        new.geometry_ids = len(x.geometry_ids)
        return new


class TestApi(unittest.TestCase):
    """Tests lower-level API."""

    def setUp(self):
        self.maxDiff = None

    def assert_count_equal(self, a, b):
        self.assertMultiLineEqual(str(a), str(b))

    def make_arbitrary_multibody_stuff(self, *, finalize=True, **kwargs):
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=0.01)
        util.add_arbitrary_multibody_stuff(plant, **kwargs)
        if finalize:
            plant.Finalize()
            diagram = builder.Build()
            return plant, scene_graph, diagram
        else:
            return plant, scene_graph, builder

    def test_meta_arbitrary_multibody_stuff(self):
        """Ensures that the test model is logical and can be compared /
        falsified."""
        plant_a, scene_graph_a, _ = self.make_arbitrary_multibody_stuff(
            finalize=False)
        # Ensure that we span all relevant joint classes.
        joint_cls_set = {type(x) for x in me.get_joints(plant_a)}
        self.assertEqual(joint_cls_set, set(util.JOINT_CLS_LIST))

        # Ensure that we can copy via a subgraph pre-Finalize.
        builder_b = DiagramBuilder()
        plant_b, scene_graph_b = AddMultibodyPlantSceneGraph(
            builder_b, plant_a.time_step())
        subgraph_a = mut.MultibodyPlantSubgraph(mut.get_elements_from_plant(
            plant_a, scene_graph_a))
        subgraph_a.add_to(plant_b, scene_graph_b)
        # Check equivalence.
        util.assert_plant_equals(
            plant_a, scene_graph_a, plant_b, scene_graph_b)
        # Ensure that this is "physically" valid.
        plant_a.Finalize()

        # Checking for determinism, making a slight change to trigger an error.
        for slight_difference in [False, True]:
            plant_b, scene_graph_b, _ = self.make_arbitrary_multibody_stuff(
                slight_difference=slight_difference)

            if not slight_difference:
                util.assert_plant_equals(
                    plant_a, scene_graph_a, plant_b, scene_graph_b)
            else:
                with self.assertRaises(AssertionError):
                    util.assert_plant_equals(
                        plant_a, scene_graph_a, plant_b, scene_graph_b)

    def test_multibody_plant_elements(self):
        plant, scene_graph, _ = self.make_arbitrary_multibody_stuff(
            num_bodies=1)

        other_plant, _, _ = self.make_arbitrary_multibody_stuff(num_bodies=1)

        # Test nominal usage.
        elem = mut.get_elements_from_plant(plant, scene_graph)
        actual_count = MultibodyPlantElementsCount.of(elem)
        expected_count = MultibodyPlantElementsCount()
        expected_count.__dict__.update(
            model_instances=3,
            bodies=2,
            frames=6,
            joints=1,
            joint_actuators=1,
            geometry_ids=3,
        )
        self.assert_count_equal(actual_count, expected_count)

        # Test copying.
        elem_copy = copy.copy(elem)
        self.assertIsNot(elem, elem_copy)
        self.assertEqual(elem, elem_copy)

        # Test subgraph invariant.
        mut.check_subgraph_invariants(elem)

        @contextmanager
        def check_subgraph_negative():
            elem_copy = copy.copy(elem)
            yield elem_copy
            with self.assertRaises(AssertionError):
                mut.check_subgraph_invariants(elem_copy)

        # Check negative cases:
        # - subgraph model instance in plant model instances
        with check_subgraph_negative() as elem_copy:
            elem_copy.model_instances.add(ModelInstanceIndex(100))
        # - subgraph bodies in subgraph model instances.
        with check_subgraph_negative() as elem_copy:
            elem_copy.model_instances.remove(world_model_instance())
        # - subgraph element must be part of the subgraph plant.
        with check_subgraph_negative() as elem_copy:
            elem_copy.bodies.add(other_plant.world_body())
        # - subgraph frames must be attached to subgraph bodies
        # - subgraph joints only connected to subgraph bodies
        # - subgrpah geometries must be attached to subgraph bodies
        with check_subgraph_negative() as elem_copy:
            elem_copy.bodies.remove(plant.world_body())
        # - subgraph joint actuators must solely act on subgraph joints
        with check_subgraph_negative() as elem_copy:
            joint, = elem.joints
            elem_copy.joints.remove(joint)

        # Test usage without SceneGraph.
        elem_copy_no_scene_graph = copy.copy(elem)
        elem_copy_no_scene_graph.scene_graph = None
        elem_copy_no_scene_graph.geometry_ids = set()
        self.assertEqual(
            mut.get_elements_from_plant(plant), elem_copy_no_scene_graph)

        # Tests that we can't add elements with any intersection.
        with self.assertRaises(AssertionError):
            elem += elem_copy

        # Tests that we can add "disjoint" stuff.
        elem_copy = copy.copy(elem)
        last_body = list(elem.bodies)[-1]
        elem_copy.bodies.remove(last_body)
        self.assertNotEqual(elem, elem_copy)
        elem_world_body_only = mut.MultibodyPlantElements(plant, scene_graph)
        elem_world_body_only.bodies.add(last_body)
        elem_copy += elem_world_body_only
        self.assertEqual(elem, elem_copy)

    def create_manual_map(
            self, plant_a, scene_graph_a, plant_b, scene_graph_b):
        # Manually construct map.
        a_to_b = mut.MultibodyPlantElementsMap(
            plant_a, plant_b,
            scene_graph_a, scene_graph_b)
        empty_a = mut.MultibodyPlantElements(plant_a, scene_graph_a)
        self.assertEqual(empty_a, a_to_b.make_empty_elements_src())
        empty_b = mut.MultibodyPlantElements(plant_b, scene_graph_b)
        self.assertEqual(empty_b, a_to_b.make_empty_elements_dest())
        b_to_a = mut.MultibodyPlantElementsMap(
            plant_b, plant_a,
            scene_graph_b, scene_graph_a)
        elem_a = mut.get_elements_from_plant(plant_a, scene_graph_a)
        elem_b = mut.get_elements_from_plant(plant_b, scene_graph_b)
        for field in mut._FIELDS:
            for a, b in zip(getattr(elem_a, field), getattr(elem_b, field)):
                getattr(a_to_b, field)[a] = b
                getattr(b_to_a, field)[b] = a
        self.assert_count_equal(
            MultibodyPlantElementsCount.of(a_to_b),
            MultibodyPlantElementsCount.of(elem_a))
        self.assertNotEqual(a_to_b, b_to_a)
        self.assertEqual(a_to_b, b_to_a.inverse())
        self.assertEqual(b_to_a, a_to_b.inverse())
        return a_to_b

    def test_multibody_plant_elements_map(self):
        """Tests basic container API for MultibodyPlantElementsMap.
        All `copy_*` functionality is tested (implicitly) in
        `test_subgraph_add_to_copying`."""
        plant_a, scene_graph_a, _ = self.make_arbitrary_multibody_stuff(
            num_bodies=1)
        plant_b, scene_graph_b, _ = self.make_arbitrary_multibody_stuff(
            num_bodies=1)
        self.assertIsNot(plant_a, plant_b)
        self.assertIsNot(scene_graph_a, scene_graph_b)
        util.assert_plant_equals(
            plant_a, scene_graph_a, plant_b, scene_graph_b)
        self.create_manual_map(
            plant_a, scene_graph_a, plant_b, scene_graph_b)

    def test_subgraph_construction_and_mutation(self):
        """Ensures that we always have a "subgraph" on construction and with
        mutation."""
        plant, scene_graph, _ = self.make_arbitrary_multibody_stuff(
            num_bodies=10)
        subgraph = mut.MultibodyPlantSubgraph(
            mut.get_elements_from_plant(plant, scene_graph))
        elem = subgraph.elements_src
        mut.check_subgraph_invariants(elem)

        # remove_body
        tmp = copy.copy(subgraph)
        body = plant.world_body()
        elem_removed = tmp.remove_body(body)
        mut.check_subgraph_invariants(tmp.elements_src)
        self.assertIn(body, elem_removed.bodies)
        self.assertNotIn(body, tmp.elements_src.bodies)
        expected_count = MultibodyPlantElementsCount()
        expected_count.__dict__.update(
            model_instances=0,
            bodies=1,
            frames=10,
            joints=4,
            joint_actuators=1,
            geometry_ids=1,
        )
        self.assert_count_equal(
            MultibodyPlantElementsCount.of(elem_removed),
            expected_count)

        # remove_frame
        tmp = copy.copy(subgraph)
        frame = plant.world_frame()
        elem_removed = tmp.remove_frame(frame)
        mut.check_subgraph_invariants(tmp.elements_src)
        self.assertIn(frame, elem_removed.frames)
        self.assertNotIn(frame, tmp.elements_src.frames)
        expected_count = MultibodyPlantElementsCount()
        expected_count.__dict__.update(
            model_instances=0,
            bodies=0,
            frames=1,
            joints=0,
            joint_actuators=0,
            geometry_ids=0,
        )
        self.assert_count_equal(
            MultibodyPlantElementsCount.of(elem_removed),
            expected_count)

        # remove_joint
        tmp = copy.copy(subgraph)
        joint = list(elem.joints)[0]
        elem_removed = tmp.remove_joint(joint)
        mut.check_subgraph_invariants(tmp.elements_src)
        self.assertIn(joint, elem_removed.joints)
        self.assertNotIn(joint, tmp.elements_src.joints)
        expected_count = MultibodyPlantElementsCount()
        expected_count.__dict__.update(
            model_instances=0,
            bodies=0,
            frames=0,
            joints=1,
            joint_actuators=0,
            geometry_ids=0,
        )
        self.assert_count_equal(
            MultibodyPlantElementsCount.of(elem_removed),
            expected_count)

        # TODO(eric.cousineau): Test remove_joint_actuator, remove_geometry_id

    def test_subgraph_add_to_copying(self):
        """Ensures that index ordering is generally the same when copying a
        plant using a MultibodyPlantSubgraph.add_to."""
        # TODO(eric.cousineau): Increas number of bodies for viz, once
        # `create_manual_map` can acommodate it.
        plant_a, scene_graph_a, _ = self.make_arbitrary_multibody_stuff(
            num_bodies=1)

        # Check for general ordering with full subgraph "cloning".
        builder_b = DiagramBuilder()
        plant_b, scene_graph_b = AddMultibodyPlantSceneGraph(
            builder_b, plant_a.time_step())
        subgraph_a = mut.MultibodyPlantSubgraph(mut.get_elements_from_plant(
            plant_a, scene_graph_a))
        a_to_b = subgraph_a.add_to(plant_b, scene_graph_b)
        plant_b.Finalize()
        util.assert_plant_equals(
            plant_a, scene_graph_a, plant_b, scene_graph_b)

        a_to_b_expected = self.create_manual_map(
            plant_a, scene_graph_a, plant_b, scene_graph_b)
        self.assertEqual(a_to_b, a_to_b_expected)

        if VISUALIZE:
            for model in me.get_model_instances(plant_b):
                util.build_with_no_control(builder_b, plant_b, model)
            print("test_subgraph_add_to_copying")
            DrakeVisualizer.AddToBuilder(builder_b, scene_graph_b)
            diagram = builder_b.Build()
            simulator = Simulator(diagram)
            simulator.set_target_realtime_rate(1.)
            simulator.Initialize()
            diagram.Publish(simulator.get_context())
            simulator.AdvanceTo(1.)


class TestWorkflows(unittest.TestCase):

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
        iiwa_subgraph = mut.MultibodyPlantSubgraph(
            mut.get_elements_from_plant(iiwa_plant))
        self.assertIsInstance(iiwa_subgraph, mut.MultibodyPlantSubgraph)

        # Make 10 copies of the IIWA in a line.
        plant = MultibodyPlant(time_step=0.01)
        models = []
        for i in range(10):
            sub_to_full = iiwa_subgraph.add_to(
                plant, model_instance_remap=f"iiwa_{i}")
            self.assertIsInstance(
                sub_to_full, mut.MultibodyPlantElementsMap)
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
            util.compare_frame_poses(
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
                sub_to_full, mut.MultibodyPlantElementsMap)
            X_WB = RigidTransform(p=[i * 0.5, 0, 0])
            model = sub_to_full.model_instances[iiwa_model]
            base_frame = plant.GetFrameByName("base", model)
            plant.WeldFrames(plant.world_frame(), base_frame, X_WB)
            models.append(model)

        plant.Finalize()
        if VISUALIZE:
            print("test_composition_array_with_scene_graph")
            DrakeVisualizer.AddToBuilder(builder, scene_graph)
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

        # N.B. The frame-welding is done so that we can easily set the
        # positions of the IIWA / WSG without having to worry about / work
        # around the floating body coordinates.

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
        iiwa_plant.Finalize()
        if VISUALIZE:
            print("test_composition")
            DrakeVisualizer.AddToBuilder(iiwa_builder, iiwa_scene_graph)
        iiwa_diagram = iiwa_builder.Build()

        iiwa_subgraph = mut.MultibodyPlantSubgraph(
            mut.get_elements_from_plant(iiwa_plant, iiwa_scene_graph))
        self.assertIsInstance(iiwa_subgraph, mut.MultibodyPlantSubgraph)
        iiwa_subgraph.remove_body(iiwa_plant.world_body())

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
        wsg_plant.Finalize()
        if VISUALIZE:
            DrakeVisualizer.AddToBuilder(wsg_builder, wsg_scene_graph)
        wsg_diagram = wsg_builder.Build()

        wsg_subgraph = mut.MultibodyPlantSubgraph(
            mut.get_elements_from_plant(wsg_plant, wsg_scene_graph))
        self.assertIsInstance(wsg_subgraph, mut.MultibodyPlantSubgraph)
        wsg_subgraph.remove_body(wsg_plant.world_body())

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=1e-3)

        iiwa_to_plant = iiwa_subgraph.add_to(plant, scene_graph)
        self.assertIsInstance(iiwa_to_plant, mut.MultibodyPlantElementsMap)
        wsg_to_plant = wsg_subgraph.add_to(plant, scene_graph)
        self.assertIsInstance(wsg_to_plant, mut.MultibodyPlantElementsMap)

        if VISUALIZE:
            DrakeVisualizer.AddToBuilder(builder, scene_graph)

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

        iiwa_d_context = iiwa_diagram.CreateDefaultContext()
        iiwa_context = iiwa_plant.GetMyContextFromRoot(iiwa_d_context)
        iiwa_plant.SetPositions(iiwa_context, iiwa_model, q_iiwa)

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
        util.compare_frame_poses(
            plant, context, iiwa_plant, iiwa_context, "base", "iiwa_link_7")
        util.compare_frame_poses(
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

    def test_exploding_iiwa_sim(self):
        """
        Shows a simulation of a falling + exploding IIWA which "changes
        topology" by being rebuilt without joints.
        """
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=1e-3)
        iiwa_file = FindResourceOrThrow(
            "drake/manipulation/models/iiwa_description/urdf/"
            "iiwa14_spheres_dense_elbow_collision.urdf")
        Parser(plant).AddModelFromFile(iiwa_file, "iiwa")
        # Add ground plane.
        X_FH = HalfSpace.MakePose([0, 0, 1], [0, 0, 0])
        plant.RegisterCollisionGeometry(
            plant.world_body(), X_FH, HalfSpace(), "ground_plane_collision",
            CoulombFriction(0.8, 0.3))
        plant.Finalize()
        # Loosey-goosey - no control.
        for model in me.get_model_instances(plant):
            util.build_with_no_control(builder, plant, model)
        if VISUALIZE:
            print("test_exploding_iiwa_sim")
            role = Role.kIllustration
            DrakeVisualizer.AddToBuilder(
                builder, scene_graph, params=DrakeVisualizerParams(role=role))
            ConnectContactResultsToDrakeVisualizer(builder, plant)
        diagram = builder.Build()
        # Set up context.
        d_context = diagram.CreateDefaultContext()
        context = plant.GetMyContextFromRoot(d_context)
        # - Hoist IIWA up in the air.
        plant.SetFreeBodyPose(
            context, plant.GetBodyByName("base"), RigidTransform([0, 0, 1.]))
        # - Set joint velocities to "spin" it in the air.
        for joint in me.get_joints(plant):
            if isinstance(joint, RevoluteJoint):
                me.set_joint_positions(plant, context, joint, 0.7)
                me.set_joint_velocities(plant, context, joint, -5.)

        def monitor(d_context):
            # Stop the simulation once there's any contact.
            context = plant.GetMyContextFromRoot(d_context)
            query_object = plant.get_geometry_query_input_port().Eval(context)
            if query_object.HasCollisions():
                return EventStatus.ReachedTermination(plant, "Contact")
            else:
                return EventStatus.DidNothing()

        # Forward simulate.
        simulator = Simulator(diagram, d_context)
        simulator.Initialize()
        simulator.set_monitor(monitor)
        if VISUALIZE:
            simulator.set_target_realtime_rate(1.)
        simulator.AdvanceTo(2.)
        # Try to push a bit further.
        simulator.clear_monitor()
        simulator.AdvanceTo(d_context.get_time() + 0.05)
        diagram.Publish(d_context)

        # Recreate simulator.
        builder_new = DiagramBuilder()
        plant_new, scene_graph_new = AddMultibodyPlantSceneGraph(
            builder_new, time_step=plant.time_step())
        subgraph = mut.MultibodyPlantSubgraph(
            mut.get_elements_from_plant(plant, scene_graph))
        # Remove all joints; make them floating bodies.
        for joint in me.get_joints(plant):
            subgraph.remove_joint(joint)
        # Remove massless bodies.
        # For more info, see: https://stackoverflow.com/a/62035705/7829525
        for body in me.get_bodies(plant):
            if body is plant.world_body():
                continue
            if body.default_mass() == 0.:
                subgraph.remove_body(body)
        # Finalize.
        to_new = subgraph.add_to(plant_new, scene_graph_new)
        plant_new.Finalize()
        if VISUALIZE:
            DrakeVisualizer.AddToBuilder(
                builder_new, scene_graph_new,
                params=DrakeVisualizerParams(role=role))
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
        if VISUALIZE:
            simulator_new.set_target_realtime_rate(1.)
        simulator_new.AdvanceTo(context_new.get_time() + 2)
        if VISUALIZE:
            input("    Press enter...")

    def test_decomposition_controller_like_workflow(self):
        """Tests subgraphs (post-finalize) for decomposition, with a
        scene-graph. Also shows a workflow of replacing joints / welding
        joints."""
        builder = DiagramBuilder()
        # N.B. I (Eric) am using ManipulationStation because it's currently
        # the simplest way to get a complex scene in pydrake.
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
            DrakeVisualizer.AddToBuilder(
                builder, station.GetOutputPort("query_object"))
        diagram = builder.Build()

        # Set the context with which things should be computed.
        d_context = diagram.CreateDefaultContext()
        context = plant.GetMyContextFromRoot(d_context)
        q_iiwa = [0.3, 0.7, 0.3, 0.6, 0.5, 0.6, 0.7]
        ndof = 7
        q_wsg = [-0.03, 0.03]
        plant.SetPositions(context, iiwa, q_iiwa)
        plant.SetPositions(context, wsg, q_wsg)

        # Add copy of only the IIWA to a control diagram.
        control_builder = DiagramBuilder()
        control_plant = control_builder.AddSystem(MultibodyPlant(time_step=0))
        # N.B. This has the same scene, but with all joints outside of the
        # IIWA frozen
        # TODO(eric.cousineau): Re-investigate weird "build_with_no_control"
        # behavior (with scene graph) with default conditions and time_step=0
        # - see Anzu commit 2cf08cfc3.
        to_control = mut.add_plant_with_articulated_subset_to(
            plant_src=plant, scene_graph_src=scene_graph,
            articulated_models_src=[iiwa], context_src=context,
            plant_dest=control_plant)
        self.assertIsInstance(to_control, mut.MultibodyPlantElementsMap)
        control_iiwa = to_control.model_instances[iiwa]

        control_plant.Finalize()
        self.assertEqual(
            control_plant.num_positions(), plant.num_positions(iiwa))

        kp = np.array([2000., 1500, 1500, 1500, 1500, 500, 500])
        kd = np.sqrt(2 * kp)
        ki = np.zeros(7)
        controller = control_builder.AddSystem(InverseDynamicsController(
            robot=control_plant, kp=kp, ki=ki, kd=kd,
            has_reference_acceleration=False))
        # N.B. Rather than use model instances for direct correspence, we could
        # use the mappings themselves within a custom system.
        control_builder.Connect(
            control_plant.get_state_output_port(control_iiwa),
            controller.get_input_port_estimated_state())
        control_builder.Connect(
            controller.get_output_port_control(),
            control_plant.get_actuation_input_port(control_iiwa))

        # Control to having the elbow slightly bent.
        q_iiwa_final = np.zeros(7)
        q_iiwa_final[3] = -np.pi / 2
        t_start = 0.
        t_end = 1.
        nx = 2 * ndof

        def q_desired_interpolator(t: ContextTimeArg) -> VectorArg(nx):
            s = (t - t_start) / (t_end - t_start)
            ds = 1 / (t_end - t_start)
            q = q_iiwa + s * (q_iiwa_final - q_iiwa)
            v = ds * (q_iiwa_final - q_iiwa)
            x = np.hstack((q, v))
            return x

        interpolator = control_builder.AddSystem(FunctionSystem(
            q_desired_interpolator))
        control_builder.Connect(
            interpolator.get_output_port(0),
            controller.get_input_port_desired_state())

        control_diagram = control_builder.Build()
        control_d_context = control_diagram.CreateDefaultContext()
        control_context = control_plant.GetMyContextFromRoot(control_d_context)
        to_control.copy_state(context, control_context)
        util.compare_frame_poses(
            plant, context, control_plant, control_context,
            "iiwa_link_0", "iiwa_link_7")
        util.compare_frame_poses(
            plant, context, control_plant, control_context,
            "body", "left_finger")

        from_control = to_control.inverse()

        def viz_monitor(control_d_context):
            # Simulate control, visualizing in original diagram.
            assert (control_context is
                    control_plant.GetMyContextFromRoot(control_d_context))
            from_control.copy_state(control_context, context)
            d_context.SetTime(control_d_context.get_time())
            diagram.Publish(d_context)
            return EventStatus.DidNothing()

        simulator = Simulator(control_diagram, control_d_context)
        simulator.Initialize()
        if VISUALIZE:
            simulator.set_monitor(viz_monitor)
            simulator.set_target_realtime_rate(1.)
        simulator.AdvanceTo(t_end)


if __name__ == "__main__":
    if "--visualize" in sys.argv:
        sys.argv.remove("--visualize")
        VISUALIZE = True
    unittest.main()
