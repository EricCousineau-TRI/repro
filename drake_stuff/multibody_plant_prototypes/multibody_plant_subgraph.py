"""
Provides simple subgraph functionality for MultibodyPlant.

The MultibodyPlantSubgraph class was designed for the following workflows:

- Decomposition (breaking a big model apart) by taking a subgraph of an
  existing plant and add it to a new plant.
    - This can be used for specifying controllers from an existing "super"
    plant.

- Composition (making a big model) by taking multiple subgraphs from
  existing plants and add them to a new plant.
    - This can be used to assemble scenes in a hierarchical way.
    - this can also be used to "cache" parsing results; rather than trying
      to remember which model instance came from which URDF / SDFormat
      file, you instead take the subgraph.

- "Un-finalizing" a MultibodyPlant by copying it entirely. This can be
  done by simply making a subgraph of the whole plant and adding it to a
  new plant.

- A "cheap" way to extract a (MultibodyPlant, SceneGraph) pair from a
  diagram that is not suited for optimization / .ToAutoDiffXd() (e.g.
  visualization).

For examples of these workflows, please see `multibody_plant_subgraph_test.py`.
"""
from collections import OrderedDict
import copy

from pydrake.geometry import GeometryId, GeometryInstance, SceneGraph
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.tree import (
    BallRpyJoint,
    BodyFrame,
    BodyIndex,
    FixedOffsetFrame,
    Frame,
    FrameIndex,
    Joint,
    JointActuator,
    JointActuatorIndex,
    JointIndex,
    ModelInstanceIndex,
    PrismaticJoint,
    RevoluteJoint,
    RigidBody,
    UniversalJoint,
    WeldJoint,
    default_model_instance,
    world_model_instance,
)
from pydrake.systems.framework import DiagramBuilder


def _get_plant_aggregate(num_func, get_func, index_cls, model_instances=None):
    items = []
    for i in range(num_func()):
        item = get_func(index_cls(i))
        if model_instances is None or item.model_instance() in model_instances:
            items.append(item)
    return items


def _check_plant_aggregate(get_func, item):
    assert get_func(item.index()) is item, (f"{item}: {item.name()}")


def GetModelInstances(plant):
    # TODO(eric.cousineau): Hoist this somewhere?
    return _get_plant_aggregate(
        plant.num_model_instances, lambda x: x,
        ModelInstanceIndex)


def GetBodies(plant, model_instances=None):
    # TODO(eric.cousineau): Hoist this somewhere?
    return _get_plant_aggregate(
        plant.num_bodies, plant.get_body, BodyIndex, model_instances)


def GetFrames(plant, model_instances=None):
    # TODO(eric.cousineau): Hoist this somewhere?
    return _get_plant_aggregate(
        plant.num_frames, plant.get_frame, FrameIndex, model_instances)


def GetFramesAttachedTo(plant, bodies):
    # TODO(eric.cousineau): Hoist this somewhere?
    frames = []
    for frame in GetFrames(plant):
        if frame.body() in bodies:
            frames.append(frame)
    return frames


def GetJoints(plant, model_instances=None):
    # TODO(eric.cousineau): Hoist this somewhere?
    return _get_plant_aggregate(
        plant.num_joints, plant.get_joint, JointIndex, model_instances)


def IsJointSolelyConnectedTo(joint, bodies):
    # TODO(eric.cousineau): Hoist this somewhere?
    parent = joint.parent_body()
    child = joint.child_body()
    return parent in bodies and child in bodies


def GetJointsSolelyConnectedTo(plant, bodies):
    # TODO(eric.cousineau): Hoist this somewhere?
    return [
        joint for joint in GetJoints(plant)
        if IsJointSolelyConnectedTo(joint, bodies)]


def GetJointActuators(plant, model_instances=None):
    # TODO(eric.cousineau): Hoist this somewhere?
    return _get_plant_aggregate(
        plant.num_actuators, plant.get_joint_actuator,
        JointActuatorIndex)


def GetJointActuatorsByJoints(plant, joints):
    # TODO(eric.cousineau): Hoist this somewhere?
    joint_actuators = []
    for joint_actuator in GetJointActuators(plant):
        if joint_actuator.joint() in joints:
            joint_actuators.append(joint_actuator)
    return joint_actuators


def AddOrGetModelInstance(plant, name):
    # TODO(eric.cousineau): Hoist this somewhere?
    if not plant.HasModelInstanceNamed(name):
        return plant.AddModelInstance(name)
    else:
        return plant.GetModelInstanceByName(name)


def GetGeometries(plant, scene_graph, bodies):
    # Returns all geometry ids attached to bodies. Assumes corresponding frames
    # have been added.
    geometries = []
    inspector = scene_graph.model_inspector()
    for geometry in inspector.GetAllGeometryIds():
        body = plant.GetBodyFromFrameId(inspector.GetFrameId(geometry))
        if body in bodies:
            geometries.append(geometry)
    return geometries


def model_instance_remap_to_default(plant_src, model_instance_src, plant_dest):
    """Remap to default model instance (intended for plant_dest)."""
    return default_model_instance()


def create_model_instance_remap_by_name(func):
    """Creates remap based on transforming model instance name.
    Example:

        @create_model_instance_remap_by_name
        def model_instance_remap_with_prefix(name_src):
            name_dest = f"my_custom_prefix::{name_src}"
            return name_dest
    """

    def remap(plant_src, model_instance_src, plant_dest):
        name = plant_src.GetModelInstanceName(model_instance_src)
        new_name = func(name)
        return AddOrGetModelInstance(plant_dest, new_name)

    return remap


@create_model_instance_remap_by_name
def model_instance_remap_same_name(name):
    return name


def make_multibody_plant_elements_cls(
        name, container_cls, container_extend=None):

    class Impl:
        """Provides a container for elements specific to
        MultibodyPlantSubgraph."""
        # N.B. This will be renamed.
        def __init__(self):
            self.model_instances = container_cls()
            self.bodies = container_cls()
            self.frames = container_cls()
            self.joints = container_cls()
            self.joint_actuators = container_cls()
            # TODO(eric.cousineau): How to handle force elements?
            self.geometries = container_cls()
            # TODO(eric.cousineau): How to handle collision filters?

        def __copy__(self):
            """Makes a "level 2" shallow copy."""
            new = Impl()
            new.model_instances = copy.copy(model_instances)
            new.bodies = copy.copy(bodies)
            new.frames = copy.copy(frames)
            new.joints = copy.copy(joints)
            new.joint_actuators = copy.copy(joint_actuators)
            new.geometries = copy.copy(geometries)
            return new

        if container_extend:
            def __iadd__(self, other):
                assert isinstance(other, Impl), (other)
                container_extend(self.model_instances, other.model_instances)
                container_extend(self.bodies, other.bodies)
                container_extend(self.frames, other.frames)
                container_extend(self.joints, other.joints)
                container_extend(self.joint_actuators, other.joint_actuators)
                container_extend(self.geometries, other.geometries)
                return self

    # Rename the class.
    Impl.__name__ = name
    Impl.__qualname__ = name
    return Impl


MultibodyPlantElementsList = make_multibody_plant_elements_cls(
    "MultibodyPlantElementsList",
    container_cls=list,
    container_extend=list.extend,
)
MultibodyPlantElementsDict = make_multibody_plant_elements_cls(
    "MultibodyPlantElementsList",
    container_cls=OrderedDict,
)


class MultibodyPlantAssociations(MultibodyPlantElementsDict):
    """Mappings of subgraph elements from plant_src to plant_dest."""
    def __init__(self, plant_src, plant_dest):
        self.plant_src = plant_src
        self.plant_dest = plant_dest
        super().__init__()

    def copy_state(self, context_src, context_dest):
        """Copies the (physical) state from context_src to context_dest."""
        plant_src = self.plant_src
        plant_dest = self.plant_dest
        for body_src, body_dest in self.bodies.items():
            if body_dest.is_floating():
                X_WB = plant_src.CalcRelativeTransform(
                    context_src, plant_src.world_frame(),
                    body_src.body_frame())
                V_WB = plant_src.EvalBodySpatialVelocityInWorld(
                    context_src, body_src)
                plant_dest.SetFreeBodyPose(context_dest, body_dest, X_WB)
                plant_dest.SetFreeBodySpatialVelocity(
                    body_dest, V_WB, context_dest)
        for joint_src, joint_dest in self.joints.items():
            qj = get_joint_positions(plant_src, context_src, joint_src)
            set_joint_positions(plant_dest, context_dest, joint_dest, qj)
            vj = get_joint_velocities(plant_src, context_src, joint_src)
            set_joint_velocities(plant_dest, context_dest, joint_dest, vj)


def get_joint_positions(plant, context, joint):
    # TODO(eric.cousineau): Hoist to C++ / pydrake.
    q = plant.GetPositions(context)
    start = joint.position_start()
    count = joint.num_positions()
    return q[start:start + count].copy()


def set_joint_positions(plant, context, joint, qj):
    # TODO(eric.cousineau): Hoist to C++ / pydrake.
    q = plant.GetPositions(context)
    start = joint.position_start()
    count = joint.num_positions()
    q[start:start + count] = qj
    plant.SetPositions(context, q)


def get_joint_velocities(plant, context, joint):
    # TODO(eric.cousineau): Hoist to C++ / pydrake.
    v = plant.GetVelocities(context)
    start = joint.velocity_start()
    count = joint.num_velocities()
    return v[start:start + count].copy()


def set_joint_velocities(plant, context, joint, vj):
    # TODO(eric.cousineau): Hoist to C++ / pydrake.
    v = plant.GetVelocities(context)
    start = joint.velocity_start()
    count = joint.num_velocities()
    v[start:start + count] = vj
    plant.SetVelocities(context, v)


def _get_model_instances(items):
    return set(x.model_instance() for x in items)


# TODO(eric.cousineau): Add some sort of "is connected" topology query for
# users?


class SubgraphPolicy:
    """Provides a policy to mutate a subgraph, and then mutate a "destination
    plant" after the subgraph is added to the model."""
    def mutate_subgraph(self, subgraph):
        raise NotImplemented

    def after_add(self, subgraph):
        raise NotImplemented


class DisconnectFromWorldSubgraphPolicy(SubgraphPolicy):
    """Removes the world body, and any joints, frames, etc. connected to it."""
    def mutate_subgraph(self, subgraph):
        elem_added = None
        elem_removed = MultibodyPlantElementsList()
        elem_removed += subgraph.remove_body(subgraph.plant_src.world_body())
        return elem_added, elem_removed

    def after_add(self, subgraph):
        elem_added = None
        elem_removed = None
        return elem_added, elem_removed


class FreezeJointSubgraphPolicy(SubgraphPolicy):
    """Defines a "policy" on subgraphs to allow a set of joints to be replaced
    with a weld."""

    def __init__(self, freeze_joints_src, context_src):
        self._freeze_joints_src = freeze_joints_src
        self._context_src = context_src

    @classmethod
    def from_plant(cls, plant, context_src, model_instances=None):
        # Get all non-weld joints.
        joints_src = GetJoints(plant, model_instances)
        for joint in list(joints_src):
            if joint.num_positions() == 0:
                joints_src.remove(joint)
        return cls(joints_src, context_src)

    def mutate_subgraph(self, subgraph):
        """Remove joints from subgraph. (They must already be part of the
        subgraph)."""
        elem_added = None
        elem_removed = MultibodyPlantElementsList()
        for joint_src in self._freeze_joints_src:
            elem_removed += subgraph.remove_joint(joint_src)
        return elem_added, elem_removed

    def after_add(self, src_to_dest):
        """Replace joints with welds. plant_src must be finalized, with the
        provided context."""
        elem_added = MultibodyPlantElementsList()
        elem_removed = None
        for joint_src in self._freeze_joints_src:
            # Ensure that this joint was removed.
            assert joint_src not in src_to_dest.joints
            # Replicate kinematics, but frozen.
            frame_P_src = joint_src.parent_body().body_frame()
            frame_C_src = joint_src.child_body().body_frame()
            X_PC = src_to_dest.plant_src.CalcRelativeTransform(
                self._context_src, frame_P_src, frame_C_src)
            frame_P_dest = src_to_dest.frames[frame_P_src]
            frame_C_dest = src_to_dest.frames[frame_C_src]
            weld = src_to_dest.plant_dest.WeldFrames(
                frame_P_dest, frame_C_dest, X_PC)
            src_to_dest.joints[joint_src] = weld
            elem_added.joints.append(weld)
        return elem_added, elem_removed


_DEFAULT_POLICIES = []
_POLICIES_ALREADY_MUTATED_SUBGRAPH = object()


class MultibodyPlantSubgraph:
    """
    Identifies a subpgraph of a MultibodyPlant, used to extracted from the
    original plant and "grafted" onto another plant. See module-level docstring
    for more information about workflows.

    The MultibodyPlantSubgraph only identifies topology; computations should be
    delegated to subplants.

    Invariants:

    - It does *not* matter if the source plant is finalize or not.
        - However, certain SubgraphPolicy's may require a context (e.g.
        freezing joints in place).

    - All subgraph model instances must be part of the given plant.

    - All subgraph bodies must be part of the subgraph  model instances.

    - All subgraph frames must be attached to the subgraph bodies.

    - All subgraph joints must be solely connected to the subgraph bodies.

    - All subgraph joint actuators must act solely on the subgraph joints.

    - All subgraph geometries must be attached to the subgraph bodies.

      Note that geometries have additional (hack) constraints:
            - The geometry must use the "scoped name" of
              "{body_model_instance}::{geometry_name}". This is necessary so
              that the geometry can be renamed appropriately.
            - A collision geometry *cannot* have any other roles. This is due
              to a constraint in how `MultibodyPlant` handles collision
              geometry.

    - This class only deals with SceneGraph's model geometry; it does not try
      to mutate any Context-stored geometries.
    """
    def __init__(
            self,
            plant_src,
            elem_src,
            scene_graph_src=None,
            policies=[],
            ):
        """Allows explicit specification for things like welding in place of
        articulation."""
        assert isinstance(plant_src, MultibodyPlant)
        assert isinstance(elem_src, MultibodyPlantElementsList)
        if scene_graph_src:
            assert isinstance(scene_graph_src, SceneGraph)
        if policies is None:
            policies = []
        self._plant_src = plant_src
        self._scene_graph_src = scene_graph_src
        self._elem_src = elem_src
        self._check_invariants()
        self._policies = []
        if _POLICIES_ALREADY_MUTATED_SUBGRAPH in policies:
            policies.remove(_POLICIES_ALREADY_MUTATED_SUBGRAPH)
            self._policies = policies
        else:
            for policy in policies:
                self.add_policy(policy)

    def add_policy(self, policy):
        """Adds a policy, and mutates this subgraph according to it."""
        assert isinstance(policy, SubgraphPolicy)
        elem_added, elem_removed = policy.mutate_subgraph(self)
        self._policies.append(policy)
        return elem_added, elem_removed

    def __copy__(self):
        """Defines a shallow copy."""
        return MultibodyPlantSubgraph(
            plant_src=self.plant_src,
            elem_src=copy.copy(self._elem_src),
            scene_graph_src=self._scene_graph_src,
            policies=[_POLICIES_ALREADY_MUTATED_SUBGRAPH] + self._policies,
        )

    @classmethod
    def from_bodies(
            cls,
            plant,
            bodies,
            scene_graph=None,
            model_instances=None):
        """Constructs a subgraph from a list of bodies and all elements
        associated with those bodies."""
        elem = MultibodyPlantElementsList()
        elem.bodies = list(bodies)
        if model_instances is None:
            elem.model_instances = list(_get_model_instances(elem.bodies))
        else:
            elem.model_instances = model_instances
        elem.joints = GetJointsSolelyConnectedTo(plant, elem.bodies)
        elem.joint_actuators = GetJointActuatorsByJoints(plant, elem.joints)
        elem.frames = GetFramesAttachedTo(plant, elem.bodies)
        if scene_graph is not None:
            elem.geometries = GetGeometries(plant, scene_graph, elem.bodies)
        return cls(plant, elem, scene_graph)

    @classmethod
    def from_plant(
            cls,
            plant,
            scene_graph=None,
            model_instances=None):
        """Constructs a subgraph from model_instances (or all of them if none
        is specified."""
        # Purposefully include world and default model instance.
        if model_instances is None:
            model_instances = GetModelInstances(plant)
        bodies = GetBodies(plant, model_instances)
        return cls.from_bodies(
            plant, bodies, scene_graph=scene_graph,
            model_instances=model_instances)

    @property
    def plant_src(self):
        """Accesses "source plant"."""
        return self._plant_src

    @property
    def scene_graph_src(self):
        """Accesses "source scene_graph"; may be None."""
        return self._scene_graph_src

    @property
    def elements_src(self):
        """Accesses a shallow copy of elements contained within subgraph."""
        return copy.copy(self._elem_src)

    def remove_body(self, body):
        """Removes body and all associated elements from this subgraph.
        Returns:
            MultibodyPlantElementsList containing all removed elements.
        """
        elem_src = self._elem_src
        elem_removed = MultibodyPlantElementsList()
        frames = [
            x for x in elem_src.frames
            if x.body() is body]
        for x in frames:
            elem_removed += self.remove_frame(x)
        joints = [
            x for x in elem_src.joints
            if body in (x.parent_body(), x.child_body())]
        for x in joints:
            elem_removed += self.remove_joint(x)
        if self._scene_graph_src is not None:
            geometries = GetGeometries(
                self._plant_src, self._scene_graph_src, [body])
            for x in geometries:
                elem_removed += self.remove_geometry(x)
        return elem_removed

    def remove_frame(self, frame):
        """Removes a frame from this subgraph.
        Returns:
            MultibodyPlantElementsList containing all removed elements.
        """
        # N.B. Since most elements interface with bodies, not frames, then it
        # is OK to remove a frame and not really have other dependent elements.
        elem_src = self._elem_src
        elem_removed = MultibodyPlantElementsList()
        elem_src.frames.remove(frame)
        elem_removed.frames.append(frame)
        return elem_removed

    def remove_joint(self, joint):
        """Removes a joint and its associated actuators from this subgraph.
        Returns:
            MultibodyPlantElementsList containing all removed elements.
        """
        elem_src = self._elem_src
        elem_removed = MultibodyPlantElementsList()
        elem_src.joints.remove(joint)
        elem_removed.joints.append(joint)
        joint_actuators = [
            x for x in elem_src.joint_actuators
            if x.joint() is joint]
        for x in joint_actuators:
            elem_removed += self.remove_joint_actuator(x)
        return elem_removed

    def remove_joint_actuator(self, joint_actuator):
        """Removes a joint actuator from this subgraph.
        Returns:
            MultibodyPlantElementsList containing all removed elements.
        """
        elem_src = self._elem_src
        elem_removed = MultibodyPlantElementsList()
        elem_src.joint_actuators.remove(joint_actuator)
        elem_removed.joint_actuators.append(joint_actuator)
        return elem_removed

    def remove_geometry(self, geometry):
        """Removes a geometry id from this subgraph.
        Returns:
            MultibodyPlantElementsList containing all removed elements.
        """
        elem_src = self._elem_src
        elem_src.geometries.remove(geometry)
        elem_removed.geometries.append(geometry)
        return elem_removed

    def add_to(
            self,
            plant_dest,
            scene_graph_dest=None,
            model_instance_remap=model_instance_remap_same_name):
        """Adds this subgraph onto a given plant_dest and scene_graph_dest.

        Args:
            plant_dest: "Destination plant".
            scene_graph_dest: "Destination scene graph".
                If this is specified, there must be a source scene_graph.
            model_instance_remap:
                Function of the form:

                    func(plant_src, model_instance_src, plant_dest)
                        -> model_instance_dest

                If this is a string, it will be a remap that simply renames all
                source model instances to a model instance of the given name.
        Returns:
            src_to_dest (MultibodyPlantAssociations) containing all elemen
                associations from this subgraph's plant_src (and
                scene_graph_src) to plant_dest (and scene_graph_dest).
        """
        assert isinstance(plant_dest, MultibodyPlant)
        if isinstance(model_instance_remap, str):
            model_instance_name = model_instance_remap
            model_instance_remap = create_model_instance_remap_by_name(
                lambda x: model_instance_name)

        plant_src = self._plant_src
        scene_graph_src = self._scene_graph_src
        elem_src = self._elem_src

        src_to_dest = MultibodyPlantAssociations(plant_src, plant_dest)

        # TODO(eric.cousineau): It would be nice if there were cloning
        # operations on the tree elements which did not copy the index, and
        # took this mapping.... Meh for now.

        def setitem(d, k, v):
            # Exlcusively set an item.
            assert k not in d, k
            d[k] = v

        def add_model_instance_dest(model_instance_src):
            # Remaps a model instance to be added to the destination plant.
            model_instance_dest = model_instance_remap(
                plant_src, model_instance_src, plant_dest)
            setitem(
                src_to_dest.model_instances,
                model_instance_src, model_instance_dest)
            return model_instance_dest

        def add_body_dest(body_src):
            # Clones a Body to be added to the destination plant.
            model_instance_src = body_src.model_instance()
            model_instance_dest = src_to_dest.model_instances[
                model_instance_src]
            body_dest = plant_dest.AddRigidBody(
                name=body_src.name(),
                M_BBo_B=body_src.default_spatial_inertia(),
                model_instance=model_instance_dest,
            )
            setitem(src_to_dest.bodies, body_src, body_dest)
            # Set default state.
            X_WB = plant_src.GetDefaultFreeBodyPose(body_src)
            plant_dest.SetDefaultFreeBodyPose(body_dest, X_WB)
            # Register body frame.
            frame_src = body_src.body_frame()
            frame_dest = body_dest.body_frame()
            setitem(src_to_dest.frames, frame_src, frame_dest)
            return frame_src

        def add_frame_dest(frame_src):
            # Clones a Frame to be added to the destination plant.
            # N.B. BodyFrame's should be handled outside of this method.
            assert not isinstance(frame_src, BodyFrame), (
                f"{frame_src}, {frame_src.name()}")
            parent_frame_src = frame_src.body().body_frame()
            parent_frame_dest = src_to_dest.frames[parent_frame_src]
            model_instance_src = frame_src.model_instance()
            model_instance_dest = (
                src_to_dest.model_instances[model_instance_src])
            frame_dest = FixedOffsetFrame(
                name=frame_src.name(), P=parent_frame_dest,
                X_PF=frame_src.GetFixedPoseInBodyFrame(),
                model_instance=model_instance_dest,
            )
            plant_dest.AddFrame(frame_dest)
            F = frame_src
            setitem(src_to_dest.frames, frame_src, frame_dest)

        def add_joint_dest(joint_src):
            # Clones a Joint to be added to the destination plant.
            P = joint_src.frame_on_parent()
            frame_on_parent_dest = src_to_dest.frames[
                joint_src.frame_on_parent()]
            frame_on_child_dest = src_to_dest.frames[
                joint_src.frame_on_child()]
            if isinstance(joint_src, BallRpyJoint):
                joint_dest = BallRpyJoint(
                    name=joint_src.name(),
                    frame_on_parent=frame_on_parent_dest,
                    frame_on_child=frame_on_child_dest,
                    damping=joint_src.damping(),
                )
            elif isinstance(joint_src, PrismaticJoint):
                joint_dest = PrismaticJoint(
                    name=joint_src.name(),
                    frame_on_parent=frame_on_parent_dest,
                    frame_on_child=frame_on_child_dest,
                    axis=joint_src.translation_axis(),
                    damping=joint_src.damping(),
                )
            elif isinstance(joint_src, RevoluteJoint):
                joint_dest = RevoluteJoint(
                    name=joint_src.name(),
                    frame_on_parent=frame_on_parent_dest,
                    frame_on_child=frame_on_child_dest,
                    axis=joint_src.revolute_axis(),
                    damping=joint_src.damping(),
                )
            elif isinstance(joint_src, UniversalJoint):
                joint_dest = UniversalJoint(
                    name=joint_src.name(),
                    frame_on_parent=frame_on_parent_dest,
                    frame_on_child=frame_on_child_dest,
                    damping=joint_src.damping(),
                )
            elif isinstance(joint_src, WeldJoint):
                joint_dest = WeldJoint(
                    name=joint_src.name(),
                    parent_frame_P=frame_on_parent_dest,
                    child_frame_C=frame_on_child_dest,
                    X_PC=joint_src.X_PC(),
                )
            else:
                assert False, f"Cannot clone: {type(joint_src)}"
            joint_dest.set_position_limits(
                joint_src.position_lower_limits(),
                joint_src.position_upper_limits())
            joint_dest.set_velocity_limits(
                joint_src.velocity_lower_limits(),
                joint_src.velocity_upper_limits())
            joint_dest.set_acceleration_limits(
                joint_src.acceleration_lower_limits(),
                joint_src.acceleration_upper_limits())
            joint_dest.set_default_positions(
                joint_src.default_positions())
            plant_dest.AddJoint(joint_dest)
            setitem(src_to_dest.joints, joint_src, joint_dest)

        def add_joint_actuator_dest(joint_actuator_src):
            # Clones a JointActuator to be added to the destination plant.
            joint_src = joint_actuator_src.joint()
            joint_dest = src_to_dest.joints[joint_src]
            joint_actuator_dest = plant_dest.AddJointActuator(
                joint_actuator_src.name(), joint_dest,
                effort_limit=joint_actuator_src.effort_limit())
            setitem(
                src_to_dest.joint_actuators,
                joint_actuator_src, joint_actuator_dest)

        def add_geometry_dest(geometry_src):
            # Clones the geometry represented by GeometryId to be added to
            # the destination plant / scene_graph.
            assert scene_graph_src is not None
            inspector_src = scene_graph_src.model_inspector()
            inspector_dest = scene_graph_dest.model_inspector()
            frame_id_src = inspector_src.GetFrameId(geometry_src)
            body_src = plant_src.GetBodyFromFrameId(frame_id_src)
            assert body_src is not None
            body_dest = src_to_dest.bodies[body_src]
            frame_id_dest = plant_dest.GetBodyFrameIdOrThrow(body_dest.index())
            assert frame_id_dest is not None
            geometry_instance_dest = inspector_src.CloneGeometryInstance(
                geometry_src)

            # Use new "scoped" name.
            model_instance_name_src = plant_src.GetModelInstanceName(
                body_src.model_instance())
            model_instance_name_dest = plant_dest.GetModelInstanceName(
                body_dest.model_instance())
            prefix_src = f"{model_instance_name_src}::"
            scoped_name_src = geometry_instance_dest.name()
            assert scoped_name_src.startswith(prefix_src), (
                f"'{scoped_name_src}' should start with '{prefix_src}'")
            unscoped_name = scoped_name_src[len(prefix_src):]
            scoped_name_dest = f"{model_instance_name_dest}::{unscoped_name}"
            geometry_instance_dest.set_name(scoped_name_dest)

            # At present, collision geometries that are to be used for
            # simulation must be registered with the plant (rather than
            # queried). For this reason, all collision geometries must *only*
            # be collision geometries, and have to go through the plant.
            # TODO(eric.cousineau): How to relax this constraint? How can we
            # register with SceneGraph only? See Sean's TODO in
            # MultibodyPlant.RegisterCollisionGeometry.
            proximity_properties = (
                geometry_instance_dest.proximity_properties())
            if proximity_properties is not None:
                assert geometry_instance_dest.perception_properties() is None
                assert geometry_instance_dest.illustration_properties() is None
                geometry_dest = plant_dest.RegisterCollisionGeometry(
                    body=body_dest,
                    X_BG=geometry_instance_dest.pose(),
                    shape=geometry_instance_dest.release_shape(),
                    name=unscoped_name,
                    properties=proximity_properties)
            else:
                # Register as normal.
                source_id_dest = plant_dest.get_source_id()
                geometry_dest = scene_graph_dest.RegisterGeometry(
                    source_id_dest, frame_id_dest, geometry_instance_dest)
            setitem(src_to_dest.geometries, geometry_src, geometry_dest)

        for model_instance_src in elem_src.model_instances:
            add_model_instance_dest(model_instance_src)

        setitem(
            src_to_dest.bodies,
            plant_src.world_body(), plant_dest.world_body())
        builtin_bodies_src = [plant_src.world_body()]
        setitem(
            src_to_dest.frames,
            plant_src.world_frame(), plant_dest.world_frame())
        builtin_frames_src = [plant_src.world_frame()]

        for body_src in elem_src.bodies:
            if body_src in builtin_bodies_src:
                continue
            body_frame_src = add_body_dest(body_src)
            builtin_frames_src.append(body_frame_src)

        for frame_src in elem_src.frames:
            if frame_src in builtin_frames_src:
                # Already mapped.
                continue
            add_frame_dest(frame_src)

        for joint_src in elem_src.joints:
            add_joint_dest(joint_src)

        for joint_actuator_src in elem_src.joint_actuators:
            add_joint_actuator_dest(joint_actuator_src)

        if scene_graph_dest is not None:
            for geometry_src in elem_src.geometries:
                add_geometry_dest(geometry_src)

        for policy in self._policies:
            policy.after_add(src_to_dest)

        return src_to_dest

    def _check_invariants(self):
        # Ensures that current subgraph elements / topology satisifes class
        # invariants.
        plant = self._plant_src
        scene_graph = self._scene_graph_src
        elem_src = self._elem_src

        # Check bodies.
        for body in elem_src.bodies:
            assert isinstance(body, RigidBody)
            _check_plant_aggregate(plant.get_body, body)
            assert body.model_instance() in elem_src.model_instances

        # Check frames.
        for frame in elem_src.frames:
            assert isinstance(frame, Frame)
            _check_plant_aggregate(plant.get_frame, frame)
            assert frame.body() in elem_src.bodies
            assert frame.model_instance() in elem_src.model_instances

        # Check joints.
        for joint in elem_src.joints:
            assert isinstance(joint, Joint)
            _check_plant_aggregate(plant.get_joint, joint)
            assert IsJointSolelyConnectedTo(joint, elem_src.bodies)
            assert joint.model_instance() in elem_src.model_instances

        # Check actuators.
        for joint_actuator in elem_src.joint_actuators:
            assert isinstance(joint_actuator, JointActuator), joint_actuator
            _check_plant_aggregate(plant.get_joint_actuator, joint_actuator)
            assert joint_actuator.joint() in elem_src.joints
            assert joint_actuator.model_instance() in elem_src.model_instances

        # Check geometries.
        if scene_graph is not None:
            assert plant.geometry_source_is_registered()
            inspector = scene_graph.model_inspector()
            for geometry in elem_src.geometries:
                assert isinstance(geometry, GeometryId)
                body = plant.GetBodyFromFrameId(inspector.GetFrameId(geometry))
                assert body in elem_src.bodies
        else:
            assert elem_src.geometries == []


def parse_as_multibody_plant_subgraph(
        model_file, with_scene_graph=True, make_parser=Parser):
    """Parses a MultibodyPlant into subgraph for later use.

    This can be used to "cache" parsing results, and make it so that you an add
    a model multiple times without reparsing it, as well as being able to add
    models to a new scene without reparsing it.

    Args:
        model_file: File to parse.
        with_scene_graph:
            If true, will add a scene_graph via a diagram.
            Otherwise, will use plant directly.
        make_parser:
            How to construct the Parser given the plant. Override this if you
            need package paths.
    Returns:
        subgraph, model_instance
    """
    # TODO(eric.cousineau): Handle adding multiple models?
    arbitrary_dt = 0.1
    if with_scene_graph:
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=arbitrary_dt)
    else:
        plant = MultibodyPlant(time_step=arbitrary_dt)
        scene_graph = None
    model = make_parser(plant).AddModelFromFile(model_file, "model")
    # N.B. Finalize and Build are not necessary.
    return MultibodyPlantSubgraph.from_plant(
        plant, scene_graph), model


def copy_plant_and_scene_graph(plant_src, scene_graph_src):
    """Copies (plant_src, scene graph) into a new diagram.

    This can be used to copy a plant+scene_graph specifically for using in
    optimization (where you cannot have visualizers, etc. attached).

    Returns:
        plant, scene_graph, src_to_dest, diagram
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, plant_src.time_step())
    subgraph = MultibodyPlantSubgraph.from_plant(plant_src, scene_graph_src)
    src_to_dest = subgraph.add_to(plant, scene_graph)
    plant.Finalize()
    diagram = builder.Build()
    return plant, scene_graph, src_to_dest, diagram
