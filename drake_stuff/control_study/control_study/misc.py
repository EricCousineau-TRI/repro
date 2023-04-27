import numpy as np

from pydrake.geometry import DrakeVisualizer, DrakeVisualizerParams, Role
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.math import (
    SpatialAcceleration,
    SpatialForce,
    SpatialVelocity,
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlant, MultibodyPlantConfig
from pydrake.multibody.tree import ModelInstanceIndex

from pydrake.geometry import DrakeVisualizer, DrakeVisualizerParams, Role
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.math import (
    SpatialAcceleration,
    SpatialForce,
    SpatialVelocity,
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlant, MultibodyPlantConfig
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem


from pydrake.all import BasicVector

from control_study.systems import (
    declare_simple_cache,
)
from control_study.trajectories import (
    make_se3_spline_trajectory,
)

def unzip(xs):
    return tuple(zip(*xs, strict=True))


class SimpleLogger(LeafSystem):
    def __init__(self, period_sec, func):
        super().__init__()
        # WARNING: This is undeclared state! This will only work for a single
        # simulation.
        self.log = []

        def on_discrete_update(context, raw_state):
            instant = func(context)
            self.log.append(instant)

        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=period_sec,
            offset_sec=0.0,
            update=on_discrete_update,
        )


def pad_tape(ts, Xs, pad):
    ts = np.array(ts)
    ts += pad
    ts = [0.0] + ts.tolist() + [ts[-1] + pad]
    ts = np.asarray(ts)
    Xs = [Xs[0]] + Xs + [Xs[-1]]
    return ts, Xs


def make_sample_pose_traj(dT, X_WG, X_extr, X_intr):
    Xs = [
        X_WG,
        X_extr @ X_WG @ X_intr,
        X_WG,
        X_extr.inverse() @ X_WG @ X_intr.inverse(),
        X_WG,
    ]
    ts = np.arange(len(Xs)) * dT
    ts, Xs = pad_tape(ts, Xs, pad=dT / 4.0)
    traj = make_se3_spline_trajectory(ts, Xs)
    t_f = ts[-1]

    def traj_saturate(t):
        # Saturate at end.
        if t >= t_f:
            t = t_f
        X, V, A = traj(t)
        return X, V, A

    return traj_saturate, t_f


def make_sim_setup(time_step):
    builder = DiagramBuilder()
    config = MultibodyPlantConfig(
        time_step=time_step,
        discrete_contact_solver="sap",
    )
    plant, scene_graph = AddMultibodyPlant(config, builder)
    DrakeVisualizer.AddToBuilder(
        builder,
        scene_graph,
        params=DrakeVisualizerParams(role=Role.kIllustration),
    )
    parser = Parser(plant)
    parser.AddModelsFromUrl(
        "package://drake/manipulation/models/franka_description/urdf/panda_arm.urdf"
    )
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("panda_link0"),
    )
    plant.Finalize()

    frame_G = plant.GetFrameByName("panda_link8")
    # Blech.
    model = ModelInstanceIndex(2)
    builder.ExportOutput(plant.get_state_output_port(), "state")
    builder.ExportInput(plant.get_actuation_input_port(model), "torque")
    diagram = builder.Build()
    return diagram, plant, scene_graph, frame_G


class PlantAccess:
    state_output_port = None
    torque_input_port = None

    def read_plant_context(self, root_context):
        raise NotImplementedError()

    def write_plant_context(self, root_context, context):
        raise NotImplementedError()


class DirectPlant(PlantAccess):
    @staticmethod
    def AddToBuilder(builder, diagram, plant):
        access = DirectPlant(diagram, plant)
        builder.AddSystem(diagram)
        return access

    def __init__(self, diagram, plant):
        super().__init__()
        self.state_output_port = diagram.GetOutputPort("state")
        self.torque_input_port = diagram.GetInputPort("torque")
        self.plant = plant

    def read_plant_context(self, root_context):
        return self.plant.GetMyContextFromRoot(root_context)

    def write_plant_context(self, root_context, context):
        assert context is self.read_plant_context(root_context)


def calc_dynamics(plant, context):
    M = plant.CalcMassMatrix(context)
    C = plant.CalcBiasTerm(context)
    tau_g = plant.CalcGravityGeneralizedForces(context)
    return M, C, tau_g


class EulerAccelPlant(LeafSystem, PlantAccess):
    @staticmethod
    def AddToBuilder(builder, diagram, plant, period_sec):
        access = EulerAccelPlant(diagram, plant, period_sec)
        builder.AddSystem(access)
        return access

    def __init__(self, diagram, plant, period_sec):
        super().__init__()
        self.plant = plant
        self.num_q = plant.num_positions()
        assert plant.num_velocities() == self.num_q
        self.period_sec = period_sec

        self.diagram = diagram
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.context = self.plant.GetMyContextFromRoot(self.diagram_context)

        self.state_index = self.DeclareDiscreteState(2 * self.num_q)
        # N.B. We do not want this to be an initialization event because it
        # accumulates / integrates.
        self.DeclarePeriodicDiscreteUpdateEvent(
            self.period_sec,
            0.0,
            self.discrete_update,
        )
        self.DeclarePeriodicPublishEvent(
            period_sec=self.period_sec,
            offset_sec=0.0,
            publish=self.on_publish,
        )
        self.torque_input_port = self.DeclareVectorInputPort(
            "u", BasicVector(self.num_q)
        )
        self.state_output_port = self.DeclareStateOutputPort(
            "x", self.state_index
        )

    def discrete_update(self, sys_context, discrete_state):
        # Poll inputs.
        u = self.torque_input_port.Eval(sys_context)
        x = self.get_state(sys_context)
        # Set state and compute dynamics.
        self.plant.SetPositionsAndVelocities(self.context, x)
        M, C, tau_g = calc_dynamics(self.plant, self.context)
        vd = np.linalg.inv(M) @ (u + tau_g - C)
        # Integrate.
        q = x[:self.num_q]
        v = x[self.num_q:]
        h = self.period_sec
        v_new = v + h * vd
        q_new = q + h * v + 0.5 * h * h * vd
        x_new = np.concatenate([q_new, v_new])
        # Store result.
        discrete_state.set_value(self.state_index, x_new)

    def on_publish(self, sys_context):
        x = self.get_state(sys_context)
        self.plant.SetPositionsAndVelocities(self.context, x)
        self.diagram.ForcedPublish(self.diagram_context)

    def get_state(self, sys_context):
        x_state = sys_context.get_discrete_state(self.state_index)
        x = x_state.get_value()
        return x

    def set_state(self, sys_context, x):
        x_state = (
            sys_context.get_mutable_state()
            .get_mutable_discrete_state(self.state_index)
        )
        x_state.set_value(x)

    def read_plant_context(self, root_context):
        sys_context = self.GetMyContextFromRoot(root_context)
        x = self.get_state(sys_context)
        self.plant.SetPositionsAndVelocities(self.context, x)
        return self.context

    def write_plant_context(self, root_context, context):
        sys_context = self.GetMyContextFromRoot(root_context)
        x = self.plant.GetPositionsAndVelocities(context)
        self.set_state(sys_context, x)
