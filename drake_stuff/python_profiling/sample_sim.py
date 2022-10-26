"""
See README for more details.

Can use --use_cc to check profiling with C++ components.
"""

import argparse
import cProfile as profile
from contextlib import contextmanager
import dataclasses as dc
import os
import pstats
import signal
import subprocess

import numpy as np

from pydrake.common.cpp_param import List
from pydrake.common.value import Value
from pydrake.geometry import DrakeVisualizer, DrakeVisualizerParams, Role
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.math import SpatialForce, SpatialVelocity
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlant,
    ConnectContactResultsToDrakeVisualizer,
    ExternallyAppliedSpatialForce,
    MultibodyPlantConfig,
)
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (
    BasicVector,
    DiagramBuilder,
    LeafSystem,
)

from components import get_frame_spatial_velocity, set_default_frame_pose


def _use_cc():
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cc", action="store_true")
    args, argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + argv
    return args.use_cc


if _use_cc():
    from components_cc import FloatingBodyPoseController
else:
    from components import FloatingBodyPoseController


def fix_port(parent_context, port, value):
    context = port.get_system().GetMyContextFromRoot(parent_context)
    return port.FixValue(context, value)


def eval_port(port, parent_context):
    context = port.get_system().GetMyContextFromRoot(parent_context)
    return port.Eval(context)


class FloatingObject:
    def __init__(self, name, plant, model, *, body_name, frame_P_name):
        self._plant = plant
        self.name = name
        self.model = model
        self.body = plant.GetBodyByName(body_name, model)
        self.frame_P = plant.GetFrameByName(frame_P_name, model)
        # Pending build_control().
        self.controller = None
        self.post_init = None

    def build_control(self, builder, frame_T):
        plant = self._plant
        self.controller = FloatingBodyPoseController.AddToBuilder(
            builder,
            plant,
            self.model,
            frame_T,
            self.frame_P,
            add_centering=False,
            connect_to_plant=False,
            name=f"controller_{self.name}",
        )

        def post_init(context):
            plant_context = plant.GetMyContextFromRoot(context)
            X_TP_init = plant.CalcRelativeTransform(
                plant_context, frame_T, self.frame_P
            )
            self._X_TPdes_command = fix_port(
                context, self.controller.X_TPdes_input, X_TP_init
            )
            self._V_TPdes_command = fix_port(
                context, self.controller.V_TPdes_input, SpatialVelocity.Zero()
            )

        self.post_init = post_init

    def set_desired_reference(self, X_TPdes, V_TPdes=None):
        if V_TPdes is None:
            V_TPdes = SpatialVelocity.Zero()
        self._X_TPdes_command.GetMutableData().set_value(X_TPdes)
        self._V_TPdes_command.GetMutableData().set_value(V_TPdes)


class ForceAggregator(LeafSystem):
    # TODO(eric.cousineau): Replace with Drake system when available:
    # https://github.com/robotlocomotion/drake/pull/18171

    def __init__(self, num_inputs):
        super().__init__()
        forces_cls = Value[List[ExternallyAppliedSpatialForce]]
        self._num_inputs = num_inputs
        for i in range(self._num_inputs):
            self.DeclareAbstractInputPort(
                f"forces_in_{i}", model_value=forces_cls(),
            )
        self.DeclareAbstractOutputPort(
            "forces_out", alloc=forces_cls, calc=self._calc_forces,
        )

    def _calc_force_input(self, context, i):
        return self.get_input_port(i).Eval(context)

    def _calc_forces(self, context, output):
        forces_out = []
        for i in range(self._num_inputs):
            forces_out += self._calc_force_input(context, i)
        output.set_value(forces_out)

    @staticmethod
    def AddToBuilder(builder, force_ports):
        num_inputs = len(force_ports)
        system = builder.AddSystem(ForceAggregator(num_inputs))
        for i, port in enumerate(force_ports):
            builder.Connect(
                port, system.get_input_port(i),
            )
        return system


@dc.dataclass
class Setup:
    diagram: object
    diagram_context: object
    plant: object
    plant_context: object
    main: object
    secondary: object
    simulator: object
    frame_T: object


def build_sim(
    *,
    time_step,
    add_main=True,
    add_secondary=True,
    no_gravity=True,
    add_control=True,
):
    builder = DiagramBuilder()
    config = MultibodyPlantConfig(
        time_step=time_step,
        discrete_contact_solver="sap",
        contact_model="hydroelastic_with_fallback",
    )
    plant, scene_graph = AddMultibodyPlant(config, builder)

    frame_T = plant.world_frame()
    parser = Parser(plant)

    floaters = []

    sphere_file = "./sphere.sdf"

    if add_main:
        main_model = parser.AddModelFromFile(sphere_file, "main")
        main = FloatingObject(
            name="main",
            plant=plant,
            model=main_model,
            body_name="body",
            frame_P_name="body",
        )
    else:
        main = None

    if add_secondary:
        secondary_model = parser.AddModelFromFile(sphere_file, "secondary")
        secondary = FloatingObject(
            name="secondary",
            plant=plant,
            model=secondary_model,
            body_name="body",
            frame_P_name="body",
        )
    else:
        secondary = None

    floaters = []
    if add_main:
        floaters.append(main)
    if add_secondary:
        floaters.append(secondary)

    if no_gravity:
        plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, 0.0])

    plant.Finalize()

    L = 0.05
    if add_main:
        set_default_frame_pose(
            plant, main.frame_P, RigidTransform([0, 0, L]),
        )
    if add_secondary:
        set_default_frame_pose(
            plant, secondary.frame_P, RigidTransform([0, -5 * L, L]),
        )

    if add_control:
        force_ports = []
        for floater in floaters:
            floater.build_control(builder, frame_T)
            force_ports.append(floater.controller.forces_output)

        force_agg = ForceAggregator.AddToBuilder(builder, force_ports)
        builder.Connect(
            force_agg.get_output_port(),
            plant.get_applied_spatial_force_input_port(),
        )

    DrakeVisualizer.AddToBuilder(
        builder,
        scene_graph,
        params=DrakeVisualizerParams(role=Role.kIllustration),
    )
    ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()

    if add_control:
        for floater in floaters:
            floater.post_init(diagram_context)

    simulator = Simulator(diagram, context=diagram_context)
    simulator.set_target_realtime_rate(1.0)

    plant_context = plant.GetMyContextFromRoot(diagram_context)

    return Setup(
        plant=plant,
        plant_context=plant_context,
        diagram=diagram,
        diagram_context=diagram_context,
        main=main,
        secondary=secondary,
        simulator=simulator,
        frame_T=frame_T,
    )


@contextmanager
def use_cprofile(output_file):
    """
    Use cprofile in specific context.
    """
    pr = profile.Profile()
    pr.enable()
    yield
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats("tottime", "cumtime")
    stats.dump_stats(output_file)


@contextmanager
def use_py_spy(output_file, *, sudo=False):
    """Use py-spy in specific context."""
    args = ["py-spy", "record", "-o", output_file, "--pid", str(os.getpid())]
    if sudo:
        args = ["sudo"] + args
    p = subprocess.Popen(args)
    # TODO(eric.cousineau): Startup time of py-spy may lag behind other
    # instructions. However, maybe can assume this isn't critical for profiling
    # purposes?
    yield
    p.send_signal(signal.SIGINT)
    p.wait()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cprofile", type=str)
    parser.add_argument("--py_spy", type=str)
    parser.add_argument(
        "--no_control", dest="add_control", action="store_false"
    )
    args = parser.parse_args()

    plant_time_step = 0.0004
    t_sim = 1.0

    setup = build_sim(
        time_step=plant_time_step,
        add_control=args.add_control,
    )

    setup.simulator.Initialize()
    print("Simulate...")
    if args.cprofile is not None:
        with use_cprofile(args.cprofile):
            setup.simulator.AdvanceTo(t_sim)
    if args.py_spy is not None:
        with use_py_spy(args.py_spy):
            setup.simulator.AdvanceTo(t_sim)
    else:
        setup.simulator.AdvanceTo(t_sim)
    print("Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
