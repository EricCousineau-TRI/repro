import dataclasses as dc
from enum import Enum

import numpy as np

from pydrake.common.value import AbstractValue, Value
from pydrake.math import RigidTransform
from pydrake.multibody.math import SpatialAcceleration, SpatialVelocity
from pydrake.systems.framework import (
    DiagramBuilder,
    FixedInputPortValue,
    InputPort,
    LeafSystem,
    OutputPort,
)

from control_study.systems import maybe_attach_zoh


@dc.dataclass
class SpatialMotionInputPorts:
    X: InputPort
    V: InputPort
    A: InputPort

    def assert_valid(self):
        assert self.X is not None
        assert self.V is not None
        assert self.A is not None

    def eval(self, sys_context):
        self.assert_valid()
        return SpatialMotionInstant(
            X=self.X.Eval(sys_context),
            V=self.V.Eval(sys_context),
            A=self.A.Eval(sys_context),
        )


@dc.dataclass
class SpatialMotionOutputPorts:
    X: OutputPort
    V: OutputPort
    A: OutputPort

    def eval(self, sys_context):
        self.assert_valid()
        X = self.X.Eval(sys_context)
        V = self.V.Eval(sys_context)
        A = self.A.Eval(sys_context)
        return SpatialMotionInstant(X, V, A)


def connect_spatial_motion_ports(
    builder: DiagramBuilder,
    outputs: SpatialMotionOutputPorts,
    inputs: SpatialMotionInputPorts,
    *,
    zoh_dt: float = None,
):
    outputs.assert_valid()
    inputs.assert_valid()
    output_X = maybe_attach_zoh(builder, outputs.X, zoh_dt)
    builder.Connect(output_X, inputs.X)
    output_V = maybe_attach_zoh(builder, outputs.V, zoh_dt)
    builder.Connect(output_V, inputs.V)
    output_A = maybe_attach_zoh(builder, outputs.A, zoh_dt)
    builder.Connect(output_A, inputs.A)


@dc.dataclass
class SpatialMotionInstant:
    X: RigidTransform
    V: SpatialVelocity
    A: SpatialAcceleration

    def __iter__(self):
        as_tuple = (self.X, self.V, self.A)
        return iter(as_tuple)

    @staticmethod
    def make_dummy():
        return SpatialMotionInstant(
            X=RigidTransform(),
            V=SpatialVelocity.Zero(),
            A=SpatialAcceleration.Zero(),
        )


def declare_spatial_motion_inputs(system, frames, *, name_X, name_V, name_A):
    model_X = Value[RigidTransform]()
    model_V = Value[SpatialVelocity]()
    model_A = Value[SpatialAcceleration]()
    return SpatialMotionInputPorts(
        frames=frames,
        X=system.DeclareAbstractInputPort(name_X, model_X),
        V=system.DeclareAbstractInputPort(name_V, model_V),
        A=system.DeclareAbstractInputPort(name_A, model_A),
    )


def declare_spatial_motion_outputs(
    system,
    *,
    name_X,
    calc_X,
    name_V,
    calc_V,
    name_A,
    calc_A=None,
    prerequisites_of_calc=None,
):
    alloc_X = Value[RigidTransform]
    alloc_V = Value[SpatialVelocity]
    if prerequisites_of_calc is None:
        prerequisites_of_calc = {system.all_sources_ticket()}
    ports = SpatialMotionOutputPorts(
        X=system.DeclareAbstractOutputPort(
            name_X,
            alloc=alloc_X,
            calc=calc_X,
            prerequisites_of_calc=prerequisites_of_calc,
        ),
        V=system.DeclareAbstractOutputPort(
            name_V,
            alloc=alloc_V,
            calc=calc_V,
            prerequisites_of_calc=prerequisites_of_calc,
        ),
    )
    alloc_A = Value[SpatialAcceleration]
    ports.A = system.DeclareAbstractOutputPort(
        name_A,
        alloc=alloc_A,
        calc=calc_A,
        prerequisites_of_calc=prerequisites_of_calc,
    )
    return ports
