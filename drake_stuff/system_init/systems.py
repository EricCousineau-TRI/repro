"""
Generic system helpers.

(Export from Anzu)
"""

from types import SimpleNamespace

import numpy as np

from pydrake.common.value import AbstractValue, Value
from pydrake.multibody.math import (
    SpatialAcceleration,
    SpatialForce,
    SpatialVelocity,
)
from pydrake.systems.framework import (
    BasicVector,
    Diagram,
    DiagramBuilder,
    DiscreteUpdateEvent,
    LeafSystem,
    PortDataType,
    TriggerType,
    ValueProducer,
)


def get_all_systems(system):
    # Should use SystemVisitor pattern. See:
    # https://github.com/RobotLocomotion/drake/issues/18602
    systems = []

    def recurse(system):
        systems.append(system)
        if isinstance(system, Diagram):
            for sub_system in system.GetSystems():
                recurse(sub_system)

    recurse(system)
    return systems


def simulator_initialize_repeatedly(simulator, *, count=10):
    """
    Tries to ensure simultaneous initialization events converge to a stable
    value.

    Useful when there are discrete systems (e.g. ZOH, etc.) that are cascaded.

    For more info, see:
    https://github.com/RobotLocomotion/drake/pull/18551#issuecomment-1384319905
    """
    for i in range(count):
        simulator.Initialize()
    # Do a custom "hack" event to notify that we're done w/ repeated
    # initialization.
    diagram_context = simulator.get_mutable_context()
    post_count = 0
    for system in get_all_systems(simulator.get_system()):
        post_repeated_init_wrapped = getattr(
            system, "_post_repeated_init_wrapped", None
        )
        if post_repeated_init_wrapped is not None:
            context = system.GetMyMutableContextFromRoot(diagram_context)
            # TODO(eric.cousineau): Rather than doing this in isolation,
            # consider factoring it out? Otherwise, cache invalidation happens.
            post_repeated_init_wrapped(context)
            post_count += 1
    return post_count


def declare_simple_init(
    sys,
    on_init,
    *,
    default=None,
    post_repeated_init=None,
    description="init",
):
    """
    Declares a simple discrete initialization event.

    Note:
        Be sure to use this in conjunction with
        simulator_initialize_repeatedly().
    """
    if default is None:
        default = SimpleNamespace()
    init_state_index = sys.DeclareAbstractState(Value[object](default))

    def update(context, raw_state):
        abstract_state = raw_state.get_mutable_abstract_state(init_state_index)
        init_state = abstract_state.get_mutable_value()
        on_init(context, init_state)

    def eval_init_state(context):
        return context.get_abstract_state(init_state_index).get_value()

    sys.DeclareInitializationUnrestrictedUpdateEvent(update=update)

    if post_repeated_init is not None:

        def post_repeated_init_wrapped(context):
            raw_state = context.get_mutable_abstract_state()
            abstract_state = raw_state.get_mutable_value(init_state_index)
            init_state = abstract_state.get_mutable_value()
            post_repeated_init(context, init_state)

        sys._post_repeated_init_wrapped = post_repeated_init_wrapped

    return eval_init_state
