"""
Generic system helpers.
"""

import copy
import pickle
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


def declare_input_port(system, name, model):
    if isinstance(model, (BasicVector, int)):
        return system.DeclareVectorInputPort(name, model)
    else:
        assert isinstance(model, AbstractValue)
        return system.DeclareAbstractInputPort(name, model)


def declare_output_port(system, name, model, calc):
    if isinstance(model, (BasicVector, int)):
        system.DeclareVectorOutputPort(name, model, calc)
    else:
        assert isinstance(model, AbstractValue)
        system.DeclareAbstractOutputPort(name, model.Clone, calc)


def reset_dict(*, dest, src, do_copy=True):
    """Resets dict in-place."""
    for k in list(dest.keys()):
        del dest[k]
    for k, v in src.items():
        if do_copy:
            v = copy.deepcopy(v)
        dest[k] = v


def reset_simple_namespace(*, dest, src, do_copy=True):
    reset_dict(dest=dest.__dict__, src=src.__dict__, do_copy=do_copy)


def declare_simple_cache(
    sys, calc, *, description="cache", default=None, return_reset=False
):
    if default is None:
        default = SimpleNamespace()
    cache_model_value = Value[object](default)

    def calc_wrapped(context, abstract_value):
        cache = abstract_value.get_mutable_value()
        return calc(context, cache=cache)

    cache_entry = sys.DeclareCacheEntry(
        description=description,
        value_producer=ValueProducer(
            allocate=cache_model_value.Clone,
            calc=calc_wrapped,
        ),
    )

    def reset_cache(context):
        # Reset cache to default.
        # TODO(eric.cousineau): Bind CacheEntryValue.mark_out_of_date() instead
        # of manually recomputing?
        cache = cache_entry.Eval(context)
        reset_simple_namespace(dest=cache, src=default)
        calc(context, cache=cache)

    if return_reset:
        return cache_entry, reset_cache
    else:
        return cache_entry


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

    Warning:
        If using with a simulated plant, you may want to use
        `clean_plant_state`.
    """
    assert count >= 1
    monitor = simulator.get_monitor()
    simulator.clear_monitor()
    for i in range(count - 1):
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
    if monitor is not None:
        simulator.set_monitor(monitor)
    # Do one final initialization.
    simulator.Initialize()
    return post_count


def declare_simple_init(
    sys,
    on_init=None,
    *,
    default=None,
    post_init_warmup=None,
    description="init",
    setter=False,
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

    def get_init_state(context):
        return context.get_abstract_state(init_state_index).get_value()

    def set_init_state(context, init):
        abstract_state = context.get_mutable_abstract_state(init_state_index)
        abstract_state.set_value(init)

    def update(context, raw_state):
        abstract_state = raw_state.get_mutable_abstract_state(init_state_index)
        init_state = abstract_state.get_mutable_value()
        on_init(context, init_state)

    if on_init is not None:
        sys.DeclareInitializationUnrestrictedUpdateEvent(update=update)

    if post_init_warmup is not None:

        def post_repeated_init_wrapped(context):
            raw_state = context.get_mutable_abstract_state()
            abstract_state = raw_state.get_mutable_value(init_state_index)
            init_state = abstract_state.get_mutable_value()
            post_init_warmup(context, init_state)

        sys._post_repeated_init_wrapped = post_repeated_init_wrapped

    if setter:
        return get_init_state, set_init_state
    else:
        return get_init_state

def ensure_finite_abstract_value(model_value):
    spatial_types = (SpatialVelocity, SpatialForce, SpatialAcceleration)
    if isinstance(model_value.get_value(), spatial_types):
        # Ensure we zero out the model value (rather than use NaNs).
        if np.any(~np.isfinite(model_value.get_value().get_coeffs())):
            model_value = model_value.Clone()
            model_value.get_mutable_value().SetZero()
    return model_value


def attach_zoh(builder, output_port, dt):
    assert dt > 0
    if output_port.get_data_type() == PortDataType.kAbstractValued:
        model_value = output_port.Allocate()
        model_value = ensure_finite_abstract_value(model_value)
        zoh = ModifiedZoh(dt, model_value)
    else:
        zoh = ModifiedZoh(dt, output_port.size())
    builder.AddSystem(zoh)
    builder.Connect(output_port, zoh.get_input_port())
    parent_system_name = output_port.get_system().get_name()
    parent_name = output_port.get_name()
    zoh.set_name(f"{parent_system_name}.{parent_name}.modified_zoh")
    return zoh.get_output_port()


def maybe_attach_zoh(builder, output_port, dt):
    if dt is None:
        return output_port
    return attach_zoh(builder, output_port, dt)


class ModifiedZoh(LeafSystem):
    """
    Modified zero-order hold that adds an initialization event.
    """

    # TODO(eric.cousineau): Remove this once ZeroOrderHold has this affordance,
    # e.g. https://github.com/RobotLocomotion/drake/pull/18356

    def __init__(self, period_sec, value):
        super().__init__()
        is_abstract = isinstance(value, AbstractValue)
        if not is_abstract:
            vector_size = value
            self.DeclareVectorInputPort("u", BasicVector(vector_size))
            state_index = self.DeclareDiscreteState(vector_size)

            def update(context, discrete_state):
                input = self.get_input_port().Eval(context)
                discrete_state.set_value(0, input)

            self.DeclareInitializationDiscreteUpdateEvent(update)
            self.DeclarePeriodicDiscreteUpdateEvent(
                period_sec,
                0.0,
                update,
            )
            self.DeclareStateOutputPort("y", state_index)
        else:
            abstract_model_value = value
            self.DeclareAbstractInputPort("u", abstract_model_value)
            state_index = self.DeclareAbstractState(abstract_model_value)

            def update(context, raw_state):
                abstract_state = raw_state.get_mutable_abstract_state()
                abstract_value = abstract_state.get_mutable_value(0)
                input = self.get_input_port().EvalAbstract(context)
                abstract_value.SetFrom(input)

            self.DeclareInitializationUnrestrictedUpdateEvent(update)
            self.DeclarePeriodicUnrestrictedUpdateEvent(
                period_sec, 0.0, update
            )
            self.DeclareStateOutputPort("y", state_index)


def make_discrete_update(update, trigger_type=TriggerType.kPeriodic):
    def wrapped(system, context, event, xd):
        update(context, xd)

    return DiscreteUpdateEvent(trigger_type, wrapped)
