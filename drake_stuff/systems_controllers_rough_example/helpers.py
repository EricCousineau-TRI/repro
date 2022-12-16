"""
A smattering of unorganized helper funcs.
"""

from types import SimpleNamespace

import numpy as np

from pydrake.common.value import Value
from pydrake.systems.framework import ValueProducer


def np_print_more_like_matlab():
    np.set_printoptions(
        formatter={
            "object": lambda x: f"{x: 06.4f}",
            "float_kind": lambda x: f"{x: 06.4f}",
        },
        linewidth=150,
    )


def dict_inverse(d):
    return {v: k for k, v in d.items()}


def trace_to_output(diagram_or_builder, input_port):
    # see: https://stackoverflow.com/a/74802395/7829525
    system = input_port.get_system()
    input_locator = (system, input_port.get_index())
    connection_map = diagram_or_builder.connection_map()
    connection_map = dict_inverse(connection_map)
    output_system, output_index = connection_map[input_locator]
    output_port = output_system.get_output_port(output_index)
    return output_port


def simple_cache_declare(sys, calc, *, description="cache", default=None):
    """
    Declares cache. Can optionally use with `simple_cache_ensure_init`.
    """
    if default is None:
        default = SimpleNamespace()
    assert not hasattr(default, "init")
    cache_model_value = Value[object](default)

    def calc_wrapped(context, abstract_value):
        cache = abstract_value.get_mutable_value()
        return calc(context, cache=cache)

    return sys.DeclareCacheEntry(
        description=description,
        value_producer=ValueProducer(
            allocate=cache_model_value.Clone,
            calc=calc_wrapped,
        ),
    )


def simple_cache_ensure_init(context, cache, on_init):
    """
    Adds a very hacky mechanism to hack initialization in a somewhat modular way.
    Should be called every time cache is eval'd. See test for example.

    Blech, this is uber hacky. Rip this out pending the correct initialization
    mechanism. https://github.com/RobotLocomotion/drake/issues/12649
    """
    init = getattr(cache, "init", None)
    if hasattr(cache, "prev_time"):
        possible_reset = cache.prev_time > context.get_time()
    else:
        assert init is None
        possible_reset = False
    if init is None or possible_reset:
        # Reset!
        cache.init = SimpleNamespace()
        on_init(context, cache)
    cache.prev_time = context.get_time()


def maybe_eval_port(port, context):
    if port.HasValue(context):
        return port.Eval(context)
    else:
        return None


def fix_port(port, parent_context, value):
    context = port.get_system().GetMyContextFromRoot(parent_context)
    return port.FixValue(context, value)


def eval_port(port, parent_context):
    context = port.get_system().GetMyContextFromRoot(parent_context)
    return port.Eval(context)
