"""
Shows example of cache stuff working (hopefully).
"""

import time
from types import SimpleNamespace
import unittest

from pydrake.common.value import Value
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.systems.primitives import ZeroOrderHold

from systems_controllers_rough_example.helpers import (
    np_print_more_like_matlab,
    simple_cache_declare,
    simple_cache_ensure_init,
)


class SimpleCacheSystem(LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareAbstractInputPort("input", Value[object]())

        def on_init(context, cache):
            # Record testing information.
            cache.on_init_at.append(context.get_time())
            cache.init.wall_time = time.time()
            # Latch input.
            cache.init.input = self.get_input_port().Eval(context)
            # Initialize.
            cache.value = 0

        def calc_cache(context, cache):
            simple_cache_ensure_init(context, cache, on_init)
            cache.value += 1

        # Normally you should not expose this.
        self.cache_entry = simple_cache_declare(
            self, calc_cache, default=SimpleNamespace(on_init_at=[])
        )

        def calc_value(context, output):
            cache = self.cache_entry.Eval(context)
            output.set_value(cache.value)

        self.DeclareAbstractOutputPort(
            "value", alloc=Value[object], calc=calc_value
        )


class Test(unittest.TestCase):
    def test_simple_cache(self):
        period_sec = 1.0
        builder = DiagramBuilder()
        simple = builder.AddSystem(SimpleCacheSystem())
        # Add ZOH to poll input.
        zoh = builder.AddSystem(ZeroOrderHold(period_sec, Value[object]()))
        builder.Connect(simple.get_output_port(), zoh.get_input_port())
        diagram = builder.Build()
        diagram_context = diagram.CreateDefaultContext()
        diagram_context_init = diagram_context.Clone()
        simple_context = simple.GetMyContextFromRoot(diagram_context)
        simulator = Simulator(diagram, diagram_context)

        simple.get_input_port().FixValue(simple_context, Value[object]("a"))
        t_first = time.time()
        simulator.Initialize()
        simulator.AdvanceTo(2 * period_sec)
        cache = simple.cache_entry.Eval(simple_context)
        self.assertEqual(cache.on_init_at, [0.0])
        self.assertGreater(cache.init.wall_time, t_first)
        self.assertEqual(cache.init.input, "a")
        self.assertEqual(simple.get_output_port().Eval(simple_context), 3.0)

        # Reset.
        diagram_context.SetTimeStateAndParametersFrom(diagram_context_init)

        simple.get_input_port().FixValue(simple_context, Value[object]("b"))
        t_second = time.time()
        simulator.Initialize()
        simulator.AdvanceTo(2 * period_sec)

        # Same cache object (with persistence).
        self.assertIs(cache, simple.cache_entry.Eval(simple_context))
        # Initialized again.
        self.assertEqual(cache.on_init_at, [0.0, 0.0])
        # - However, reinitialized.
        self.assertGreater(cache.init.wall_time, t_second)
        self.assertEqual(cache.init.input, "b")
        # But same value.
        self.assertEqual(simple.get_output_port().Eval(simple_context), 3.0)


if __name__ == "__main__":
    np_print_more_like_matlab()
    unittest.main()
