# (Export from Anzu)
from types import SimpleNamespace
import unittest

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem

from system_init.systems import (
    declare_simple_init,
    get_all_systems,
    simulator_initialize_repeatedly,
)


class PostRepeatedCheck(LeafSystem):
    def __init__(self):
        super().__init__()

        def on_init(context, init_state):
            init_state.counter += 1

        def post_repeated_init(context, init_state):
            init_state.repeated_and_finalized = True

        self.get_init_state = declare_simple_init(
            self,
            on_init,
            default=SimpleNamespace(
                counter=0,
                repeated_and_finalized=False,
            ),
            post_repeated_init=post_repeated_init,
        )


class Test(unittest.TestCase):
    def test_simple_init(self):
        builder = DiagramBuilder()
        check = builder.AddSystem(PostRepeatedCheck())
        diagram = builder.Build()

        all_systems = get_all_systems(diagram)
        self.assertEqual(
            all_systems,
            [
                diagram,
                check,
            ],
        )

        diagram_context = diagram.CreateDefaultContext()
        diagram_context_init = diagram_context.Clone()
        check_context = check.GetMyContextFromRoot(diagram_context)
        simulator = Simulator(diagram, diagram_context)

        initialize_count = 10

        check_init = check.get_init_state(check_context)

        self.assertEqual(check_init.counter, 0)
        self.assertFalse(check_init.repeated_and_finalized)

        post_count = simulator_initialize_repeatedly(
            simulator, count=initialize_count
        )
        self.assertEqual(post_count, 1)

        self.assertEqual(check_init.counter, initialize_count)
        self.assertTrue(check_init.repeated_and_finalized)

        # Reset.
        diagram_context.SetTimeStateAndParametersFrom(diagram_context_init)
        # Offset time.
        t_offset = 10.0
        diagram_context.SetTime(t_offset)

        old_check_init = check_init
        check_init = check.get_init_state(check_context)
        self.assertIsNot(check_init, old_check_init)

        self.assertEqual(check_init.counter, 0)
        self.assertFalse(check_init.repeated_and_finalized)

        post_count = simulator_initialize_repeatedly(
            simulator, count=initialize_count
        )
        self.assertEqual(post_count, 1)

        self.assertEqual(check_init.counter, initialize_count)
        self.assertTrue(check_init.repeated_and_finalized)


if __name__ == "__main__":
    unittest.main()
