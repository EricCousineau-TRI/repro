from contextlib import contextmanager
from functools import partial
from textwrap import dedent
import unittest

import numpy as np

from pydrake.common.value import AbstractValue, Value
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import BasicVector_, BasicVector, DiagramBuilder

from drake_py_meta.function_system import (
    ContextTimeArg,
    FunctionSystem,
    MultiOutput,
    VectorArg,
    connect_printer,
    make_printer,
    value_to_annotation,
)
from drake_py_meta.primitive_systems import ConstantSource

DEBUG = False


def print_only() -> None:
    """Prints a constant statement.
    Output is explicitly annotated."""
    custom_print("print_only")


def print_time(t: ContextTimeArg) -> None:
    """Prints the context time.
    Output is explicitly annotated."""
    custom_print(f"t: {t:.2f}")


def print_vector(x: VectorArg(3)) -> None:
    """Prints an input vector.
    Output is explicitly annotated."""
    custom_print(f"x: {x}")


def print_more(t: ContextTimeArg, s: str) -> None:
    """Prints with multiple inupt arguments, one being abstract.
    Output is explicitly annotated."""
    custom_print(f"t: {t:.2f}, s: {s}")


def times_two(x: VectorArg(3)):
    """Example of a simple, pure function.
    Output is not annotated, thus inferred via evaluation."""
    return x*2


def scalar_times_three(z: float):
    """Example of pure function with a scalar.
    Output is not annotated, thus inferred via evaluation."""
    return z*3


def np_scalar_times_three(z: float):
    """Example of pure function with a NumPy scalar.
    Output is not annotated, thus inferred via evaluation."""
    return np.array(z*3)


def multi_output(
        t: ContextTimeArg,
        s_in: str,
        x_in: VectorArg(3),
        ) -> MultiOutput['s_out': str, 'x_out': VectorArg(3)]:
    """Example of multi-input, multi-output, pure function, mixing time,
    vectors, and abstract values.
    Output must be explicitly annotated."""
    s_out = f"t: {t:.2f}, s_in: {s_in}"
    x_out = t + times_two(x_in)
    return s_out, x_out


class TestFunctionSystem(unittest.TestCase):
    def check_diagram_print_capture(self, add_systems, expected_text, do_advance=True):  # noqa
        # TODO(eric.cousineau): Checking printing is a bit awkward, especially
        # this way. Find a simpler way to test, but one that's still evident in
        # how it functions.
        builder = DiagramBuilder()
        # Define some constants.
        # - Abstract (string).
        s_port = builder.AddSystem(
            ConstantSource("hello world")).get_output_port(0)
        # - Vector.
        x_port = builder.AddSystem(
            ConstantSource([1, 2, 3])).get_output_port(0)
        # - Vector of size 1, which could be interpreted as a scalar.
        z_port = builder.AddSystem(ConstantSource([10.])).get_output_port(0)
        dt = 0.1
        # Add test-specific systems.
        add_systems(builder, dt, x_port, s_port, z_port)
        # Build and simulate, capturing output.
        diagram = builder.Build()
        simulator = Simulator(diagram)
        t_end = dt
        with capture_custom_print_lines() as lines:
            custom_print("--- Initialize ---")
            simulator.Initialize()
            if do_advance:
                custom_print("--- AdvanceTo ---")
                simulator.AdvanceTo(t_end)
        # Check.
        actual_text = "\n".join(lines)
        if DEBUG:
            print()
            print(actual_text)
        self.assertEqual(actual_text, expected_text)

    def test_print_only(self):

        def add_systems(builder, dt, x_port, s_port, z_port):
            system = FunctionSystem(print_only)
            self.assertEqual(system.num_input_ports(), 0)
            self.assertEqual(system.num_output_ports(), 0)
            system.DeclarePeriodicPublish(dt)
            builder.AddSystem(system)

        self.check_diagram_print_capture(
            add_systems,
            dedent("""\
                --- Initialize ---
                print_only
                --- AdvanceTo ---
                print_only
            """.rstrip()),
        )

    def test_print_time(self):

        def add_systems(builder, dt, x_port, s_port, z_port):
            system = FunctionSystem(print_time)
            self.assertEqual(system.num_input_ports(), 0)
            self.assertEqual(system.num_output_ports(), 0)
            system.DeclarePeriodicPublish(dt)
            builder.AddSystem(system)

        self.check_diagram_print_capture(
            add_systems,
            dedent("""\
                --- Initialize ---
                t: 0.00
                --- AdvanceTo ---
                t: 0.10
            """.rstrip()),
        )

    def test_print_vector(self):

        def add_systems(builder, dt, x_port, s_port, z_port):
            system = FunctionSystem(print_vector)
            self.assertEqual(system.num_input_ports(), 1)
            self.assertEqual(system.num_output_ports(), 0)
            system.DeclarePeriodicPublish(dt)
            builder.AddSystem(system)
            builder.Connect(x_port, system.GetInputPort("x"))

        self.check_diagram_print_capture(
            add_systems,
            dedent("""\
                --- Initialize ---
                x: [ 1.  2.  3.]
                --- AdvanceTo ---
                x: [ 1.  2.  3.]
            """.rstrip()),
        )

    def test_print_more(self):

        def add_systems(builder, dt, x_port, s_port, z_port):
            system = FunctionSystem(print_more)
            self.assertEqual(system.num_input_ports(), 1)
            self.assertEqual(system.num_output_ports(), 0)
            system.DeclarePeriodicPublish(dt)
            builder.AddSystem(system)
            builder.Connect(s_port, system.GetInputPort("s"))

        self.check_diagram_print_capture(
            add_systems,
            dedent("""\
                --- Initialize ---
                t: 0.00, s: hello world
                --- AdvanceTo ---
                t: 0.10, s: hello world
            """.rstrip()),
        )

    def test_times_two(self):

        def add_systems(builder, dt, x_port, s_port, z_port):
            system = FunctionSystem(times_two)
            self.assertEqual(system.num_input_ports(), 1)
            self.assertEqual(system.num_output_ports(), 1)
            builder.AddSystem(system)
            builder.Connect(x_port, system.GetInputPort("x"))
            printer = builder.AddSystem(make_custom_printer(VectorArg(3), dt))
            builder.Connect(
                system.get_output_port(0), printer.get_input_port(0))

        self.check_diagram_print_capture(
            add_systems,
            dedent("""\
                --- Initialize ---
                [ 2.  4.  6.]
                --- AdvanceTo ---
                [ 2.  4.  6.]
            """.rstrip()),
        )

    def test_scalar_times_three(self):
        for times_three in (scalar_times_three, np_scalar_times_three):

            def add_systems(builder, dt, x_port, s_port, z_port):
                system = FunctionSystem(times_three)
                self.assertEqual(system.num_input_ports(), 1)
                self.assertEqual(system.num_output_ports(), 1)
                builder.AddSystem(system)
                builder.Connect(z_port, system.GetInputPort("z"))
                printer = builder.AddSystem(make_custom_printer(float, dt))
                builder.Connect(
                    system.get_output_port(0), printer.get_input_port(0))

            self.check_diagram_print_capture(
                add_systems,
                dedent("""\
                    --- Initialize ---
                    30.0
                    --- AdvanceTo ---
                    30.0
                """.rstrip()),
            )

    def test_multi_output(self):

        def add_systems(builder, dt, x_port, s_port, z_port):
            system = FunctionSystem(multi_output)
            self.assertEqual(system.num_input_ports(), 2)
            self.assertEqual(system.num_output_ports(), 2)
            builder.AddSystem(system)
            builder.Connect(s_port, system.GetInputPort("s_in"))
            builder.Connect(x_port, system.GetInputPort("x_in"))
            printer = builder.AddSystem(make_custom_printer(str, dt))
            builder.Connect(
                system.GetOutputPort("s_out"), printer.get_input_port(0))
            printer = builder.AddSystem(make_custom_printer(VectorArg(3), dt))
            builder.Connect(
                system.GetOutputPort("x_out"), printer.get_input_port(0))

        self.check_diagram_print_capture(
            add_systems,
            dedent("""\
                --- Initialize ---
                t: 0.00, s_in: hello world
                [ 2.  4.  6.]
                --- AdvanceTo ---
                t: 0.10, s_in: hello world
                [ 2.1  4.1  6.1]
            """.rstrip()),
        )

    def test_abstract_annotations(self):
        test_str = "s"
        test_vector = BasicVector([1., 2., 3.])

        def check(func, u):
            system = FunctionSystem(func)
            context = system.CreateDefaultContext()
            system.get_input_port(0).FixValue(context, u)
            return system.get_output_port(0).Eval(context)

        def explicit_abstract_input(value: Value[str]):
            self.assertIsInstance(value, AbstractValue)
            return value.get_value()

        self.assertEqual(test_str, check(explicit_abstract_input, test_str))

        def explicit_abstract_output(value: str) -> Value[str]:
            self.assertIsInstance(value, str)
            return AbstractValue.Make(value)

        self.assertEqual(test_str, check(explicit_abstract_output, test_str))

        def implicit_abstract_output(value: str):
            self.assertIsInstance(value, str)
            return AbstractValue.Make(value)

        self.assertEqual(test_str, check(implicit_abstract_output, test_str))

        def explicit_basic_vector(value: BasicVector(3)) -> BasicVector(3):
            self.assertIsInstance(value, BasicVector)
            return value

        np.testing.assert_equal(
            test_vector.get_value(),
            check(explicit_basic_vector, test_vector))

        def bad_basic_vector_cls(value: BasicVector) -> BasicVector:
            pass

        with self.assertRaises(AssertionError) as cm:
            check(bad_basic_vector_cls, test_vector)
        self.assertIn(
            "Must supply BasicVector_[] instance, not type",
            str(cm.exception))

        def bad_basic_vector_value_cls(value: Value[BasicVector]) -> float:
            return 0.

        with self.assertRaises(AssertionError) as cm:
            check(bad_basic_vector_value_cls, test_vector)
        self.assertIn(
            "Cannot specify Value[BasicVector_[]]",
            str(cm.exception))

    def test_annotation_repr(self):
        self.assertEqual(repr(ContextTimeArg), "ContextTimeArg")
        self.assertEqual(repr(VectorArg(3)), "VectorArg(3)")
        multi_output = MultiOutput['a': float, 'b': VectorArg(3)]
        self.assertEqual(
            repr(multi_output),
            "MultiOutput['a': float, 'b': VectorArg(3)]")

    def assert_annotations_equal(self, a, b):
        if isinstance(a, VectorArg):
            self.assertIsInstance(b, VectorArg)
            self.assertEqual(a._size, b._size)
        elif BasicVector_.is_instantiation(type(a)):
            self.assertTrue(BasicVector_.is_instantiation(type(b)))
            self.assertEqual(a.size(), b.size())
        else:
            self.assertIs(a, b)

    def test_value_to_annotation(self):
        test_vector = np.array([1., 2., 3.])
        test_basic_vector = BasicVector(test_vector)
        self.assert_annotations_equal(
            value_to_annotation(test_vector.tolist()), VectorArg(3))
        self.assert_annotations_equal(
            value_to_annotation(test_vector), VectorArg(3))
        self.assert_annotations_equal(
            value_to_annotation(test_basic_vector), test_basic_vector)
        self.assert_annotations_equal(
            value_to_annotation(AbstractValue.Make(test_basic_vector)),
            test_basic_vector)
        self.assert_annotations_equal(
            value_to_annotation(1.), float)
        self.assert_annotations_equal(
            value_to_annotation(np.array(1.)), float)
        self.assert_annotations_equal(
            value_to_annotation("hello"), str)
        self.assert_annotations_equal(
            value_to_annotation(AbstractValue.Make("hello")), Value[str])
        self.assert_annotations_equal(
            value_to_annotation(None), None)

    def test_multi_function(self):

        def add_systems(builder, dt, x_port, s_port, z_port):
            funcs = (scalar_times_three, print_only)
            system = FunctionSystem(funcs)
            system.DeclarePeriodicPublish(dt)
            self.assertEqual(system.num_input_ports(), 1)
            self.assertEqual(system.num_output_ports(), 1)
            builder.AddSystem(system)
            builder.Connect(z_port, system.GetInputPort("z"))
            printer = builder.AddSystem(make_custom_printer(float, dt))
            builder.Connect(
                system.GetOutputPort("output"), printer.get_input_port(0))

        self.check_diagram_print_capture(
            add_systems,
            dedent("""\
                --- Initialize ---
                print_only
                30.0
                --- AdvanceTo ---
                print_only
                30.0
            """.rstrip()),
        )

    def test_connect_printer(self):

        def add_systems(builder, dt, x_port, s_port, z_port):

            def local_print(name):
                return partial(mock_redirect_print, prefix=f"{name}: ")

            printer_s = connect_printer(
                builder, s_port, print=local_print("s"))
            printer_s.DeclarePeriodicPublish(dt)
            printer_s_direct = connect_printer(
                builder, s_port, print=local_print("s_direct"),
                use_direct_type=True)
            printer_s_direct.DeclarePeriodicPublish(dt)
            printer_x = connect_printer(
                builder, x_port, print=local_print("x"))
            printer_x.DeclarePeriodicPublish(dt)
            printer_x_direct = connect_printer(
                builder, x_port, print=local_print("x_direct"),
                use_direct_type=True)
            printer_x_direct.DeclarePeriodicPublish(dt)
            printer_z = connect_printer(
                builder, z_port, print=local_print("z"))
            printer_z.DeclarePeriodicPublish(dt)
            printer_z_direct = connect_printer(
                builder, z_port, print=local_print("z_direct"),
                use_direct_type=True)
            printer_z_direct.DeclarePeriodicPublish(dt)

        self.check_diagram_print_capture(
            add_systems,
            dedent("""\
                --- Initialize ---
                s: hello world
                s_direct: Value[str]('hello world')
                x: [ 1.  2.  3.]
                x_direct: BasicVector([1, 2, 3])
                z: [ 10.]
                z_direct: BasicVector([10])
            """.rstrip()),
            do_advance=False,
        )

    def test_bad_inferred_output(self):

        def bad_inferred_output():
            nonlocal called
            called = True

        called = False
        with self.assertRaises(AssertionError) as cm:
            FunctionSystem(bad_inferred_output)
        self.assertIn(
            "For function 'bad_inferred_output': 'None' is not a valid",
            str(cm.exception))
        self.assertTrue(called)

    def test_bad_function(self):

        def bad_function(good: int, bad, *worse, **worst):
            pass

        with self.assertRaises(RuntimeError) as cm:
            FunctionSystem(bad_function)
        self.assertIn("bad_function", str(cm.exception))
        self.assertIn("Needs annotation: bad", str(cm.exception))
        self.assertIn("Invalid kind: *worse", str(cm.exception))
        self.assertIn("Invalid kind: **worst", str(cm.exception))

    def test_bad_multi_output(self):
        with self.assertRaises(AssertionError) as cm:
            MultiOutput[None]
        self.assertEqual(
            "MultiOutput must be of non-zero length.",
            str(cm.exception))

        def bad_num_outputs() -> MultiOutput['s': str]:
            return 1, 2

        system = FunctionSystem(bad_num_outputs)
        context = system.CreateDefaultContext()
        with self.assertRaises(AssertionError) as cm:
            system.GetOutputPort("s").Eval(context)
        self.assertEqual(
            "Output must be tuple of length 1. Got length 2 instead.",
            str(cm.exception))

        def bad_output_type() -> MultiOutput['s': str]:
            return [1, 2]

        system = FunctionSystem(bad_output_type)
        context = system.CreateDefaultContext()
        with self.assertRaises(AssertionError) as cm:
            system.GetOutputPort("s").Eval(context)
        self.assertEqual(
            "Output type must be a tuple. Got <class 'list'> instead.",
            str(cm.exception))

    def test_scalar_as_vector_disabled(self):

        def expects_scalar(z: float) -> float:
            print(z)
            return z + 3

        builder = DiagramBuilder()
        system = builder.AddSystem(
            FunctionSystem(expects_scalar, scalar_as_vector=False))
        value = builder.AddSystem(ConstantSource([1.]))
        with self.assertRaises(RuntimeError) as cm:
            builder.Connect(value.get_output_port(0), system.GetInputPort("z"))
        self.assertIn(
            "Cannot mix vector-valued and abstract-valued",
            str(cm.exception))

    def test_bad_multi_func(self):
        funcs = (times_two, times_two)
        with self.assertRaises(RuntimeError) as cm:
            FunctionSystem(funcs)
        self.assertIn(
            "System FunctionSystem[times_two, times_two] already has an "
            "input port named x",
            str(cm.exception))


def custom_print(s, prefix=""):
    # This will be replaced during test.
    # This should not be called, as all examples tested using this function
    # will have their output return types annotated.
    print(f"ESCAPED PRINT: {prefix}{s}")
    assert False


def mock_redirect_print(s, prefix=""):
    # See `make_custom_printer`.
    custom_print(s, prefix=prefix)


def make_custom_printer(cls, dt):
    # Use mock_redirect_print so we can still mock `custom_print` via
    # `capture_custom_print_lines`.
    printer = make_printer(cls, print=mock_redirect_print)
    printer.DeclarePeriodicPublish(dt)
    return printer


@contextmanager
def capture_custom_print_lines():
    """Permits a scoped capture of calls to `custom_print` within this
    module."""
    global custom_print
    lines = []

    def fake(obj, prefix=""):
        if isinstance(obj, BasicVector):
            line = f"BasicVector({obj})"
        elif isinstance(obj, AbstractValue):
            line = f"{type(obj).__name__}({repr(obj.get_value())})"
        else:
            line = str(obj)
        lines.append(f"{prefix}{line}")

    try:
        old_print = custom_print
        custom_print = fake
        yield lines
    finally:
        custom_print = old_print


if __name__ == "__main__":
    unittest.main()
