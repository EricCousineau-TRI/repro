from collections import OrderedDict
from contextlib import contextmanager
import inspect
from functools import partial
from textwrap import indent

import numpy as np

from pydrake.common.value import AbstractValue
from pydrake.systems.analysis import Simulator
from pydrake.systems.primitives import (
    ConstantVectorSource_, ConstantValueSource,
)
from pydrake.systems.framework import (
    DiagramBuilder, LeafSystem, Context, PortDataType, BasicVector,
)


def _as_abstract_value(annotation):
    if annotation is ContextTime:
        return annotation
    elif isinstance(annotation, type):
        value = annotation()
    else:
        value = annotation
    # Resolve.
    if isinstance(value, AbstractValue):
        return value
    else:
        return AbstractValue.Make(value)


def _example_value(model):
    if model is ContextTime:
        return float()
    else:
        return model.get_value()


def _is_vector(x):
    x = np.asarray(x)
    return (x.ndim in (1, 2) and x.size in x.shape)


def is_array(x):
    return isinstance(x, (np.ndarray, list))


class PortType:
    def __init__(self, example):
        if is_array(example):
            self.type = PortDataType.kVectorValued
            assert _is_vector(example), (f"Must be vector: {example}")
            self.model = BasicVector(example)
        else:
            self.type = PortDataType.kAbstractValued
            self.model = _as_abstract_value(example)

    @staticmethod
    def declare_input_port(system, name, port_type):
        # DeclareInputPort does not work with kAbstractValued?
        if port_type.type == PortDataType.kAbstractValued:
            return system.DeclareAbstractInputPort(
                name=name, model_value=port_type.model)
        else:
            return system.DeclareVectorInputPort(
                name=name, model_vector=port_type.model)

    @staticmethod
    def declare_output_port(system, name, port_type, **kwargs):
        if port_type.type == PortDataType.kAbstractValued:
            return system.DeclareAbstractOutputPort(
                name=name, alloc=port_type.model.Clone, **kwargs)
        else:
            return system.DeclareVectorOutputPort(
                name=name, model_value=port_type.model, **kwargs)


def _get_input_port_types(func):
    sig = inspect.signature(func)
    accepted_kinds = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    errors = []
    input_port_types = OrderedDict()
    for name, param in sig.parameters.items():
        if param.kind not in accepted_kinds:
            errors.append(f"Invalid kind: {param}")
            continue
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            errors.append(f"Needs annotation: {param}")
            continue
        input_port_types[name] = PortType(annotation)
    if errors:
        # TODO(eric): Print func source, e.g. `{file}:{func}:{line}`?
        raise RuntimeError(
            f"Errors for {func}:\n" +
            indent("\n".join(errors), "  "))
    return input_port_types


_unable_to_convert = object()


def _as_ordered_dict(x):
    if isinstance(x, OrderedDict):
        return x
    elif isinstance(x, (tuple, list)):
        return OrderedDict(x)
    else:
        return _unable_to_convert


def _as_output_port_types(output):
    output_example_kwargs = _as_ordered_dict(output)
    assert output_example_kwargs is not _unable_to_convert, (
        f"Must return an OrderedDict for outputs.")
    output_port_type = OrderedDict(
        (name, PortType(value))
        for name, value in output_example_kwargs)
    return output_port_type


# Fake annotation that implies a vector of a given size.
def VectorXd(size):
    return np.zeros(size)


# Annotation that implies time from the context should be used.
ContextTime = object()


class MultiOutput:
    def __init__(self, *items):
        self.output_types = _as_ordered_dict(items)
        assert self.output_types is not _unable_to_convert


class PyFunctionBaseSystem(LeafSystem):
    """Base Class."""

    def __init__(self, func, unsqueeze_scalar=False, set_name=True):
        LeafSystem.__init__(self)
        cls = type(self)
        if set_name:
            self.set_name(f"{cls.__name__}[{func.__name__}]")
        # TODO(eric.cousineau): Set the name?
        # TODO(eric.cousineau): Have this return a class, so it's easy to
        # override the name?
        assert unsqueeze_scalar == False
        self._declare_input_ports(func)
        self._declare_computation(func)

    def _declare_input_ports(self, func):
        input_port_types = _get_input_port_types(func)
        self._input_evals = dict()
        self._input_examples = dict()
        for name, port_type in input_port_types.items():
            if port_type.model is ContextTime:
                input_eval = Context.get_time
            else:
                input_port = PortType.declare_input_port(self, name, port_type)
                input_eval = input_port.Eval
            self._input_examples[name] = _example_value(port_type.model)
            self._input_evals[name] = input_eval

    def _declare_computation(self, func):
        raise NotImplemented()

    def _get_input_kwargs(self, context):
        kwargs = dict()
        for name, input_eval in self._input_evals.items():
            kwargs[name] = input_eval(context)
        return kwargs

    def _get_input_kwargs_example(self):
        kwargs = dict()
        for name, example in self._input_examples.items():
            kwargs[name] = example
        return kwargs


def _set_value(container, value):
    if isinstance(container, BasicVector):
        container.SetFromVector(value)
    else:
        container.set_value(value)


class PyFunctionSystem(PyFunctionBaseSystem):
    """Generic function system."""
    def _declare_computation(self, func):
        output = inspect.signature(func).return_annotation

        def call_func(context):
            kwargs = self._get_input_kwargs(context)
            return func(**kwargs)

        if isinstance(output, MultiOutput):
            # Multi-output (as tuple).
            item_iter = enumerate(output.output_types.items())

            def calc_output_i(context, output_i, i):
                # TODO(eric.cousineau): Use caching to consolidate output?
                value = call_func(context)
                _set_value(output_i, value[i])

            for i, (name, output_type) in item_iter:
                port_type = PortType(output_type)
                PortType.declare_output_port(
                    self, name, port_type,
                    calc=partial(calc_output_i, i=i))

        else:
            # Single output.
            if output is inspect.Parameter.empty:
                # Infer via evaluation.
                output = func(**self._get_input_kwargs_example())

            def calc_output(context, output):
                _set_value(output, call_func(context))

            port_type = PortType(output)
            PortType.declare_output_port(
                self, "output", port_type, calc=calc_output)


def ConstantSource(value):
    """Creates a ConstantVectorSource or ConstantValueSource."""
    if is_array(value):
        return ConstantVectorSource_(value)
    else:
        return ConstantValueSource(AbstractValue.Make(value))


class PyFunctionPublishSystem(PyFunctionBaseSystem):
    """Creates a system that publishes for a given ``func``. You must
    explicitly declare your publish period."""
    def _declare_computation(self, func):
        self._func = func

    def DoPublish(self, context, events):
        # Call base method to ensure we do not get recursion.
        LeafSystem.DoPublish(self, context, events)
        input_kwargs = self._get_input_kwargs(context)
        self._func(**input_kwargs)


@contextmanager
def fake_print():
    lines = []

    def fake(line):
        lines.append(str(line))

    try:
        global print
        print = fake
        yield lines
    finally:
        print = __builtins__.print


def check_output(add_test_systems):
    builder = DiagramBuilder()

    def add_printer(cls):

        def print_value(value: cls):
            print(value)

        printer = builder.AddSystem(
            PyFunctionPublishSystem(print_value, set_name=False))
        printer.DeclarePeriodicPublish(dt)
        return printer

    # Define some constants.
    s_port = builder.AddSystem(
        ConstantSource("hello world")).get_output_port(0)
    x_port = builder.AddSystem(
        ConstantSource([1, 2, 3])).get_output_port(0)
    dt = 0.1
    add_test_systems(builder, dt, x_port, s_port, add_printer)
    diagram = builder.Build()
    simulator = Simulator(diagram)
    t_end = dt
    with fake_print() as output:
        simulator.Initialize()
        simulator.AdvanceTo(t_end)
    return output


def print_only():
    print("print_only")

def print_time(t: ContextTime):
    print(f"t: {t:.2f}")

def print_vector(x: VectorXd(3)):
    print(f"x: {x}")

def print_more(t: ContextTime, s: str):
    print(f"t: {t:.2f}, s: {s}")

def times_two(x: VectorXd(3)):
    return x*2

def multi_output(
        t: ContextTime,
        s_in: str,
        x_in: VectorXd(3),
        ) -> MultiOutput(('s_out', str), ('x_out', VectorXd(3))):
    s_out = f"t: {t:.2f}, s_in: {s_in}"
    x_out = t + times_two(x_in)
    return s_out, x_out


def main():

    def check(builder, dt, x_port, s_port, add_printer):
        system = PyFunctionPublishSystem(print_only)
        system.DeclarePeriodicPublish(dt)
        builder.AddSystem(system)

    assert check_output(check) == [
        "print_only",
        "print_only",
    ]

    def check(builder, dt, x_port, s_port, add_printer):
        system = PyFunctionPublishSystem(print_time)
        system.DeclarePeriodicPublish(dt)
        builder.AddSystem(system)

    assert check_output(check) == [
        "t: 0.00",
        "t: 0.10",
    ]

    def check(builder, dt, x_port, s_port, add_printer):
        system = PyFunctionPublishSystem(print_vector)
        system.DeclarePeriodicPublish(dt)
        builder.AddSystem(system)
        builder.Connect(x_port, system.GetInputPort("x"))

    assert check_output(check) == [
        "x: [ 1.  2.  3.]",
        "x: [ 1.  2.  3.]",
    ]

    def check(builder, dt, x_port, s_port, add_printer):
        system = PyFunctionPublishSystem(print_more)
        system.DeclarePeriodicPublish(dt)
        builder.AddSystem(system)
        builder.Connect(s_port, system.GetInputPort("s"))

    assert check_output(check) == [
        "t: 0.00, s: hello world",
        "t: 0.10, s: hello world",
    ]

    def check(builder, dt, x_port, s_port, add_printer):
        system = PyFunctionSystem(times_two)
        builder.AddSystem(system)
        builder.Connect(x_port, system.GetInputPort("x"))
        printer = add_printer(VectorXd(3))
        builder.Connect(system.get_output_port(0), printer.get_input_port(0))

    assert check_output(check) == [
        "[ 2.  4.  6.]",
        "[ 2.  4.  6.]",
    ]

    def check(builder, dt, x_port, s_port, add_printer):
        system = PyFunctionSystem(multi_output)
        builder.AddSystem(system)
        builder.Connect(s_port, system.GetInputPort("s_in"))
        builder.Connect(x_port, system.GetInputPort("x_in"))
        printer = add_printer(str)
        builder.Connect(
            system.GetOutputPort("s_out"), printer.get_input_port(0))
        printer = add_printer(VectorXd(3))
        builder.Connect(
            system.GetOutputPort("x_out"), printer.get_input_port(0))

    assert check_output(check) == [
        "t: 0.00, s_in: hello world",
        "[ 2.  4.  6.]",
        "t: 0.10, s_in: hello world",
        "[ 2.1  4.1  6.1]",
    ]


if __name__ == "__main__":
    main()
