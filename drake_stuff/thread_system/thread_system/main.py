"""
Python example of offloading discrete system work to a separate thread /
process.

See README for more info.
"""

import atexit
from contextlib import contextmanager
import copy
import dataclasses as dc
import functools
import multiprocessing as mp
import multiprocessing.dummy as mp_dummy
import threading
import time
import weakref

from pydrake.all import (
    AbstractValue,
    ApplySimulatorConfig,
    BasicVector,
    DiagramBuilder,
    LeafSystem,
    Simulator,
    SimulatorConfig,
    Value,
)

import pyinstrument
from thread_system.py_spy_lib import use_py_spy


_custom_atexit_queue = []


def custom_atexit_register(func):
    # N.B. multiprocessing / threading does not always seem to play happily
    # with `atexit`?
    _custom_atexit_queue.append(func)


def custom_atexit_dispatch():
    for func in _custom_atexit_queue:
        func()
    _custom_atexit_queue.clear()


# Just in case.
atexit.register(custom_atexit_dispatch)


class AbstractClock(LeafSystem):
    def __init__(self):
        super().__init__()

        def calc_t(context, output):
            t = context.get_time()
            output.set_value(t)

        self.DeclareAbstractOutputPort("t", Value[object], calc_t)


class PrintSystem(LeafSystem):
    def __init__(self, period_sec, *, prefix=""):
        super().__init__()

        self.DeclareAbstractInputPort("t", Value[object]())

        def on_discrete_update(context, raw_state):
            t = context.get_time()
            u = self.get_input_port().Eval(context)
            print(f"{prefix}t={t:.3g}, y={u:.3g}")

        # TODO: Make publish.
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec, 0.0, on_discrete_update
        )


def busy_sleep_until(t_next):
    # busy wait
    i = 1
    coeff = 1 + 1e-5
    while time.time() < t_next:
        i *= coeff


def busy_sleep(dt):
    t_next = time.time() + dt
    busy_sleep_until(t_next)


class BusyDiscreteSystem(LeafSystem):
    def __init__(self, period_sec=0.01, prefix="", do_print=True):
        super().__init__()

        self.u = self.DeclareAbstractInputPort("u", Value[object]())
        state_index = self.DeclareAbstractState(Value[object]())
        offset = 0.0

        def on_init(context, raw_state):
            u = self.u.Eval(context)
            x = u + offset
            if do_print:
                print(f"{prefix}init,   x={x:.3g}")
            abstract_state = raw_state.get_mutable_abstract_state(state_index)
            abstract_state.set_value(x)

        self.DeclareInitializationUnrestrictedUpdateEvent(on_init)

        def on_discrete_update(context, raw_state):
            u = self.u.Eval(context)
            x = u + offset

            # Simulate Python-based work (something that should lock up the
            # GIL).
            busy_sleep(period_sec)

            t = context.get_time()
            if do_print:
                print(f"{prefix}update, t={t:.3g}, x={u:.3g}")
            abstract_state = raw_state.get_mutable_abstract_state(state_index)
            abstract_state.set_value(x)

        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec, 0.0, on_discrete_update
        )

        def calc_y(context, output):
            x = context.get_abstract_state(state_index).get_value()
            output.set_value(x)

        self.DeclareAbstractOutputPort("y", Value[object], calc_y)


def declare_input_port(system, name, model):
    if isinstance(model, BasicVector):
        return system.DeclareVectorInputPort(name, model)
    else:
        assert isinstance(model, AbstractValue)
        return system.DeclareAbstractInputPort(name, model)


def declare_output_port(system, name, model, calc):
    if isinstance(model, BasicVector):
        return system.DeclareVectorOutputPort(name, model, calc)
    else:
        assert isinstance(model, AbstractValue)
        return system.DeclareAbstractOutputPort(name, model.Clone, calc)


def get_input_ports(system):
    out = []
    for i in range(system.num_input_ports()):
        out.append(system.get_input_port(i))
    return out


def get_output_ports(system):
    out = []
    for i in range(system.num_output_ports()):
        out.append(system.get_output_port(i))
    return out


@dc.dataclass
class Inputs:
    t_system: float
    system_input_values: dict
    is_init: bool = False


@dc.dataclass
class Outputs:
    system_output_values: dict
    t_system: float


def make_simulator(system):
    config = SimulatorConfig(
        integration_scheme="explicit_euler",
        max_step_size=0.1,
    )
    simulator = Simulator(system)
    ApplySimulatorConfig(config, simulator)
    return simulator


class SystemWorker:
    def __init__(self, system, *, make_simulator=make_simulator):
        self.system = system
        self.system_inputs = {
            x.get_name(): x for x in get_input_ports(system)
        }
        self.system_outputs = {
            x.get_name(): x for x in get_output_ports(system)
        }
        self.system_simulator = make_simulator(system)
        self.period_lag_max = None

    def start(self):
        raise NotImplemented()

    def stop(self):
        raise NotImplemented()

    def put_inputs(self, inputs):
        raise NotImplemented()

    def get_latest_outputs(self):
        raise NotImplemented()

    def maybe_get_latest_outputs(self):
        raise NotImplemented()

    def process(self, inputs):
        system_simulator = self.system_simulator
        system_context = system_simulator.get_context()
        for name, system_input in self.system_inputs.items():
            value = inputs.system_input_values[name]
            system_input.FixValue(system_context, value)
        if inputs.is_init:
            system_context.SetTime(inputs.t_system)
            system_simulator.Initialize()
        else:
            # TODO(eric.cousineau): How handle slowdowns / clock drift?
            # Perhaps step a few times and put items into queue?
            t = system_context.get_time()
            t_next = inputs.t_system

            if self.period_lag_max is not None:
                period_lag = t_next - t
                if period_lag > self.period_lag_max:
                    t_next = t + self.period_lag_max

            system_simulator.AdvanceTo(t_next)
            system_simulator.AdvancePendingEvents()
        system_output_values = {}
        for name, system_output in self.system_outputs.items():
            system_output_values[name] = system_output.Eval(system_context)
        outputs = Outputs(
            system_output_values=system_output_values,
            t_system=system_context.get_time(),
        )
        return outputs


class DirectWorker(SystemWorker):
    def __init__(self, system):
        super().__init__(system)
        self.inputs = None

    def start(self):
        pass

    def stop(self):
        pass

    def put_inputs(self, inputs):
        self.inputs = copy.deepcopy(inputs)

    def get_latest_outputs(self):
        return self.maybe_get_latest_outputs()

    def maybe_get_latest_outputs(self):
        outputs = self.process(self.inputs)
        outputs = copy.deepcopy(outputs)
        return outputs


class ThreadWorker(threading.Thread, SystemWorker):
    def __init__(self, system):
        threading.Thread.__init__(self)
        SystemWorker.__init__(self, system)
        self.lock = threading.Lock()
        self.should_stop = threading.Event()
        self.inputs = None
        self.outputs = None

    @staticmethod
    def _atexit(self_ref):
        self = self_ref()
        if self is not None:
            self.stop()

    def start(self):
        cleanup = functools.partial(ThreadWorker._atexit, weakref.ref(self))
        custom_atexit_register(cleanup)
        super().start()

    def stop(self):
        self.should_stop.set()
        self.join()

    def put_inputs(self, inputs):
        with self.lock:
            self.inputs = copy.deepcopy(inputs)

    def get_latest_outputs(self):
        while True:
            outputs = self.maybe_get_latest_outputs()
            if outputs is not None:
                return outputs
            else:
                time.sleep(1e-4)

    def maybe_get_latest_outputs(self):
        with self.lock:
            outputs = self.outputs
            self.outputs = None
        return outputs

    def run(self):
        while True:
            if self.should_stop.is_set():
                break
            with self.lock:
                inputs = copy.deepcopy(self.inputs)
                self.inputs = None
            if inputs is None:
                time.sleep(1e-4)
                continue
            outputs = self.process(inputs)
            with self.lock:
                self.outputs = copy.deepcopy(outputs)


# N.B. We must use `fork` so we can pass in an already constructed system.
# The alternative is to make systems pickleable, or add a pickleable factory
# method.
ctx = mp.get_context("fork")
# ctx = mp_dummy  # This should be about the same as the ThreadWorker version.


class MultiprocessWorker(ctx.Process, SystemWorker):
    def __init__(self, system):
        ctx.Process.__init__(self)
        SystemWorker.__init__(self, system)
        # This seems to make performance a ton easier.
        # See https://stackoverflow.com/a/56118981/7829525
        # self.manager = ctx.Manager()
        # self.should_stop = self.manager.Event()
        # self.inputs = self.manager.Queue()
        # self.outputs = self.manager.Queue()

        self.should_stop = ctx.Event()
        self.inputs = ctx.SimpleQueue()
        self.outputs = ctx.SimpleQueue()

    @staticmethod
    def _atexit(self_ref):
        self = self_ref()
        if self is not None:
            self.stop()

    def start(self):
        cleanup = functools.partial(
            MultiprocessWorker._atexit, weakref.ref(self)
        )
        custom_atexit_register(cleanup)
        ctx.Process.start(self)

    def stop(self):
        self.should_stop.set()
        self.join()

    def put_inputs(self, inputs):
        self.inputs.put(inputs)

    def get_latest_outputs(self):
        # Block on first.
        outputs = self.outputs.get()
        # Once flushed, try to consume more.
        latest = self.maybe_get_latest_outputs()
        if latest is not None:
            outputs = latest
        return outputs

    def maybe_get_latest_outputs(self):
        output = None
        while not self.outputs.empty():
            output = self.outputs.get()
        return output

    def run(self):
        while True:
            if self.should_stop.is_set():
                break
            inputs = None
            while not self.inputs.empty():
                # Drain the queue.
                inputs = self.inputs.get()
            if inputs is None:
                time.sleep(1e-6)
                continue
            outputs = self.process(inputs)
            self.outputs.put(outputs)


class WorkerSystem(LeafSystem):
    def __init__(
        self,
        worker,
        period_sec,
        deterministic=True,
    ):
        super().__init__()
        worker.start()
        system_inputs = worker.system_inputs
        system_outputs = worker.system_outputs

        self.inputs = {}
        self.outputs = {}

        # Undeclared state!
        self._system_input_values = {}
        self._system_output_values = {}
        self._has_discrete_update_inputs = False
        # hack
        self.t_final = 0.0

        for name, system_input in system_inputs.items():
            self.inputs[name] = declare_input_port(
                self, name, system_input.Allocate()
            )

        def calc_output(name, context, output):
            value = self._system_output_values[name]
            output.set_value(value)

        for name, system_output in system_outputs.items():
            calc_output_i = functools.partial(calc_output, name)
            self.outputs[name] = declare_output_port(
                self, name, system_output.Allocate(), calc_output_i
            )

        def put_inputs(context, *, is_init=False):
            for name, input in self.inputs.items():
                self._system_input_values[name] = input.Eval(context)
            inputs = Inputs(
                t_system=context.get_time(),
                system_input_values=self._system_input_values,
                is_init=is_init,
            )
            worker.put_inputs(inputs)

        def on_init(context, raw_state):
            put_inputs(context, is_init=True)
            self._system_output_values = worker.get_latest_outputs()
            self._has_discrete_update_inputs = False

        self.DeclareInitializationUnrestrictedUpdateEvent(on_init)

        def get_discrete_update_outputs(context):
            outputs = None
            if deterministic:
                if self._has_discrete_update_inputs:
                    outputs = worker.get_latest_outputs()
                    assert outputs is not None
            else:
                outputs = worker.maybe_get_latest_outputs()

            if outputs is not None:
                self._system_output_values = outputs.system_output_values
                self.t_final = outputs.t_system

        def on_discrete_update(context, raw_state):
            get_discrete_update_outputs(context)
            put_inputs(context)
            self._has_discrete_update_inputs = True

        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec, 0.0, on_discrete_update
        )
        # TODO(eric.cousineau): Publish events?


def main():
    # run(DirectWorker, num_systems=1, deterministic=True, do_print=True)
    # run(ThreadWorker, num_systems=1, deterministic=True, do_print=True)
    # run(MultiprocessWorker, num_systems=1, deterministic=True, do_print=True)

    # run(DirectWorker, num_systems=1, deterministic=True)
    # run(ThreadWorker, num_systems=1, deterministic=True)
    # run(ThreadWorker, num_systems=1, deterministic=False)
    # run(MultiprocessWorker, num_systems=1, deterministic=False)
    # run(MultiprocessWorker, num_systems=1, deterministic=False)

    # run(ThreadWorker, num_systems=3, deterministic=True)
    run(MultiprocessWorker, num_systems=5, deterministic=True)
    run(MultiprocessWorker, num_systems=5, deterministic=False)

    # run(DirectWorker, num_systems=5, deterministic=True)
    # run(ThreadWorker, num_systems=5, deterministic=True)
    # run(MultiprocessWorker, num_systems=5, deterministic=True)

    # run(ThreadWorker, num_systems=20, deterministic=False)
    # run(MultiprocessWorker, num_systems=5, deterministic=False)


@contextmanager
def use_pyinstrument(output_file):
    p = pyinstrument.Profiler()
    with p:
        yield
    r = pyinstrument.renderers.SpeedscopeRenderer()
    with open(output_file, "w") as f:
        f.write(p.output(r))


def run(
    worker_cls,
    *,
    num_systems,
    deterministic=True,
    do_print=False,
    verbose=False,
):
    print()
    print(
        f"{worker_cls.__name__}, num_systems={num_systems}, "
        f"deterministic={deterministic}"
    )

    builder = DiagramBuilder()

    clock = builder.AddSystem(AbstractClock())

    if do_print:
        period_sec = 0.1
        t_sim = period_sec * 4
    else:
        period_sec = 0.002
        t_sim = 1.0
    wrapper_period_sec = period_sec
    # wrapper_period_sec = period_sec / 10

    worker_systems = []
    for i in range(num_systems):
        busy_system = BusyDiscreteSystem(
            period_sec, prefix=f"[{i} busy ] ", do_print=False,
        )
        worker = worker_cls(busy_system)

        if not deterministic:
            # Hack :(
            worker.period_lag_max = period_sec * 2

        worker_system = builder.AddSystem(
            WorkerSystem(
                worker,
                period_sec=wrapper_period_sec,
                deterministic=deterministic,
            )
        )
        builder.Connect(
            clock.get_output_port(), worker_system.get_input_port()
        )
        if do_print:
            print_system = builder.AddSystem(
                PrintSystem(period_sec, prefix=f"[{i} print] ")
            )
            builder.Connect(
                worker_system.get_output_port(), print_system.get_input_port()
            )

        worker_systems.append(worker_system)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    t_sim_start = simulator.get_context().get_time()
    t_wall_start = time.time()

    simulator.AdvanceTo(t_sim)
    # with use_py_spy("./profile.svg"):
    # with use_pyinstrument("./speedscope.json"):
    #     simulator.AdvanceTo(t_sim)

    dt_sim = simulator.get_context().get_time() - t_sim_start
    dt_wall = time.time() - t_wall_start
    rate = dt_sim / dt_wall
    print(f"Rate: {rate:.3g}")

    dt_lag_final = simulator.get_context().get_time() - worker_system.t_final
    print(f"Final lag: {dt_lag_final:.3g}s")

    context = worker_system.GetMyContextFromRoot(simulator.get_context())
    y = worker_system.get_output_port().Eval(context)
    print(f"y: {y:.3g}")

    custom_atexit_dispatch()


if __name__ == "__main__":
    main()
