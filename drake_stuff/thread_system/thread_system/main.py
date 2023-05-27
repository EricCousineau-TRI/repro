import atexit
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
    BasicVector,
    DiagramBuilder,
    LeafSystem,
    Simulator,
    Value,
)


class AbstractClock(LeafSystem):
    def __init__(self):
        super().__init__()

        def calc_t(context, output):
            t = context.get_time()
            output.set_value(t)

        self.DeclareAbstractOutputPort("t", Value[object], calc_t)


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

        def on_discrete_update(context, raw_state):
            u = self.u.Eval(context)

            # Simulate Python-based work (something that should lock up the
            # GIL).
            busy_sleep(period_sec)

            t = context.get_time()
            if do_print:
                print(f"{prefix}t={t:.3g}, u={u:.3g}")
            abstract_state = raw_state.get_mutable_abstract_state(state_index)
            abstract_state.set_value(u)

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


_custom_atexit_queue = []


def custom_atexit_register(func):
    _custom_atexit_queue.append(func)


def custom_atexit_dispatch():
    for func in _custom_atexit_queue:
        func()


@dc.dataclass
class Inputs:
    system_t: float
    system_input_values: dict
    is_init: bool = False


@dc.dataclass
class Outputs:
    system_output_values: dict


class SystemWorker:
    def __init__(self, system):
        self.system = system
        self.system_inputs = {
            x.get_name(): x for x in get_input_ports(system)
        }
        self.system_outputs = {
            x.get_name(): x for x in get_output_ports(system)
        }
        self.system_simulator = Simulator(system)

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
            system_context.SetTime(inputs.system_t)
            system_simulator.Initialize()
        else:
            # TODO(eric.cousineau): How handle slowdowns / clock drift?
            # Perhaps step a few times and put items into queue?
            system_simulator.AdvanceTo(inputs.system_t)
        system_output_values = {}
        for name, system_output in self.system_outputs.items():
            system_output_values[name] = system_output.Eval(system_context)
        return system_output_values


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
                time.sleep(1e-6)

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
                time.sleep(1e-6)
                continue
            outputs = self.process(inputs)
            with self.lock:
                self.outputs = copy.deepcopy(outputs)


ctx = mp.get_context("fork")
# ctx = mp_dummy


class MpWorker(ctx.Process, SystemWorker):
    def __init__(self, system):
        ctx.Process.__init__(self)
        SystemWorker.__init__(self, system)
        # This seems to make performance a ton easier.
        # See https://stackoverflow.com/a/56118981/7829525
        self.manager = ctx.Manager()
        self.should_stop = self.manager.Event()
        self.inputs = self.manager.Queue()
        self.outputs = self.manager.Queue()

    @staticmethod
    def _atexit(self_ref):
        self = self_ref()
        if self is not None:
            self.stop()

    def start(self):
        cleanup = functools.partial(MpWorker._atexit, weakref.ref(self))
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
            system_output_values = self.process(inputs)
            self.outputs.put(system_output_values)


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

        # Undeclared state!
        self._system_input_values = {}
        self._system_output_values = {}

        self.inputs = {}
        self.outputs = {}

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

        def make_inputs_value(context):
            for name, input in self.inputs.items():
                self._system_input_values[name] = input.Eval(context)
            return Inputs(
                system_t=context.get_time(),
                system_input_values=self._system_input_values,
            )

        def on_init(context, raw_state):
            inputs = make_inputs_value(context)
            inputs.is_init = True
            worker.put_inputs(inputs)
            self._system_output_values = worker.get_latest_outputs()

            # Place once more for next update to consume.
            inputs = copy.copy(inputs)
            inputs.is_init = False
            worker.put_inputs(inputs)

        self.DeclareInitializationUnrestrictedUpdateEvent(on_init)

        def on_discrete_update(context, raw_state):
            if deterministic:
                self._system_output_values = worker.get_latest_outputs()
            else:
                maybe = worker.maybe_get_latest_outputs()
                if maybe is not None:
                    self._system_output_values = maybe

            inputs = make_inputs_value(context)
            worker.put_inputs(inputs)

        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec, 0.0, on_discrete_update
        )
        # TODO(eric.cousineau): Publish events?


def main():
    run(DirectWorker, num_systems=1, deterministic=True, do_print=True)
    run(ThreadWorker, num_systems=1, deterministic=True, do_print=True)
    run(MpWorker, num_systems=1, deterministic=True, do_print=True)

    run(DirectWorker, num_systems=5, deterministic=True)
    run(ThreadWorker, num_systems=5, deterministic=True)
    run(MpWorker, num_systems=5, deterministic=True)
    # run(ThreadWorker, num_systems=5, deterministic=False)
    # run(MpWorker, num_systems=5, deterministic=False)


def run(worker_cls, *, num_systems, deterministic=True, do_print=False):
    print()
    print(
        f"{worker_cls.__name__}, num_systems={num_systems}, "
        f"deterministic={deterministic}"
    )

    # sometimes there are weird startup transients...

    builder = DiagramBuilder()

    clock = builder.AddSystem(AbstractClock())

    t_sim = 0.1
    period_sec = t_sim / 10
    wrapper_period_sec = period_sec
    # wrapper_period_sec = period_sec / 10

    worker_systems = []
    for i in range(num_systems):
        busy_system = BusyDiscreteSystem(
            period_sec, prefix=f"[{i}] ", do_print=do_print,
        )
        worker = worker_cls(busy_system)
        worker_system = builder.AddSystem(
            WorkerSystem(worker, period_sec=wrapper_period_sec)
        )
        builder.Connect(
            clock.get_output_port(),
            worker_system.get_input_port(),
        )
        worker_systems.append(worker_system)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    t_sim_start = simulator.get_context().get_time()
    t_wall_start = time.time()

    simulator.AdvanceTo(t_sim)

    dt_sim = simulator.get_context().get_time()
    dt_wall = time.time() - t_wall_start
    rate = dt_sim / dt_wall
    print(f"Rate: {rate:.3g}")

    context = worker_system.GetMyContextFromRoot(simulator.get_context())
    y = worker_system.get_output_port().Eval(context)
    print(f"y: {y:.3g}")

    custom_atexit_dispatch()


if __name__ == "__main__":
    main()
