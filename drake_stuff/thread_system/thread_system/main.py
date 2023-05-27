import atexit
import dataclasses as dc
import functools
import multiprocessing as mp
import multiprocessing.dummy as mp_dummy
import threading
import time
import weakref

from pydrake.all import (
    Value,
    LeafSystem,
    AbstractValue,
    BasicVector,
    DiagramBuilder,
    Simulator,
)


def busy_sleep_until(t_next):
    # busy wait
    i = 1
    coeff = 1 + 1e-5
    while time.time() < t_next:
        i *= coeff


def basic_sleep_until(t_next):
    dt = t_next - time.time()
    dt_sleep = max(dt / 10, 1e-6)
    while time.time() < t_next:
        time.sleep(dt_sleep)


def busy_sleep(dt):
    t_next = time.time() + dt
    busy_sleep_until(t_next)


def rough_sleep(dt):
    time.sleep(dt)


class ExampleDiscreteSystem(LeafSystem):
    def __init__(self, period_sec=0.01):
        super().__init__()

        self.u = self.DeclareAbstractInputPort("u", Value[object]())
        state_index = self.DeclareAbstractState(Value[object]())

        # Hack
        t_next = None

        def on_discrete_update(context, raw_state):
            u = self.u.Eval(context)

            # TODO(eric.cousineau): Use more consistent sleep_until().
            busy_sleep(period_sec)
            # nonlocal t_next
            # if t_next is not None:
            #     # busy_sleep_until(t_next)
            #     # basic_sleep_until(t_next)
            #     # rough_sleep(period_sec * 0.75)
            #     t_next += period_sec
            # else:
            #     t_next = time.time() + period_sec

            t = context.get_time()
            print(f"t={t}, u={u}")
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


class DirectSystem(LeafSystem):
    def __init__(self, system, period_sec):
        super().__init__()
        # Undeclared state!

        system_context = system.CreateDefaultContext()
        system_simulator = Simulator(system, system_context)

        system_inputs = {x.get_name(): x for x in get_input_ports(system)}
        system_outputs = {x.get_name(): x for x in get_output_ports(system)}
        system_output_values = {}

        self.inputs = {}
        self.outputs = {}

        for name, system_input in system_inputs.items():
            self.inputs[name] = declare_input_port(
                self, name, system_input.Allocate()
            )

        def calc_output(name, context, output):
            value = system_output_values[name]
            output.set_value(value)

        for name, system_output in system_outputs.items():
            calc_output_i = functools.partial(calc_output, name)
            self.outputs[name] = declare_output_port(
                self, name, system_output.Allocate(), calc_output_i
            )

        # state_index = self.DeclareAbstractState(Value[object]())

        def set_inputs(context, raw_state):
            for name, system_input in system_inputs.items():
                input = self.inputs[name]
                if input.HasValue(context):
                    value = input.Eval(context)
                    system_input.FixValue(system_context, value)

        def read_outputs(context, raw_state):
            for name, system_output in system_outputs.items():
                system_output_values[name] = system_output.Eval(system_context)

        def on_init(context, raw_state):
            # abstract_state = raw_state.get_mutable_abstract_state(state_index)
            # abstract_state.set_value(u)
            set_inputs(context, raw_state)
            system_simulator.Initialize()
            read_outputs(context, raw_state)

        self.DeclareInitializationUnrestrictedUpdateEvent(on_init)

        def on_discrete_update(context, raw_state):
            set_inputs(context, raw_state)
            system_simulator.AdvanceTo(context.get_time())
            read_outputs(context, raw_state)

        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec, 0.0, on_discrete_update
        )


def _stop_thread(thread_ref):
    thread = thread_ref()
    if thread is not None:
        thread.running = False
        thread.join()


_custom_atexit_queue = []


def custom_atexit_register(func):
    _custom_atexit_queue.append(func)


def custom_atexit_dispatch():
    for func in _custom_atexit_queue:
        func()


class ThreadSystem(LeafSystem):
    def __init__(self, system, period_sec):
        super().__init__()
        # Undeclared state!

        Process = threading.Thread
        Lock = threading.Lock

        # Critical section.
        lock = Lock()
        do_init = False
        system_t = None
        system_context = system.CreateDefaultContext()
        system_simulator = Simulator(system, system_context)

        system_inputs = {x.get_name(): x for x in get_input_ports(system)}
        system_outputs = {x.get_name(): x for x in get_output_ports(system)}
        system_input_values = {}
        system_output_values = {}

        self.inputs = {}
        self.outputs = {}

        for name, system_input in system_inputs.items():
            self.inputs[name] = declare_input_port(
                self, name, system_input.Allocate()
            )

        def calc_output(name, context, output):
            value = system_output_values[name]
            output.set_value(value)

        for name, system_output in system_outputs.items():
            calc_output_i = functools.partial(calc_output, name)
            self.outputs[name] = declare_output_port(
                self, name, system_output.Allocate(), calc_output_i
            )

        # state_index = self.DeclareAbstractState(Value[object]())

        def set_inputs():
            for name, system_input in system_inputs.items():
                value = system_input_values[name]
                system_input.FixValue(system_context, value)

        def read_outputs():
            for name, system_output in system_outputs.items():
                system_output_values[name] = system_output.Eval(system_context)

        def do_update():
            # TODO(eric.cousineau): How to handle best-effort running?
            set_inputs()
            system_simulator.AdvanceTo(system_t)
            read_outputs()

        class MyThread(Process):
            def __init__(self):
                super().__init__()
                self.running = True

            def run(self):
                with lock:
                    prev_system_t = system_t
                while self.running:
                    with lock:
                        if do_init:
                            set_inputs()
                            system_simulator.Initialize()
                            read_outputs()
                        if system_t != prev_system_t:
                            prev_system_t = system_t
                            do_update()
                    time.sleep(1e-6)

        thread = MyThread()
        thread.daemon = True
        thread.start()
        # atexit.register(_stop_thread, weakref.ref(thread))
        custom_atexit_register(
            functools.partial(_stop_thread, weakref.ref(thread))
        )

        def mailbox_inputs(context, raw_state):
            for name, system_input in system_inputs.items():
                input = self.inputs[name]
                system_input_values[name] = input.Eval(context)

        def on_init(context, raw_state):
            # abstract_state = raw_state.get_mutable_abstract_state(state_index)
            # abstract_state.set_value(u)
            nonlocal do_init
            with lock:
                mailbox_inputs(context, raw_state)
                do_init = True

        self.DeclareInitializationUnrestrictedUpdateEvent(on_init)

        def on_discrete_update(context, raw_state):
            nonlocal system_t
            with lock:
                # WARNING: This is non-deterministic.
                mailbox_inputs(context, raw_state)
                system_t = context.get_time()

        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec, 0.0, on_discrete_update
        )


ctx = mp.get_context("fork")


@dc.dataclass
class Inputs:
    system_t: float
    system_input_values: dict
    is_init: bool = False


@dc.dataclass
class Outputs:
    system_output_values: dict


class MpProcess(ctx.Process):
    def __init__(self, system, system_inputs, system_outputs):
        super().__init__()
        self.system = system
        self.system_inputs = system_inputs
        self.system_outputs = system_outputs

        self.should_stop = ctx.Event()
        self.inputs = ctx.Queue()
        self.outputs = ctx.Queue()

    @staticmethod
    def _atexit(self_ref):
        self = self_ref()
        if self is not None:
            self.stop()

    def make_atexit_callback(self):
        return functools.partial(MpProcess._atexit, weakref.ref(self))

    def stop(self):
        self.should_stop.set()
        self.join()

    def put(self, inputs):
        self.inputs.put(inputs)

    def get(self):
        return self.outputs.get()

    def maybe_get_latest(self):
        output = None
        while not self.outputs.empty():
            output = self.outputs.get()
        return output

    def run(self):
        system = self.system
        system_inputs = self.system_inputs
        system_outputs = self.system_outputs

        system_context = system.CreateDefaultContext()
        system_simulator = Simulator(system, system_context)

        system_input_values = {}
        system_output_values = {}

        def set_inputs():
            for name, system_input in system_inputs.items():
                value = system_input_values[name]
                system_input.FixValue(system_context, value)

        def read_outputs():
            for name, system_output in system_outputs.items():
                system_output_values[name] = system_output.Eval(system_context)

        while True:
            if self.should_stop.is_set():
                break
            if self.inputs.empty():
                continue
            while not self.inputs.empty():
                # Drain the queue.
                inputs = self.inputs.get()
            system_input_values = inputs.system_input_values
            set_inputs()

            if inputs.is_init:
                system_context.SetTime(inputs.system_t)
                system_simulator.Initialize()
            else:
                # TODO(eric.cousineau): How handle slowdowns?
                system_simulator.AdvanceTo(inputs.system_t)

            read_outputs()
            self.outputs.put(system_output_values)


class MultiprocessSystem(LeafSystem):
    def __init__(self, system, period_sec, deterministic=True):
        super().__init__()
        # Undeclared state!
        system_inputs = {x.get_name(): x for x in get_input_ports(system)}
        system_outputs = {x.get_name(): x for x in get_output_ports(system)}
        system_input_values = {}
        system_output_values = {}

        self.inputs = {}
        self.outputs = {}

        for name, system_input in system_inputs.items():
            self.inputs[name] = declare_input_port(
                self, name, system_input.Allocate()
            )

        def calc_output(name, context, output):
            value = system_output_values[name]
            output.set_value(value)

        for name, system_output in system_outputs.items():
            calc_output_i = functools.partial(calc_output, name)
            self.outputs[name] = declare_output_port(
                self, name, system_output.Allocate(), calc_output_i
            )

        thread = MpProcess(system, system_inputs, system_outputs)
        thread.start()
        custom_atexit_register(thread.make_atexit_callback())

        def mailbox_inputs(context):
            for name, system_input in system_inputs.items():
                input = self.inputs[name]
                system_input_values[name] = input.Eval(context)

        def on_init(context, raw_state):
            nonlocal system_output_values
            mailbox_inputs(context)
            inputs = Inputs(
                system_t=context.get_time(),
                system_input_values=system_input_values,
                is_init=True,
            )
            thread.put(inputs)
            system_output_values = thread.get()

            if deterministic:
                # Place once more for next update to consume.
                inputs = Inputs(
                    system_t=context.get_time(),
                    system_input_values=system_input_values,
                )
                thread.put(inputs)

        self.DeclareInitializationUnrestrictedUpdateEvent(on_init)

        def on_discrete_update(context, raw_state):
            nonlocal system_output_values

            # WARNING: This is non-deterministic.
            # new_system_output_values = thread.maybe_get()
            # if new_system_output_values is not None:
            #     new_system_output_values = system_output_values

            if deterministic:
                system_output_values = thread.get()
            else:
                maybe = thread.maybe_get_latest()
                if maybe is not None:
                    system_output_values = maybe

            mailbox_inputs(context)
            inputs = Inputs(
                system_t=context.get_time(),
                system_input_values=system_input_values,
            )
            thread.put(inputs)

        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec, 0.0, on_discrete_update
        )


class AbstractClock(LeafSystem):
    def __init__(self):
        super().__init__()

        def calc_t(context, output):
            t = context.get_time()
            output.set_value(t)

        self.DeclareAbstractOutputPort("t", Value[object], calc_t)


def main():
    builder = DiagramBuilder()

    clock = builder.AddSystem(AbstractClock())

    period_sec = 0.01
    t_sim = 0.1
    wrapper_period_sec = period_sec * 0.1

    # wrapper_cls = DirectSystem
    # wrapper_cls = ThreadSystem
    # wrapper_cls = MultiprocessSystem
    wrapper_cls = functools.partial(MultiprocessSystem, deterministic=False)

    # sometimes there are weird startup transients...

    my_systems = []
    for i in range(10):
        my_system = ExampleDiscreteSystem(period_sec)
        my_system = builder.AddSystem(
            wrapper_cls(my_system, period_sec=wrapper_period_sec)
        )
        builder.Connect(
            clock.get_output_port(),
            my_system.get_input_port(),
        )
        my_systems.append(my_system)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)

    t_start = time.time()
    simulator.AdvanceTo(t_sim)
    t_wall = time.time() - t_start
    rate = t_sim / t_wall
    print(f"Rate: {rate}")

    my_context = my_system.GetMyContextFromRoot(simulator.get_context())
    y = my_system.get_output_port().Eval(my_context)
    print(f"y: {y}")

    custom_atexit_dispatch()


if __name__ == "__main__":
    main()
