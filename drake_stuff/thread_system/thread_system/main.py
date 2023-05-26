import atexit
import functools
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


class ExampleDiscreteSystem(LeafSystem):
    def __init__(self, period_sec=0.01):
        super().__init__()

        self.u = self.DeclareAbstractInputPort("u", Value[object]())
        state_index = self.DeclareAbstractState(Value[object]())

        def on_discrete_update(context, raw_state):
            u = self.u.Eval(context)
            # TODO(eric.cousineau): Use more consistent sleep_until().
            time.sleep(period_sec * 0.75)
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


class ThreadSystem(LeafSystem):
    def __init__(self, system, period_sec):
        super().__init__()
        # Undeclared state!

        # Critical section.
        lock = threading.Lock()
        system_t = None
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

        def read_outputs():
            for name, system_output in system_outputs.items():
                system_output_values[name] = system_output.Eval(system_context)

        def do_update():
            # TODO(eric.cousineau): How to handle best-effort running?
            system_simulator.AdvanceTo(system_t)
            read_outputs()

        class MyThread(threading.Thread):
            def __init__(self):
                super().__init__()
                self.running = True

            def run(self):
                with lock:
                    prev_system_t = system_t
                print("Running")
                while self.running:
                    with lock:
                        if system_t != prev_system_t:
                            prev_system_t = system_t
                            do_update()
                    time.sleep(1e-6)
                print("Done")

        thread = MyThread()
        thread.daemon = True
        thread.start()
        atexit.register(_stop_thread, weakref.ref(thread))

        def set_inputs(context, raw_state):
            for name, system_input in system_inputs.items():
                input = self.inputs[name]
                value = input.Eval(context)
                system_input.FixValue(system_context, value)

        def on_init(context, raw_state):
            # abstract_state = raw_state.get_mutable_abstract_state(state_index)
            # abstract_state.set_value(u)
            with lock:
                set_inputs(context, raw_state)
                system_simulator.Initialize()
                read_outputs()

        self.DeclareInitializationUnrestrictedUpdateEvent(on_init)

        def on_discrete_update(context, raw_state):
            nonlocal system_t
            with lock:
                # WARNING: This is non-deterministic.
                set_inputs(context, raw_state)
                system_t = context.get_time()

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

    # wrapper_cls = DirectSystem
    wrapper_cls = ThreadSystem
    period_sec = 0.01

    my_systems = []
    for i in range(1):
        my_system = ExampleDiscreteSystem(period_sec)
        # For thread system, may hit bottleneck when GIL is not released /
        # we're not sleeping?
        # Segfaults if period is slower than intendend system?
        my_system = builder.AddSystem(
            wrapper_cls(my_system, period_sec=period_sec * 10)
        )
        builder.Connect(
            clock.get_output_port(),
            my_system.get_input_port(),
        )
        my_systems.append(my_system)

    diagram = builder.Build()
    simulator = Simulator(diagram)

    my_context = my_system.GetMyContextFromRoot(simulator.get_context())

    t_sim = 0.1
    t_start = time.time()
    simulator.set_target_realtime_rate(1.0)

    simulator.AdvanceTo(t_sim)
    t_wall = time.time() - t_start
    rate = t_sim / t_wall
    print(f"Rate: {rate}")
    y = my_system.get_output_port().Eval(my_context)
    print(f"y: {y}")


if __name__ == "__main__":
    main()
