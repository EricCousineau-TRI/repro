// From: https://github.com/pybind/pybind11/issues/1723
#include <iostream>
#include <thread>
#include <future>

#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void hello() {
  std::cout << "hello\n";
}

struct Executor {

  void execute() {
    auto func = channel.get_future().get();
    func();
  }

  explicit Executor(py::object callback)
    : callback(std::move(callback)),
      channel(),
      worker([this]{execute();}) {
  }

  void run() {
    pybind11::gil_scoped_release release{};
    channel.set_value(callback);
  }

  py::object callback;
  std::promise<py::object> channel;
  std::thread worker;

  ~Executor() {
    worker.join();
  }
};

void init_module(py::module m) {
  m.def("hello", &hello, "Say hello");
  py::class_<Executor>(m, "Executor")
      .def(py::init<py::object>())
      .def("run", &Executor::run);
}

int main(int, char**) {
  py::scoped_interpreter guard{};

  py::module m("test_module");
  init_module(m);
  py::globals()["m"] = m;

  py::print("[ Eval ]");
  py::exec(R"""(
def bye():
    print('bye')

def main():
    pass

if True:
    m.hello()
    ex = m.Executor(bye)
    ex.run()

use_trace = False
if use_trace:
    import sys, trace
    sys.stdout = sys.stderr
    tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    tracer.runfunc(main)
else:
    main()
)""");

  py::print("[ Done ]");

  return 0;
}
