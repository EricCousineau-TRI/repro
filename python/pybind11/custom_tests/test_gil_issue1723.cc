// From: https://github.com/pybind/pybind11/issues/1723
#include <iostream>
#include <thread>
#include <future>
#include <functional>

#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void hello() {
  std::cout << "hello\n";
}

// Keep this as C++ code!
struct Executor {
  using Callback = std::function<void ()>;

  void execute() {
    auto func = channel.get_future().get();
    func();
  }

  explicit Executor()
    : channel(),
      worker([this]{execute();}) {
  }

  void run(Callback callback) {
    channel.set_value(callback);
  }

  void stop() {
    if (!stopped) {
      worker.join();
      stopped = true;
    } else {
      throw std::runtime_error("Cannot stop twice");
    }
  }

  std::promise<Callback> channel;
  std::thread worker;
  bool stopped{false};

  ~Executor() {
    if (!stopped) {
      std::cerr
          << "WARNING: Joining thread in destructor for Python stuff might "
             "be bad, b/c the `std::function<>` dtor could cause a decref "
             "without GIL being held. Call `stop()` explicitly instead!\n";
      stop();
    }
  }
};

void init_module(py::module m) {
  m.def("hello", &hello, "Say hello");
  py::class_<Executor>(m, "Executor")
      .def(py::init<>())
      .def("run",
        [](Executor& self, Executor::Callback callback) {
          // Acquire when executing potential Python code!
          // In general, you will want wrapped stuff to do GIL acquisition.
          // NOTE: I (eric) dunno if `cpp_function` automatically does this or
          // not...
          auto gil_callback = [callback]() {
            pybind11::gil_scoped_acquire lock{};
            callback();
          };
          self.run(gil_callback);
        },
        py::arg("callback"))
      .def("stop", &Executor::stop);
}

int main(int, char**) {
  py::scoped_interpreter guard{};

  py::module m("test_module");
  init_module(m);
  py::globals()["m"] = m;

  py::print("[ Eval ]");
  py::exec(R"""(
import time

def bye():
    print('bye')

m.hello()
ex = m.Executor()
ex.run(callback=bye)
# Give thread time to execute.
time.sleep(0.1)
ex.stop()
)""");

  py::print("[ Done ]");

  return 0;
}
