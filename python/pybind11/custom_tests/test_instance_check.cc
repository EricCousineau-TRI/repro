// For gitter convo: https://gitter.im/pybind/Lobby?at=5caa7d57016a930a457e78ec

#include <string>

#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct A {
  int value{};
};

struct B {
  std::string name;
};

void init_module(py::module m) {
  py::class_<A>(m, "A")
      .def_readwrite("value", &A::value);
  py::class_<B>(m, "B")
      .def_readwrite("name", &B::name);

  // Define-pure Python factory wrapper.
  py::object factory_meta_py =
      py::module::import("test_instance_check_util").attr("FactoryMeta");

  // Define hidden function to handle overloads.
  m.def("_tmp", [](int value) { return A{value}; });
  m.def("_tmp", [](std::string name) { return B{name}; });
  m.attr("MakeThing") = factory_meta_py(
      m.attr("_tmp"),
      py::make_tuple(m.attr("A"), m.attr("B"))
  );
}

int main(int, char**) {
  py::scoped_interpreter guard{};

  // Cheap hack to import utility.
  py::module util("test_instance_check_util");
  py::eval_file(
      "python/pybind11/custom_tests/test_instance_check_util.py",
      py::globals(), util.attr("__dict__"));
  py::module::import("sys").attr("modules")[util.attr("__name__")] = util;

  py::module m("test_module");
  init_module(m);
  py::globals()["m"] = m;

  py::print("[ Eval ]");
  py::exec(R"""(
a = m.MakeThing(1)
print(a.value)
assert isinstance(a, m.A)
assert isinstance(a, m.MakeThing)

b = m.MakeThing("Hello")
print(b.name)
assert isinstance(b, m.B)
assert isinstance(b, m.MakeThing)
)""");

  py::print("[ Done ]");

  return 0;
}
