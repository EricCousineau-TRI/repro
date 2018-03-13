#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;

class Yar {
 public:
  Yar() = default;
  Yar(const Yar&) = delete;
  Yar(Yar&&) = delete;
};

struct Stuff {
  const Yar& member;
};

int main() {
  py::scoped_interpreter guard;

  py::module m("__main__");
  py::dict d = m.attr("__dict__");
  py::dict local = py::dict();

  py::class_<Yar>(m, "Yar")
    .def(py::init());

  py::class_<Stuff>(m, "Stuff")
    .def(py::init([](const Yar* m) {
      return Stuff{*m};
    }))
    .def_readonly("member", &Stuff::member);

  py::exec(R"""(
y = Yar()
s = Stuff(y)
print(s.member)
)""", d, local);

  return 0;
}
