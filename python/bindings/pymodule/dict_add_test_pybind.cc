#include <map>
#include <string>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using std::map;
using std::string;

class Test {
 public:
  map<string, int> a_func() const {
    return {{"a", 1}};
  }

  map<string, string> b_func() const {
    return {{"b", "stuff"}};
  }
};

int main() {
  py::scoped_interpreter guard;

  py::handle h = py::module::import("sys").attr("modules")["__main__"];
  py::module m = py::reinterpret_borrow<py::module>(h);

  py::class_<Test> cls(m, "Test");
  cls
    .def(py::init())
    .def(
      "combine_func",
      [](const Test* self) {
        py::dict out;
        py::object update = out.attr("update");
        update(self->a_func());
        update(self->b_func());
        return out;
      });

  py::exec(R"""(
obj = Test()
print(obj.combine_func())
)""");
  return 0;
}

/*
Output:
{u'a': 1, u'b': u'stuff'}
*/
