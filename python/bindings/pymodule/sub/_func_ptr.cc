#include <cstddef>
#include <cmath>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using std::cout;
using std::endl;
using std::string;
using std::ostringstream;

namespace func_ptr {

int call(const std::function<int(int)>& func) {
  cout << "c pybind11: call" << endl;
  return func(1);
}

PYBIND11_PLUGIN(_func_ptr) {
  py::module m("_func_ptr", "Call Python method.");
  m.def("call", &call);
  return m.ptr();
}

}  // namespace inherit_check
