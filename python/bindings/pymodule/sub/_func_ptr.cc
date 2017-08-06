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

int call_cpp(const std::function<int(int)>& func) {
  cout << "cpp: call_cpp" << endl;
  return func(1);
}

int func_cpp(int value) {
  cout << "cpp: func_cpp" << endl;
  return 10000 * value;
}

PYBIND11_PLUGIN(_func_ptr) {
  py::module m("_func_ptr", "Call Python method.");
  m.def("call_cpp", &call_cpp);
  m.def("func_cpp", &func_cpp);
  return m.ptr();
}

}  // namespace func_ptr
