// Purpose: Debug callbacks with Eigen::Ref<> arguments.

#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>

#include <iostream>

#include <Eigen/Dense>

#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;
using namespace std;

using Callback =
    std::function<void(const Eigen::Ref<const Eigen::VectorXd>& x)>;

void call_thing(const Callback& func) {
  Eigen::VectorXd x(4);
  x << 10, 20, 30, 40;
  func(x);
}

int main(int argc, char* argv[]) {
    py::scoped_interpreter guard{};
    py::module m("test_callback_eigen_ref");

    m.def("call_thing", &call_thing);

    py::dict globals = py::globals();
    globals["m"] = m;

    py::exec(R"""(
def my_func(x):
    print("Python callback: {}".format(x))

m.call_thing(my_func)
)""");

    cout << "[ Done ]" << endl;

    return 0;
}
