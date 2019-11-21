#include <Eigen/Dense>

#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct Blerg {
  Eigen::Matrix4d matrix{Eigen::Matrix4d::Identity()};
};

using rvp = py::return_value_policy;

// For reference, see:
// https://github.com/pybind/pybind11/blob/bd24155/include/pybind11/pybind11.h#L1192-L1199
template <typename PyClass, typename C, typename D>
void def_read_write_mutable(PyClass& cls, const char* name, D C::* pm) {
  cls.def_property(
      name,
      [pm](C& self) -> auto& { return self.*pm; },
      [pm](C& self, const D& value) { self.*pm = value; });
}

void init_module(py::module m) {
  py::class_<Blerg> cls(m, "Blerg");
  cls  // BR
    .def(py::init())
    // Normally, def_readwrite returns `const D&`. Here, we explicitly return
    // it as mutable.
    .def_property(
        "matrix",
        [](Blerg& self) -> auto& { return self.matrix; },
        [](Blerg& self, Eigen::Matrix4d value) { self.matrix = value; });
  // Alternative to use external function.
  def_read_write_mutable(cls, "matrix_again", &Blerg::matrix);
}

int main(int, char**) {
  py::scoped_interpreter guard{};

  py::module m("test_module");
  init_module(m);
  py::globals()["m"] = m;

  py::print("[ Eval ]");
  py::exec(R"""(
obj = m.Blerg()
print(".matrix")
print(obj.matrix)
obj.matrix[:] = 0
print(obj.matrix)

print(".matrix_again")
print(obj.matrix_again)
obj.matrix_again[:] = 1
print(obj.matrix_again)
)""");

  py::print("[ Done ]");

  return 0;
}

/**
Output:

[ Eval ]
.matrix
[[ 1.  0.  0.  0.]
 [ 0.  1.  0.  0.]
 [ 0.  0.  1.  0.]
 [ 0.  0.  0.  1.]]
[[ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]]
.matrix_again
[[ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]]
[[ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]]
[ Done ]

*/