// Purpose: Debug unique ptr casting.

#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "constructor_stats.h"

namespace py = pybind11;
using namespace py::literals;
using namespace std;

void bind_ConstructorStats(py::module &m);

enum Label : int {
  BaseUniqueLabel,
};

template <int label>
class DefineBase {
 public:
  DefineBase(int value)
      : value_(value) {
    track_created(this, value);
  }
  // clang does not like having an implicit copy constructor when the
  // class is virtual (and rightly so).
  DefineBase(const DefineBase&) = delete;
  virtual ~DefineBase() {
    track_destroyed(this);
  }
  virtual int value() const { return value_; }
 private:
  int value_{};
};

template <int label>
class DefinePyBaseWrapped : public py::alias_wrapper<DefineBase<label>> {
 public:
  using BaseT = py::alias_wrapper<DefineBase<label>>;
  using BaseT::BaseT;
  int value() const override {
    PYBIND11_OVERLOAD(int, BaseT, value);
  }
};

// Base - wrapper alias used directly.
typedef DefineBase<BaseUniqueLabel> BaseUnique;
typedef DefinePyBaseWrapped<BaseUniqueLabel> PyBaseUnique;

template <typename... Args>
using class_unique_ = py::class_<Args...>;

int main(int argc, char* argv[]) {
    py::scoped_interpreter guard{};
    py::module m("_test_alias_wrapper");

    bind_ConstructorStats(m);

    class_unique_<BaseUnique, PyBaseUnique>(m, "BaseUnique")
        .def(py::init<int>())
        // Factory method.
        .def(py::init([]() { return new PyBaseUnique(10); }))
        .def("value", &BaseUnique::value);

    py::dict globals = py::globals();
    globals["m"] = m;

    py::str file;
    if (argc < 2) {
        file = "python/pybind11/custom_tests/test_alias_wrapper.py";
    } else {
        file = argv[1];
    }
    py::print(file);
    py::eval_file(file);

    cout << "[ Done ]" << endl;

    return 0;
}
