// https://github.com/pybind/pybind11/issues/1765

#include <iostream>
#include <map>
#include <string>
#include <variant>
#include <vector>

// #include <pybind11/embed.h>
// #include <pybind11/eval.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>

// namespace py = pybind11;

struct node_t;

typedef std::variant<int, float, std::string, node_t, std::vector<node_t>> value_t;

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& xs) {
  os << "[";
  for (auto& x : xs)
    os << x << ", ";
  os << "]";
  return os;
}

// Needs one non-pack parameter?
template <typename Arg0, typename... Args>
std::ostream& operator<<(std::ostream& os, const std::variant<Arg0, Args...>& x) {
  std::visit([&os](auto &&arg) { os << arg; }, x);
  return os;
}

template <typename... Args>
std::ostream& operator<<(std::ostream& os, const std::map<Args...>& xs) {
  os << "{";
  for (auto& pair : xs)
    os << pair.first << ": " << pair.second << ", ";
  os << "}";
  return os;
}

struct node_t : public std::map<std::string, value_t> {};

// void init_module(py::module m) {
// }

int main(int, char**) {
  node_t x = {{{"x", 1}}};
  std::cout << x << std::endl;

//   py::scoped_interpreter guard{};

//   py::module m("test_module");
//   init_module(m);
//   py::globals()["m"] = m;

//   py::print("[ Eval ]");
//   py::exec(R"""(
// print("Something")
// )""");

//   py::print("[ Done ]");

  return 0;
}
