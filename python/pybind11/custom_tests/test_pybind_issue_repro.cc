// https://github.com/pybind/pybind11/issues/1765

/* Requires pybind11 patch:

diff --git a/include/pybind11/stl.h b/include/pybind11/stl.h
index 32f8d29..e4d06e8 100644
--- a/include/pybind11/stl.h
+++ b/include/pybind11/stl.h
@@ -331,6 +331,13 @@ struct visit_helper {
 /// Generic variant caster
 template <typename Variant> struct variant_caster;
 
+template <typename Variant> struct variant_caster_name;
+
+template <template<typename...> class V, typename... Ts>
+struct variant_caster_name<V<Ts...>> {
+    static constexpr auto value = _("Union[") + detail::concat(make_caster<Ts>::name...) + _("]");
+};
+
 template <template<typename...> class V, typename... Ts>
 struct variant_caster<V<Ts...>> {
     static_assert(sizeof...(Ts) > 0, "Variant must consist of at least one alternative.");
@@ -364,7 +371,7 @@ struct variant_caster<V<Ts...>> {
     }
 
     using Type = V<Ts...>;
-    PYBIND11_TYPE_CASTER(Type, _("Union[") + detail::concat(make_caster<Ts>::name...) + _("]"));
+    PYBIND11_TYPE_CASTER(Type, variant_caster_name<Type>::value);
 };
 
 #if PYBIND11_HAS_VARIANT

*/

#include <iostream>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct node_t;

typedef std::variant<int, float, std::string, node_t, std::vector<node_t>> value_t;

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& xs) {
  os << "[";
  bool is_first = true;
  for (auto& x : xs) {
    if (!is_first) os << ", "; else is_first = false;
    os << x;
  }
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
  bool is_first = true;
  for (auto& pair : xs) {
    if (!is_first) os << ", "; else is_first = false;
    os << pair.first << ": " << pair.second;
  }
  os << "}";
  return os;
}

struct node_t : public std::map<std::string, value_t> {};

namespace pybind11::detail {

template <>
struct variant_caster_name<value_t> {
    static constexpr auto value = _("value_t");
};

// N.B. To support `cast` using inheritance rather than direct aliasing,
// explicitly override the value of `map_caster`, rather than inheriting
// directly from `type_caster<map<string, value_t>>`.
template <>
struct type_caster<node_t>
    : public map_caster<node_t, std::string, value_t> { };

}

int main(int, char**) {
  std::vector<node_t> v({
    {{{"a", 1}}},
    {{{"b", 2}}},
    {{{"c", 3}}},
    });
  node_t value = {{{"x", 1}, {"v", v}}};
  std::cout << "cpp: " << value << std::endl;

  // To Python.
  py::scoped_interpreter guard{};
  py::dict vars = py::globals();
  vars["value"] = value;

  py::exec(R"""(
print("py: {}".format(value))
value["v"] = 10.2
)""");

  // To C++.
  node_t new_value = py::cast<node_t>(vars["value"]);
  std::cout << "cpp: " << new_value << std::endl;

  return 0;
}

/* Output:

cpp: {v: [{a: 1}, {b: 2}, {c: 3}], x: 1}
py: {'v': [{'a': 1}, {'b': 2}, {'c': 3}], 'x': 1}
cpp: {v: 10.2, x: 1}

*/
