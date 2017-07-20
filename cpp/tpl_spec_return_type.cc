#include "tpl_spec_return_type.h"

// [auto / delctype(auto), defined_in_source]

// // + [spec_return_auto]
// //     Does not work. (Inference does not work)
// template<>
// auto Test::tpl_method_auto<int>(const int& x) {
//     return std::string("int -> string");
// }

// template<>
// auto Test::tpl_method_auto<double>(const double& x) {
//     return 2 * x;
// }

template<>
std::string Test::tpl_method_explicit<int>(const int& x) {
    return "int -> string";
}
