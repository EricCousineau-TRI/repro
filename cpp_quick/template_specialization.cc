#include "template_specialization.h"

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
