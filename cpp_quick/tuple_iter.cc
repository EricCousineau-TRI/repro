#include "tuple_iter.h"

#include <string>
#include <iostream>

#include "tuple_future.h"
#include "name_trait.h"

using std::cout;
using std::endl;

/* <snippet from="https://bitbucket.org/martinhofernandes/wheels/src/default/include/wheels/meta/type_traits.h%2B%2B?fileviewer=file-view-default#cl-161"> */
// @ref http://stackoverflow.com/a/13101086/170413
//! Tests if T is a specialization of Template
template <typename T, template <typename...> class Template>
struct is_specialization_of : std::false_type {};
template <template <typename...> class Template, typename... Args>
struct is_specialization_of<Template<Args...>, Template> : std::true_type {};
/* </snippet> */

int main() {
    int i = 0;
    auto f = [&](auto&& x) {
        cout << "visit[" << i << "]: " << x << endl;
        i += 1;
    };

    visit_args(f, 1, "hello");

    i = 0;
    auto t = std::make_tuple(1, "hello", 10.5);
    visit_tuple(f, t);

    cout
         << PRINT(( is_specialization_of<decltype(t), std::tuple>::value ))
         << PRINT(( is_specialization_of<decltype(i), std::tuple>::value ));

    return 0;
}
