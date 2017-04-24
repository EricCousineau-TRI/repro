#include <string>
#include <iostream>
#include <tuple>
#include <utility>

#include "tuple_future.h"
#include "name_trait.h"

using std::cout;
using std::endl;

template<typename F>
void visit_args(F&& f) { }

template<typename F, typename T1, typename... Args>
void visit_args(F&& f, T1&& t1, Args&&... args) {
    f(std::forward<T1>(t1));
    visit_args(std::forward<F>(f), std::forward<Args>(args)...);
}

// Modelling after: ../cpp_quick/tuple_future.h
template<typename F, typename Tuple, std::size_t... I>
void visit_tuple_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
    visit_args(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t))...);
}
template<typename F, typename Tuple>
void visit_tuple(F&& f, Tuple&& t) {
    visit_tuple_impl(
        std::forward<F>(f), std::forward<Tuple>(t),
        std::make_index_sequence<
            std::tuple_size<std::decay_t<Tuple>>::value>{});
}

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

    return 0;
}
