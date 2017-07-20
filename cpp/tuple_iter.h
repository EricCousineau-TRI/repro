#include <tuple>
#include <utility>

template<typename F>
void visit_args(F&& f) { }

template<typename F, typename T1, typename... Args>
void visit_args(F&& f, T1&& t1, Args&&... args) {
    f(std::forward<T1>(t1));
    visit_args(std::forward<F>(f), std::forward<Args>(args)...);
}

// Modelling after: ../cpp/tuple_future.h
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
