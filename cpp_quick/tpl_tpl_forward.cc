// @ref http://stackoverflow.com/questions/32282705/a-failure-to-instantiate-function-templates-due-to-universal-forward-reference

#include <utility>

//
// It **seems** that the templated type T<A> should
// behave the same as an bare type T with respect to
// universal references, but this is not the case.
//
template <template <typename> typename T, typename A>
decltype(auto) f (T<A> && t)
{
    return std::forward<T<A>> (t);
}

template <typename A>
struct foo
{
    A bar;
};

int main() {
    struct foo<int>        x1 { .bar = 1 };
    struct foo<int> const  x2 { .bar = 1 };
    struct foo<int> &      x3 = x1;
    struct foo<int> const& x4 = x2;

    // all calls to `f` **fail** to compile due
    // to **unsuccessful** binding of T&& to the required types
    auto r1 = f (x1);
    auto r2 = f (x2);
    auto r3 = f (x3);
    auto r4 = f (x4);
    auto r5 = f (foo<int> {1}); // only rvalue works
}
