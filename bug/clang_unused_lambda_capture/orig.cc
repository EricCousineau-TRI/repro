// https://bugs.llvm.org/show_bug.cgi?id=31815
#include <iostream>

auto sq(int c, int x) { return c * x * x; }

struct S {
    template<class Fun>
    void for_each(Fun fun) const {
        for (auto i = 1; i < 4; ++i) {
            fun(i);
        }
    }
};

int main()
{
    S s;
    auto sum = 0;
    s.for_each([&, i = 2](auto c) mutable {
        sum += sq(c, i++);
    });
    std::cout << sum;   // 70 = 1 * 4 + 2 * 9 + 3 * 16
}
