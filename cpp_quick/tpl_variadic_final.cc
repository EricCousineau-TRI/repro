#include <iostream>

#include "name_trait.h"

using std::cout;
using std::endl;

void my_func(int x) {
    cout << "my_func(x)" << endl;
}
void my_func(int x, int y) {
    cout << "my_func(x, y)" << endl;
}


// Alternative: Reverse argument, pop first few arguments, then pass reversed arguments
// Most robust: Specific receivers for tuples sets, specific unpacking mechanisms
// http://loungecpp.wikidot.com/tips-and-tricks%3aindices
// http://stackoverflow.com/a/15908420/7829525

// reversed indices...
template<unsigned... Is>
struct tuple_indices{ using type = seq; };

template<unsigned I, unsigned... Is>
struct reverse_indices {
    rgen_seq<I-1, Is..., I-1>
};
template<unsigned... Is>
struct reverse_indices<0, Is...> : forward_indices<Is...> { };


template<typename Tuple>
void apply_tuple(Tuple&& 


// Goal: std::get<Indices>(std::forward<Tuple>(t))...


// template<typename ... Args>
// void tpl_func(Args ... args, int x) {
//     cout << "tpl_func" << endl;
//     my_func(args...);
// }

int main() {
    // tpl_func(1, 2, 3);
    tpl_func(1, 2);
}
