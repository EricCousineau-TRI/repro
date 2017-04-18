#include "tpl_inst.h"

#define EVAL(x) std::cout << ">>> " #x << std::endl; x; std::cout << std::endl

int main() {
    bool b {};
    char c {};
    int x {};
    double y {};

    EVAL(tpl_func(x));
    EVAL(tpl_func(y));

    EVAL((tpl_func_var(x, y)));

    test t;
    EVAL(t.tpl_method_source(x));
    // t.tpl_method_source(y);

    EVAL(t.tpl_method_source_spec(b));
    EVAL(t.tpl_method_source_spec(c));
    EVAL(t.tpl_method_source_spec(x));
    EVAL(t.tpl_method_source_spec(y));
    return 0;
}
