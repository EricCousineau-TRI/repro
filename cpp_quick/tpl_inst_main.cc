#include "tpl_inst.h"

int main() {
    bool b {};
    char c {};
    int x {};
    double y {};

    tpl_func(x);
    tpl_func(y);

    tpl_func_var(x, y);

    test t;
    t.tpl_method_source(x);
    // t.tpl_method_source(y);

    t.tpl_method_source_spec(b);
    t.tpl_method_source_spec(c);
    t.tpl_method_source_spec(x);
    t.tpl_method_source_spec(y);
    return 0;
}
