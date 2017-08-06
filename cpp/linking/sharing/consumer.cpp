#include "producer.h"

void tmp() {
  printf("tmp: " CONSUMER "\n");
}
static void tmp_static() {
  printf("tmp_static: " CONSUMER "\n");
}

extern "C" funcs_t entry(funcs_t funcs_in) {
    printf("[ " CONSUMER " ]\n");
    funcs_in();
    return funcs_t(&tmp, &tmp_static);
}
