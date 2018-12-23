#include <stdio.h>

struct stuff_t {
    /* input, vector */
    struct {
        int size;
    } p;
    /* output, matrix */
    struct {
        int rows;
        int cols;
    } m;
};

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    struct stuff_t s = { {2}, {2, 2} };
    printf("Hello world: %d\n", s.m.cols);
    return 0;
}
