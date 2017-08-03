#include <cstdio>
void produce() {
    static int value = 0;
    ++value;
    printf("  produce: %d (%p)\n", value, (void*)&value);
}
