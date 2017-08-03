#include "producer.h"
extern "C" void entry() {
    printf(CONSUMER "\n");
    produce();
}
