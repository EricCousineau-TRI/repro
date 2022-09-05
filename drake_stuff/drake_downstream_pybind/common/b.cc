#include "common/b.h"

#include "drake/common/text_logging.h"
#include "common/a.h"

namespace example {

void FuncB() {
  FuncA();
  drake::log()->info("FuncB");
}

}  // namespace
