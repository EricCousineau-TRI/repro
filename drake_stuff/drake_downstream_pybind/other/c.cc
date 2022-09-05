#include "other/c.h"

#include "drake/common/text_logging.h"
#include "common/b.h"

namespace example {

void FuncC() {
  FuncB();
  drake::log()->info("FuncC");
}

}  // namespace
