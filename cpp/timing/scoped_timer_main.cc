#include <sstream>

#include "scoped_timer.h"

using std::ostringstream;
using namespace timing;

int main() {
  double dt = 0.1;
  Timer timer;
  for (int i = 0; i < 5; ++i) {
    ostringstream oss;
    oss << "iter " << i;
    ScopedTimerMessage timer_scope(timer, oss.str());
    ScopedWithTimer<> timer(oss.str());
    {
      SCOPED_TIMER(inner_loop);
    }
    sleep(dt);
  }
  return 0;
}
