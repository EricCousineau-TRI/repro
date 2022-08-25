#include <sys/prctl.h>
#include <ctime>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "timing_profiler/profiler.h"

namespace timing_profiler {
namespace {

// Ah derp: https://stackoverflow.com/a/18086173/7829525
inline void nanosleep_chrono(std::chrono::nanoseconds amount) {
  timespec amount_c{
    .tv_sec = amount.count() / std::nano::den,
    .tv_nsec = amount.count() % std::nano::den
  };
  if (nanosleep(&amount_c, nullptr) != 0) {
    throw std::runtime_error("bad sleep");
  }
}

void sleep_for(std::chrono::nanoseconds amount) {
  std::this_thread::sleep_for(amount);
}

void sleep_chunks(std::chrono::nanoseconds amount) {
  using clock = std::chrono::steady_clock;
  using namespace std::literals::chrono_literals;

  auto t_next = clock::now() + amount;
  while (clock::now() < t_next) {
    sleep_for(10us);
  }
}

int DoMain() {
  // https://stackoverflow.com/a/60153370/7829525
  prctl(PR_SET_TIMERSLACK, 5000U, 0, 0, 0);

  const int count = 100000;
  using namespace std::literals::chrono_literals;

  Profiler profiler;

  auto check_sleep_for = [&](std::string name, auto sleep_func) {
    LapTimer& timer_10us = profiler.AddTimer(name + ".10us");
    // LapTimer& timer_100us = profiler.AddTimer(name + "."100us");
    // LapTimer& timer_1ms = profiler.AddTimer(name + "."1ms");

    for (int i = 0; i < count; ++i) {
      timer_10us.start();
      sleep_func(10us);
      timer_10us.stop();

      // timer_100us.start();
      // sleep_func(100us);
      // timer_100us.stop();

      // timer_1ms.start();
      // sleep_func(1ms);
      // timer_1ms.stop();
    }
  };

  check_sleep_for("sleep_for", sleep_for);
  check_sleep_for("nanosleep_chrono", nanosleep_chrono);
  // check_sleep_for("clock_nanosleep_chrono", clock_nanosleep_chrono);
  // check_sleep_for("sleep_chunks", sleep_chunks);

  std::cout << GetTimersSummary(profiler.get_timers()) << std::endl;

  return 0;
}

}  // namespace
}  // namespace timing_profiler

int main() {
  return timing_profiler::DoMain();
}
