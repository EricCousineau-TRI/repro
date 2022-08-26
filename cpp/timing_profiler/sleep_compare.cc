#include <sys/prctl.h>
#include <ctime>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "timing_profiler/profiler.h"

DEFINE_int32(count, 10000, "Number of iterations to measure");
DEFINE_int32(warmup_count, 1000, "Number of warmup iterations");
DEFINE_int32(usec, 10, "Nubmer of microseconds to sleep");
DEFINE_int32(timerslack_usec, 0, "Use prctrl for timerslack. Unused if 0.");

namespace timing_profiler {
namespace {

timespec to_timespec(std::chrono::nanoseconds amount) {
  return {
    .tv_sec = amount.count() / std::nano::den,
    .tv_nsec = amount.count() % std::nano::den
  };
}

// Ah derp: https://stackoverflow.com/a/18086173/7829525
inline void nanosleep_chrono(std::chrono::nanoseconds amount) {
  timespec amount_c = to_timespec(amount);
  int result = nanosleep(&amount_c, nullptr);
  DRAKE_DEMAND(result == 0);
}

inline void clock_nanosleep_chrono(std::chrono::nanoseconds amount) {
  using clock = std::chrono::steady_clock;
  auto t_sleep = clock::now() + amount;
  timespec amount_c = to_timespec(t_sleep.time_since_epoch());
  int result = clock_nanosleep(
      CLOCK_MONOTONIC, TIMER_ABSTIME, &amount_c, nullptr);
  DRAKE_DEMAND(result == 0);
}

void sleep_for(std::chrono::nanoseconds amount) {
  std::this_thread::sleep_for(amount);
}

int DoMain() {
  if (FLAGS_timerslack_usec > 0) {
    const uint64_t timerslack_nsec = 1000 * FLAGS_timerslack_usec;
    // https://stackoverflow.com/a/60153370/7829525
    prctl(PR_SET_TIMERSLACK, timerslack_nsec, 0, 0, 0);
  }

  using namespace std::literals::chrono_literals;

  Profiler profiler;

  auto benchmark = [&](std::string name, auto sleep_func) {
    LapTimer& timer =
        profiler.AddTimer(name + "." + std::to_string(FLAGS_usec) + "us");
    const auto amount = 1us * FLAGS_usec;

    for (int i = 0; i < FLAGS_warmup_count; ++i) {
      sleep_func(amount);
    }

    for (int i = 0; i < FLAGS_count; ++i) {
      timer.start();
      sleep_func(amount);
      timer.stop();
    }
  };

  // ordering here matters! first will generally perform worst.
  benchmark("clock_nanosleep_chrono", clock_nanosleep_chrono);
  benchmark("nanosleep_chrono", nanosleep_chrono);
  benchmark("sleep_for", sleep_for);

  std::cout << GetTimersSummary(profiler.get_timers()) << std::endl;

  return 0;
}

}  // namespace
}  // namespace timing_profiler

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return timing_profiler::DoMain();
}
