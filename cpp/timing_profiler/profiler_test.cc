#include "timing_profiler/profiler.h"

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace timing_profiler {
namespace {

using std::this_thread::sleep_for;

GTEST_TEST(ProfilerTest, GoodModes) {
  // Start then stop.
  {
    LapTimer timer("");
    for (int i = 0; i < 3; ++i) {
      timer.start();
      timer.stop();
    }
    EXPECT_EQ(timer.running_stats().count(), 3);
  }

  // Lap across.
  {
    LapTimer timer("");
    for (int i = 0; i < 3; ++i) {
      timer.lap();
    }
    EXPECT_EQ(timer.running_stats().count(), 2);
  }
}

GTEST_TEST(ProfilerTest, BadModes) {
  // Stop without start.
  {
    LapTimer timer("");
    EXPECT_THROW(timer.stop(), std::runtime_error);
  }

  // Start without stop. Lap with start.
  {
    LapTimer timer("");
    timer.start();
    EXPECT_THROW(timer.start(), std::runtime_error);
    EXPECT_THROW(timer.lap(), std::runtime_error);
  }

  // Lap then start or stop.
  {
    LapTimer timer("");
    timer.lap();
    EXPECT_THROW(timer.start(), std::runtime_error);
    EXPECT_THROW(timer.stop(), std::runtime_error);
  }
}

// Only use singleton + static if it makes sense.
// Otherwise, bad mojo.
Profiler& GetProfilerSingleton() {
  static drake::never_destroyed<Profiler> profiler;
  return profiler.access();
}

// Sleeps for a total of 5ms.
// Uses explicit start() and stop().
void InnerFunc() {
  static LapTimer& timer = GetProfilerSingleton().AddTimer("InnerFunc");
  timer.start();
  const std::chrono::duration<double, std::milli> sleep_time(5.);
  sleep_for(sleep_time);
  timer.stop();
}

// Sleeps for a total of 15ms.
// Uses RAII scope.
void OuterFunc() {
  static LapTimer& timer = GetProfilerSingleton().AddTimer("OuterFunc");
  auto timer_scope = timer.scope();
  const std::chrono::duration<double, std::milli> sleep_time(10.);
  sleep_for(sleep_time);
  InnerFunc();
}

// Uses lap().
void LapFunc() {
  static LapTimer& timer = GetProfilerSingleton().AddTimer("LapFunc");
  timer.lap();
}

GTEST_TEST(ProfileTester, PrintStuff) {
  // Repeat for unique stats.
  LapFunc();
  for (int i = 0; i < 3; ++i) {
    OuterFunc();
    InnerFunc();
  }
  LapFunc();

  std::cout
      << GetTimersSummary(GetProfilerSingleton().get_timers()) << std::endl;

  const LapTimer& timer_lap =
      GetProfilerSingleton().GetTimerByLabel("LapFunc");
  const LapTimer& timer_outer =
      GetProfilerSingleton().GetTimerByLabel("OuterFunc");
  const LapTimer& timer_inner =
      GetProfilerSingleton().GetTimerByLabel("InnerFunc");
  EXPECT_EQ(timer_lap.running_stats().count(), 1);
  EXPECT_EQ(timer_outer.running_stats().count(), 3);
  EXPECT_GE(timer_outer.running_stats().mean(), 0.015);
  EXPECT_GT(
      timer_lap.running_stats().sum(), timer_outer.running_stats().sum());
  EXPECT_EQ(timer_inner.running_stats().count(), 6);
  EXPECT_GE(timer_inner.running_stats().mean(), 0.005);
  EXPECT_GT(
    timer_lap.running_stats().sum(), timer_inner.running_stats().sum());
}

}  // namespace
}  // namespace timing_profiler
