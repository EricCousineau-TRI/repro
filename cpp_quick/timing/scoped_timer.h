#include <iostream>
#include <string>
#include <cmath>

#include <chrono>
#include <limits>
#include <thread>

namespace timing {

// From: drake-distro:49e44b7:drake/common/test/measure_execution.h
using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<double>;
using TimePoint = std::chrono::time_point<Clock, Duration>;


inline void sleep(double seconds) {
  int ms = std::round(seconds * 1000);
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

class Timer {
 public:
  Timer() {}
  inline bool is_active() const { return is_active_; }
  inline double start() {
    if (is_active_) {
      throw std::runtime_error("Timer already started");
    }
    start_ = Clock::now();
    is_active_ = true;
    return elapsed_; // Previously elapsed
  }
  inline double elapsed() const {
    if (is_active_) {
      return Duration(Clock::now() - start_).count();
    } else {
      return elapsed_;
    }
  }
  inline double stop() {
    if (!is_active_) {
      throw std::runtime_error("Time is not started");
    }
    elapsed_ = elapsed();
    is_active_ = false;
    return elapsed_;
  }
 private:
  TimePoint start_;
  double elapsed_{}; // When not active
  bool is_active_{};
};

class ScopedTimer {
 public:
  ScopedTimer(Timer& timer)
      : timer_(timer) {
    timer_.start();
  }
  ScopedTimer(Timer& timer, const std::function<void(double)>& on_stop)
      : timer_(timer), on_stop_(on_stop) {
    timer_.start();
  }
  ~ScopedTimer() {
    double elapsed = timer_.stop();
    if (on_stop_) {
      on_stop_(elapsed);
    }
  }
 private:
  Timer& timer_;
  std::function<void(double)> on_stop_;
};

class ScopedTimerMessage : public ScopedTimer {
 public:
  ScopedTimerMessage(Timer& timer,
                     const std::string& message = "Elapsed time (s): ")
    : ScopedTimer(timer,
                  [=](double t) { std::cout << message << t << std::endl; })
      {}
};

}  // namespace timing
