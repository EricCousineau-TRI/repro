#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <ratio>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "drake/common/drake_assert.h"
#include "drake/common/drake_throw.h"
#include "drake/common/never_destroyed.h"  // HACK IWYU violation
#include "timing_profiler/running_stats.h"

namespace timing_profiler {

// TODO(SeanCurtis-TRI): Need to figure out how to prevent calling timer
// functions *before* calling start.

// TODO(eric.cousineau): How to make running stats and timers thread-safe and
// lock-free?

class LapTimer {
  enum class Mode { Uninitialized, NeedsStart, NeedsStop, Lap };

 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = std::chrono::time_point<Clock>;
  using StatsDuration = std::chrono::duration<double, std::ratio<1, 1>>;

  static_assert(
      Clock::period::den == 1000000000,
      "Clock must have nanosecond resolution");

  LapTimer(std::string label) : label_(label) {}
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(LapTimer)

  /**
  Starts the timer running.
  @pre Not in lapping mode.
  */
  void start() {
    if (mode_ == Mode::Uninitialized) {
      mode_ = Mode::NeedsStart;
    }
    check_mode("start", Mode::NeedsStart);
    mode_ = Mode::NeedsStop;
    start_ = Clock::now();
  }

  void cancel() {
    check_mode("cancel", Mode::NeedsStop);
    mode_ = Mode::NeedsStart;
    start_ = TimePoint{};
  }

  /**
  Stops the timer running.
  @pre Not in lapping mode.
   */
  void stop() {
    const TimePoint now = Clock::now();
    check_mode("stop", Mode::NeedsStop);
    mode_ = Mode::NeedsStart;
    record_interval(now - start_);
    start_ = TimePoint{};
  }

  /** Records intervals across calls. Puts timer in lapping mode. */
  void lap()  {
    const TimePoint now = Clock::now();
    if (mode_ == Mode::Lap) {
      DRAKE_ASSERT(start_ != TimePoint{});
      record_interval(now - start_);
    } else {
      check_mode("lap", Mode::Uninitialized);
      DRAKE_ASSERT(start_ == TimePoint{});
      mode_ = Mode::Lap;
    }
    start_ = now;
  }

  const std::string& label() const { return label_; }

  bool strict() const { return strict_; }
  void set_strict(bool strict) { strict_ = strict; }

  /**  Reports the average lap time across all recorded laps, reported in
  desired ratio against seconds.  */
  const RunningStats<double>& running_stats() const {
    // We should avoid calling on stats when a timer is running.
    // Not perfect, but eh.
    if (strict_ && mode_ == Mode::NeedsStop) {
      throw std::runtime_error(fmt::format(
          "Timer '{}': Trying to get stats, but timer is still running",
          label_));
    }
    return running_stats_;
  }

  void reset_running_stats() {
    running_stats_.reset();
    if (mode_ == Mode::Lap) {
      start_ = TimePoint{};
      mode_ = Mode::Uninitialized;
    }
  }

  /** RAII Scope. */
  class Scope {
   public:
    Scope(LapTimer* timer) : timer_(timer) {
      DRAKE_DEMAND(timer != nullptr);
      timer_->start();
    }
    ~Scope() { timer_->stop(); }

    Scope(const Scope&) = delete;
    void operator=(const Scope&) = delete;
    Scope(Scope&&) = default;
    Scope& operator=(Scope&&) = default;

   private:
    LapTimer* timer_{};
  };

  Scope scope() { return Scope(this); }

 private:
  void record_interval(TimePoint::duration delta) {
    running_stats_.Add(StatsDuration(delta).count());
  }

  void check_mode(std::string_view context, Mode desired) const {
    DRAKE_DEMAND(sane_ == 0xf000000d);
    if (mode_ != desired) {
      throw std::runtime_error(fmt::format(
          "{} for '{}': Current mode ({}) is not desired mode ({})",
          context, label_, to_string(mode_), to_string(desired)));
    }
  }

  static std::string to_string(Mode mode) {
    switch (mode) {
      case Mode::Uninitialized: return "Uninitialized";
      case Mode::NeedsStart: return "NeedsStart";
      case Mode::NeedsStop: return "NeedsStop";
      case Mode::Lap: return "NeedsStop";
    }
    DRAKE_UNREACHABLE();
  }

  // The time (in clock cycles) at which the timer was started.
  TimePoint start_;

  // What mode is the timer in?
  Mode mode_{Mode::Uninitialized};

  // Label.
  std::string label_;
  // Running stats.
  // TODO(eric.cousineau): Use histogram?
  RunningStats<double> running_stats_{};  
  bool strict_{true};
  uint64_t sane_{0xf000000d};
};

using LapTimersView = std::vector<const LapTimer*>;

/**  Creates a string representing a table of all of the averages.  */
std::string GetTimersSummary(const LapTimersView& timers);

/**  Class for storing sets of timers for profiling. _not_ threadsafe.  */
class Profiler {
 public:
  Profiler() = default;

  /**  Creates a lap timer which uses the given label for display.
   @param  label  The string to display when reporting the profiling results.
   @returns  The timer.  */
  LapTimer& AddTimer(std::string label) {
    std::lock_guard<std::mutex> lock(mutex_);
    DRAKE_DEMAND(label_to_timer_.find(label) == label_to_timer_.end());
    // TODO(eric.cousineau): Use less indirection?
    timers_.emplace_back(std::make_unique<LapTimer>(label));
    LapTimer& timer = *timers_.back();
    label_to_timer_[label] = &timer;
    timers_view_.push_back(&timer);
    return timer;
  }

  // WARNING: Very slow!
  // TODO(eric): How to make per-instance stuff easier?
  LapTimer& AddOrGetTimer(std::string label) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto iter = label_to_timer_.find(label);
      if (iter != label_to_timer_.end()) {
        return *iter->second;
      }
    }
    return AddTimer(label);
  }

  int num_timers() const {
    return timers_.size();
  }

  const LapTimer& get_timer(int i) {
    return *timers_.at(i);
  }

  const LapTimersView& get_timers() const { return timers_view_; }

  const LapTimer& GetTimerByLabel(const std::string& label) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return *label_to_timer_.at(label);
  }

  void ResetTimers() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& timer : timers_) {
      timer->reset_running_stats();
    }
  }

 private:
  mutable std::mutex mutex_;
  // The timers.
  std::vector<std::unique_ptr<LapTimer>> timers_;
  LapTimersView timers_view_;
  std::map<std::string, LapTimer*> label_to_timer_;
};

}  // namespace timing_profiler
