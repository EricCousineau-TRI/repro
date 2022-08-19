#pragma once

#include <string>
#include <vector>

#include <lcm/lcm-cpp.hpp>  // Blech.

#include "timing_profiler/running_time_stats_map_t.hpp"
#include "timing_profiler/running_time_stats_t.hpp"
#include "timing_profiler/switch_t.hpp"
#include "timing_profiler/profiler.h"
#include "timing_profiler/running_stats.h"
#include "drake/common/never_destroyed.h"

namespace timing_profiler {

running_time_stats_t ToLcmMessage(
    int64_t timestamp, const RunningStats<double>& stats);

/**
Aliases into a list of timers.

Expects list of timers to never change names, nor to be removed.
*/
class ProfilerLcmEncoder {
 public:
  ProfilerLcmEncoder(const LapTimersView* timers)
      : timers_(timers) {
    DRAKE_DEMAND(timers_ != nullptr);
  }

  /**
  Updates internal message with current statistics.
  Only resizes internal message if necesary. Not thread-safe.
  */
  const running_time_stats_map_t& UpdateMessage();

  const LapTimersView& timers() const { return *timers_; }

 private:
  const LapTimersView* timers_{};
  running_time_stats_map_t message_{};
};

/*
in whatever translation unit

ProfilerAll& prof_all() {
  static drake::never_destroyed<ProfilerAll> prof_all("PROFILER_...");
  return prof_all.access();
}
*/

class ProfilerAll {
 public:
  ProfilerAll(std::string channel, int reset_interval = 0, bool use_lcm = true)
    : lcm_encoder_(&profiler_.get_timers()),
      channel_(std::move(channel)),
      reset_interval_(reset_interval)
  {
    if (use_lcm) {
      lcm_ = std::make_unique<lcm::LCM>("udpm://231.255.66.76:6666?ttl=0");
      lcm_->subscribe("PROFILER_RESET", &ProfilerAll::HandleReset, this);
    }
  }

  Profiler& profiler() { return profiler_; }
  void Publish() {
    ++count_;
    bool need_reset = false;
    if (reset_interval_ > 0 && count_ % reset_interval_ == 0) {
      need_reset = true;
    }

    if (lcm_) {
      lcm_->publish(channel_, &lcm_encoder_.UpdateMessage());
      while (lcm_->handleTimeout(0) > 0) {}
    } else {
      if (need_reset) {
        lcm_encoder_.UpdateMessage();
        fmt::print("{}\n", GetTimersSummary(lcm_encoder_.timers()));
      }
    }

    if (need_reset) {
      Reset();
    }
  }

 private:
  void HandleReset(const lcm::ReceiveBuffer*, const std::string&) {
    Reset();
  }

  void Reset() {
    profiler_.ResetTimers();
    count_ = 0;
  }

  Profiler profiler_;
  ProfilerLcmEncoder lcm_encoder_;
  std::string channel_;
  std::unique_ptr<lcm::LCM> lcm_;
  int reset_interval_{};
  int count_{};
};

}  // namespace timing_profiler
