#include "timing_profiler/profiler_lcm_stats.h"

#include <chrono>

#include "drake/common/never_destroyed.h"
#include "timing_profiler/time.h"

namespace timing_profiler {

running_time_stats_t ToLcmMessage(
    int64_t timestamp, const RunningStats<double>& stats) {
  running_time_stats_t message{};
  message.timestamp = timestamp;
  message.count = stats.count();
  message.mean = stats.mean();
  message.m2 = stats.m2();
  message.min = stats.min();
  message.max = stats.max();
  if (stats.mean() != 0.0) {
    message.norm_stddev = stats.stddev() / stats.mean();
  } else {
    message.norm_stddev = 0.0;
  }
  return message;
}

const running_time_stats_map_t& ProfilerLcmEncoder::UpdateMessage() {
  const auto& timers = *timers_;
  // Grow as needed, only copying in new names.
  DRAKE_DEMAND(timers.size() >= static_cast<size_t>(message_.count));
  for (size_t i = message_.count; i < timers.size(); ++i) {
    message_.count += 1;
    message_.names.push_back(timers[i]->label());
    message_.stats.push_back(running_time_stats_t{});
  }

  const int64_t timestamp = CurrentTimeMicroseconds();
  message_.timestamp = timestamp;
  for (size_t i = 0; i < timers.size(); ++i) {
    message_.stats[i] = ToLcmMessage(timestamp, timers[i]->running_stats());
  }
  return message_;
}

}  // namespace timing_profiler
