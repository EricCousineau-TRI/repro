#include "timing_profiler/profiler.h"

#include <sstream>

#include <fmt/format.h>

namespace timing_profiler {

std::string GetTimersSummary(const LapTimersView& timers) {
  std::stringstream ss;
  ss << fmt::format(
      "{:<30}{:>15}{:>15}{:>10}{:>15}{:>15}",
      "Label",
      "Mean Time (s)",
      "Norm stddev",
      "Samples",
      "Min (s)",
      "Max (s)") << "\n";
  for (size_t i = 0; i < timers.size(); ++i) {
    const LapTimer& timer = *timers[i];
    const timing_profiler::RunningStats<double> stats = timer.running_stats();
    ss << fmt::format(
        "{:<30}{:>15.7g}{:>15.7g}{:>10}{:>15.7g}{:>15.7g}",
        timer.label(),
        stats.mean(),
        stats.stddev() / stats.mean(),
        stats.count(),
        stats.min(),
        stats.max());
    if (i + 1 < timers.size()) ss << "\n";
  }
  return ss.str();
}

}  // namespace timing_profiler
