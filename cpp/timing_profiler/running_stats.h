#pragma once

#include <cmath>

#include "drake/common/drake_copyable.h"

namespace timing_profiler {

/*
Simple implementation of Welford's online variance thinger:
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
*/
template <typename T>
class RunningStats {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(RunningStats)

  RunningStats() = default;

  void Add(T x) {
    if (count_ == 0) {
      min_ = x;
      max_ = x;
    } else {
      if (x < min_) {
        min_ = x;
      }
      if (x > max_) {
        max_ = x;
      }
    }
    ++count_;
    T delta = x - mean_;
    mean_ += delta / T(count_);
    T delta2 = x - mean_;
    m2_ += delta * delta2;
  }

  void reset() {
    *this = RunningStats<T>();
  }

  int count() const { return count_; }
  T sum() const { return mean_ * count_; }
  T min() const { return min_; }
  T max() const { return max_; }
  T mean() const { return mean_; }
  // Population (not sample).
  T variance() const { return m2_ / T(count_); }
  T stddev() const { return sqrt(variance()); }

  // Sum of square differences, m₂ = ∑(xᵢ − μₙ)².
  // Not useful as user-digestible quantity, but useful storing in messages
  // to reconstruct running stats from messages.
  T m2() const { return m2_; }

  RunningStats scaled(double scale) const {
    RunningStats out;
    out.count_ = count_;
    out.mean_ = mean_ * scale;
    out.m2_ = m2_ * scale * scale;
    out.min_ = min_ * scale;
    out.max_ = max_ * scale;
    return out;
  }

 private:
  int count_{};
  T mean_{};
  T m2_{};
  T min_{};
  T max_{};
};

extern template class RunningStats<double>;

}  // namespace timing_profiler
