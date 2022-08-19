#include "timing_profiler/running_stats.h"

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace timing_profiler {
namespace {

using Eigen::VectorXd;

double square(double x) { return x * x; }

GTEST_TEST(RunningStats, Compare) {
  const int count = 5;
  VectorXd xs(count);
  xs << 2.0, 1.0, 10.5, 22.1, 3.0;
  const double min = 1.0;
  const double max = 22.1;

  const double mean = xs.mean();
  const double variance =
    (xs.array().square().sum() - square(xs.array().sum()) / count) / count;
  const double stddev = sqrt(variance);

  RunningStats<double> stats;
  for (int i = 0; i < count; ++i) {
    stats.Add(xs(i));
  }
  EXPECT_EQ(stats.count(), count);
  EXPECT_EQ(stats.min(), min);
  EXPECT_EQ(stats.max(), max);
  EXPECT_EQ(stats.mean(), mean);
  EXPECT_EQ(stats.variance(), variance);
  EXPECT_EQ(stats.stddev(), stddev);

  const double scale = 0.1;
  RunningStats<double> scaled = stats.scaled(scale);
  EXPECT_EQ(scaled.count(), count);
  EXPECT_EQ(scaled.mean(), mean * scale);
  EXPECT_EQ(scaled.min(), min * scale);
  EXPECT_EQ(scaled.max(), max * scale);
  EXPECT_NEAR(scaled.variance(), variance * scale * scale, 1e-15);
  EXPECT_NEAR(scaled.stddev(), stddev * scale, 1e-15);
}

}  // namespace
}  // namespace timing_profiler
