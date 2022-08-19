import unittest

import numpy as np

import timing_profiler.common.running_stats as mut


class Test(unittest.TestCase):
    def test_running_stats(self):
        xs = np.array([2.0, 1.0, 10.5, 22.1, 3.0])
        count = len(xs)
        sum_ = np.sum(xs)
        min_ = xs.min()
        max_ = xs.max()
        mean = np.mean(xs)
        stddev = np.std(xs)
        variance = stddev**2

        stats = mut.RunningStats()
        for x in xs:
            stats.add(x)

        self.assertEqual(stats.count(), count)
        self.assertEqual(stats.sum(), sum_)
        self.assertEqual(stats.min(), min_)
        self.assertEqual(stats.max(), max_)
        self.assertEqual(stats.mean(), mean)
        np.testing.assert_allclose(stats.stddev(), stddev, atol=1e-15, rtol=0)
        np.testing.assert_allclose(
            stats.variance(), variance, atol=2e-14, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
