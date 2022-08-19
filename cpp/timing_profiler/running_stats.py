from math import sqrt


class RunningStats:
    """
    Simple implementation of Welford's online statistics computations (like
    PyTorch's BatchNorm, etc.):
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm  # noqa

    Note:
        This makes an attempt to preserve some precision (by the nautre of
        Welford's approach), but may have some minor issues.
    """
    def __init__(self):
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = 0.0
        self._max = 0.0

    def add(self, x):
        """Adds new data point and updates internal state."""
        if self._count == 0:
            self._min = x
            self._max = x
        else:
            if x < self._min:
                self._min = x
            if x > self._max:
                self._max = x
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._m2 += delta * delta2

    def count(self):
        """Number of data points."""
        return self._count

    def sum(self):
        """Sum of all data points."""
        return self._mean * self._count

    def min(self):
        """Min of all data points; 0 if no data points present."""
        return self._min

    def max(self):
        """Max of all data points; 0 if no data points present."""
        return self._max

    def mean(self):
        """Mean / average of data points."""
        return self._mean

    def m2(self):
        return self._m2

    def variance(self):
        """
        Population variance of all data points; 0 if no data points
        present.
        """
        if self._count > 0:
            return self._m2 / self._count
        else:
            return 0.0

    def stddev(self):
        """Standard deviation, or sqrt(variance). Population, not variance."""
        return sqrt(self.variance())
