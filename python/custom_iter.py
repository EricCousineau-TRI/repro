class Counter(object):
    def __init__(self, upper):
        self._upper = upper
        self._current = 0

    def __iter__(self):
        self._current = 0
        return self

    def next(self):
        if self._current < self._upper:
            self._current += 1
            return self._current
        else:
            raise StopIteration()

for c in Counter(10):
    print(c)
