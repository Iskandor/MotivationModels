class RunningAverage:
    def __init__(self):
        self._cma = 0
        self._n = 0

    def update(self, value):
        self._cma = self._cma + (value - self._cma) / (self._n + 1)
        self._n += 1

    def value(self):
        return self._cma
