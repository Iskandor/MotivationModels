import time

from utils.RunningAverage import RunningAverage


class PPOTimeEstimator:
    def __init__(self, steps):
        self.total_steps = steps
        self.remaining_steps = steps
        self.remaining_time = None

        self.time_per_step = RunningAverage()
        self.start = time.time()
        self.end = time.time()

    def update(self, step=1):
        self.end = time.time()
        self.remaining_steps -= step
        self.time_per_step.update((self.end - self.start) / step)
        self.remaining_time = self.time_per_step.value() * self.remaining_steps
        self.start = time.time()

    def __str__(self):
        remaining_time = self.remaining_time
        hours = int(remaining_time // 3600)
        remaining_time -= hours * 3600
        minutes = int(remaining_time // 60)
        remaining_time -= minutes * 60
        seconds = int(remaining_time)

        return 'Progress: {0:.0f}% ETA: {1:d}:{2:02d}:{3:02d}'.format((1 - self.remaining_steps / self.total_steps) * 100, hours, minutes, seconds)
