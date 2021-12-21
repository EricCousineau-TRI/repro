from contextlib import contextmanager
import cProfile as profile
import dataclasses as dc
import pstats
import time

import torch
import torch.autograd.profiler as profiler


@dc.dataclass
class ProfilingWallClock:
    # Wall-time delta.
    dt: float = None

    def save_to_file(self, *, base):
        return []

    @contextmanager
    def context(self):
        t_start = time.time()
        yield self
        self.dt = time.time() - t_start

    @classmethod
    def context_factory(cls):
        return cls().context()


@dc.dataclass
class ProfilingCProfile(ProfilingWallClock):
    stats: object = None

    def save_to_file(self, *, base):
        # To preview, consider using snakeviz:
        # https://jiffyclub.github.io/snakeviz/
        output_file = f"{base}_stats.txt"
        self.stats.sort_stats("tottime", "cumtime")
        self.stats.dump_stats(output_file)
        return [output_file]

    @contextmanager
    def context(self):
        pr = profile.Profile()
        pr.enable()
        with super().context():
            yield self
        pr.disable()
        self.stats = pstats.Stats(pr)