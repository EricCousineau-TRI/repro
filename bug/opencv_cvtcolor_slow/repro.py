import sys
import timeit

import cv2
import numpy as np


def main():
    np.random.seed(0)
    rgb = np.random.randint(0, high=255, size=(480, 848, 3), dtype=np.uint8)
    bgr = np.zeros_like(rgb)

    count = 1000
    scope = dict(globals())
    scope.update(locals())

    dt_cv = timeit.timeit('cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR, bgr)', number=count, globals=scope)
    print(dt_cv / count)
    sys.stdout.flush()

    dt_np = timeit.timeit('bgr[:] = rgb[:, :, ::-1]', number=count, globals=scope)
    print(dt_np / count)


if __name__ == "__main__":
    main()


"""
Ubuntu 18.04, CPython 3.6.9, 48 core machine:

0.00018751850200351326
0.00226589811098529
"""
