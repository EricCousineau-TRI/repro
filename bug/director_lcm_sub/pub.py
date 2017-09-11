#!/usr/bin/env directorPython

import time

from director import lcmUtils
from robotlocomotion import viewer_draw_t

try:
    a_channel = "DRAKE_DRAW_FRAMES_A"
    a_msg = viewer_draw_t()
    a_msg.link_name = ["a"]

    b_channel = "DRAKE_DRAW_FRAMES_B"
    b_msg = viewer_draw_t()
    b_msg.link_name = ["b"]

    i = 0
    while True:
        a_msg.timestamp = b_msg.timestamp = i
        lcmUtils.publish(a_channel, a_msg)
        lcmUtils.publish(b_channel, b_msg)
        print("pub {}".format(i))
        i += 1
        time.sleep(0.01)

except KeyboardInterrupt:
    pass
