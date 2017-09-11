#!/usr/bin/env directorPython

import sys
import time

from director import lcmUtils
from robotlocomotion import viewer_draw_t

do_sleep = False
if len(sys.argv) == 2:
    if sys.argv[1] == '--do_sleep':
        do_sleep = True
    else:
        raise Exception("Bad flag: {}".format(sys.argv[1]))

def make_msg(name):
    msg = viewer_draw_t()
    msg.num_links = 1
    msg.link_name = [name]
    msg.robot_num = [0]
    msg.position = [[0] * 3]
    msg.quaternion = [[0] * 4]
    return msg

def maybe_sleep():
    if do_sleep:
        time.sleep(0.05)

try:
    a_channel = "DRAKE_DRAW_FRAMES_A"
    a_msg = make_msg('a')

    b_channel = "DRAKE_DRAW_FRAMES_B"
    b_msg = make_msg('b')

    i = 0
    n = 3
    for i in xrange(n):
        a_msg.timestamp = b_msg.timestamp = i
        print("pub {}".format(i))
        lcmUtils.publish(a_channel, a_msg)
        maybe_sleep()
        lcmUtils.publish(b_channel, b_msg)
        maybe_sleep()
        time.sleep(1)
        print("")

except KeyboardInterrupt:
    pass
