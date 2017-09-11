#!/usr/bin/env directorPython

import time

from director import lcmUtils
from robotlocomotion import viewer_draw_t

try:
    def callback(msg, channel):
        print("channel: {}\n  name: {}\n  timestamp: {}".format(
            channel, msg.link_name, msg.timestamp))

    sub = lcmUtils.addSubscriber(
        'DRAKE_DRAW_FRAMES.*',
        messageClass = viewer_draw_t,
        callback = callback,
        callbackNeedsChannel = True)

    print "Spinning"
    while True:
        time.sleep(0.01)

except KeyboardInterrupt:
    pass
