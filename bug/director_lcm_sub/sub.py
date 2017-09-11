#!/usr/bin/env directorPython

import time

from director import lcmUtils
from robotlocomotion import viewer_draw_t
from director import consoleapp

try:
    def callback(msg, channel):
        print("channel: {}\n  name: {}\n  timestamp: {}".format(
            channel, msg.link_name[0], msg.timestamp))

    sub = lcmUtils.addSubscriber(
        'DRAKE_DRAW_FRAMES.*',
        messageClass = viewer_draw_t,
        callback = callback,
        callbackNeedsChannel = True)
    # Workaround:
    sub.setNotifyAllMessagesEnabled(True)

    app = consoleapp.ConsoleApp()
    app.start()
    # print "Spinning"
    # while True:
    #     print sub
    #     time.sleep(0.01)

except KeyboardInterrupt:
    pass
