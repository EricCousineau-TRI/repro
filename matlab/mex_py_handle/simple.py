value = 0

def simple():
    global value
    value += 1
    # Called by MATLAB, which is called by C
    print "py: Simple {}".format(value)

def call_check(f, *args):
    print f
    return f(*args)

def pass_thru(value):
    return value

class Obj(object):
    def __init__(self, func):
        self.func = func
        print "py: Store {}".format(self.func)
    def call(self, *args):
        print "py: Call {}".format(self.func)
        return self.func(*args)
