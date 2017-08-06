value = 0

def simple():
    global value
    value += 1
    # Called by MATLAB, which is called by C
    print "py: Simple X {}".format(value)
    if value == 3:
        raise Exception("Bad 3")

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

class Store(object):
    def __init__(self, value):
        self.value = value
