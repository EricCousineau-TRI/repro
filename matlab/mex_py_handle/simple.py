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
