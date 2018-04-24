# Purpose: See if there's a way to have a decorator know about the decorated
# object with current syntax.

def decorate(func):
    def wrap(*args, **kwargs):
        print("Wrapped")
        return func(*args, **kwargs)
    return wrap


@decorate
def some_func(x):
    print("some_func: {}".format(x))
    print(some_func)

@decorate
class SomeClass(object):
    def __init__(self, value):
        print("SomeClass: {}".format(value))
        self.value = value
        print(SomeClass)

some_func(10)
SomeClass(100)
