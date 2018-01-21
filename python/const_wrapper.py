# pip install wrapt

class Const(object):
    def __init__(self, obj):
        object.__setattr__(self, '_obj', obj)

    # def __getattr__(self, name):
    #     obj = self._obj
    #     print("get: {}".format(name))
    #     return getattr(obj, name)

    def __getattribute__(self, name):
        obj = object.__getattribute__(self, '_obj')
        print("get_full: {}".format(name))
        return getattr(obj, name)

    def __setitem__(self, name, value):
        print("Wrap")
        obj = object.__getattribute__(self, '_obj')
        obj[name] = value

    def __setattr__(self, name, value):
        obj = object.__getattribute__(self, '_obj')
        print("set: {}".format(name))
        print(obj)
        return setattr(obj, name, value)

class Check(object):
    def __init__(self, value):
        self._value = value
        self._map = dict()

    def _get_value(self):
        return self._value

    def _set_value(self, value):
        self._value = value

    def do_something(self, stuff):
        print("{}: {}".format(stuff, self._value))

    def __setitem__(self, key, value):
        self._map[key] = value

    value = property(_get_value, _set_value)


c = Check(10)
c_const = Const(c)

print(c_const.value)
c_const.value = 100
c_const.do_something("yo")
print(c_const == c_const)
c_const[10] = 10
