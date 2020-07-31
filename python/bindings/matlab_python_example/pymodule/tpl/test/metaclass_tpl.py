is_global = True

class TemplateMeta(type):
    def __new__(cls, name, parents, dct):
        print("CALLED")
        # print(cls)
        # print(type(cls))
        f = dct["some_func"]
        # print(dir(f))
        print(type(f.func_closure))
        print(dir(f.func_closure))
        # f.func_closure.update(var=10000)
        # return type.__new__(cls, name, parents, dct)
        return type(name, parents, dct)


class Test(object):
    __metaclass__ = TemplateMeta
    var = 10

    def __init__(self):
        pass

    def some_func(self):
        print(Test.var)


print("START")
t = Test()
print(type(Test))
print(t.some_func())
