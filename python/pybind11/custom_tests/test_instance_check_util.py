from functools import partial


class FactoryMeta(object):
    def __init__(self, f, types):
        self._f = f
        self._types = types

    def __call__(self, *args, **kwargs):
        return self._f(*args, **kwargs)

    def __instancecheck__(self, other):
        return isinstance(other, self._types)


def factory_functor(types):
    return partial(FactoryMeta, types=types)


def test_main():

    @factory_functor((int, str))
    def TestFactory(x):
        return x

    print(TestFactory("Hello"))
    assert isinstance(TestFactory(1), TestFactory)
    assert isinstance(TestFactory("hello"), TestFactory)
    assert not isinstance(TestFactory([1, 2, 3]), TestFactory)


if __name__ == "__main__":
    test_main()
