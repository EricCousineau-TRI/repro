from enum import Enum
from inspect import isclass, isfunction

class ExampleClass:
    class NestedClass:
        pass

    def example_func(self):
        pass

class ExampleEnum(Enum):
    class NestedClass:
        pass

    def example_func(self):
        pass

def main():
    assert isfunction(ExampleClass.example_func)
    assert isclass(ExampleClass.NestedClass)

    assert isfunction(ExampleEnum.example_func)
    # Different! `enum.py` seems to interpret this as an enum entry, rather
    # than just a nested class.
    assert not isclass(ExampleEnum.NestedClass)
    assert isclass(ExampleEnum.NestedClass.value)
    print("Done")

if __name__ == "__main__":
    main()
