class A:
    def __iadd__(self, other):
        pass

    def __isub__(self, other):
        return self

a1, a2 = A(), A()

a1 -= a2
print(a1)
# <__main__.A object at 0x7f34b275aa58>

a1 += a2
print(a1)
# None
