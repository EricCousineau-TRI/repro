import _cpp_inherit as m

assert m.create_base().stuff() == 1
assert m.create_child().stuff() == 10

print("Done")
