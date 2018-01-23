import test._cpp_const_pybind_test_py as m

from cpp_const import to_const

obj = m.Test(10)
print(obj.value)
obj.value = 100

obj.check_const()
obj.check_mutate()

obj_const = obj.as_const()
print(obj_const.value)
obj_const.check_const()

# Try to mutate.
obj_const.check_mutate()
obj_const.value = 10
