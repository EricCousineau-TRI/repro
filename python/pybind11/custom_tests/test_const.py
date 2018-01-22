# import trace
import _test_const as m

obj = m.Test(10)
print(obj.get_value())
obj.set_value(1000)
print(obj.get_value())
