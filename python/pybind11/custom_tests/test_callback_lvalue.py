# import trace
import _test_callback_lvalue as m

# Issue #1200
def incr(obj):
    obj.value += 1

obj = m.callback_mutate_copyable_cpp_ref(incr, 10)
assert obj.value == 11  # without patch: obj.value == 10

obj = m.callback_mutate_copyable_cpp_ptr(incr, 10)
assert obj.value == 11
