# import trace
import _test_unique_ptr as m

obj = m.UniquePtrHeld(1)
m.unique_ptr_terminal(obj)
obj = m.UniquePtrHeld(1)
obj_ref = m.unique_ptr_pass_through(obj)
