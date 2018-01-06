import gc
import traceback

import _test_unique_ptr as m

try:
    stats = m.ConstructorStats.get(m.UniquePtrHeld)

    print("Create")
    assert stats.alive() == 0
    obj = m.UniquePtrHeld(1)
    assert stats.alive() == 1
    print(" - Destroy")
    m.unique_ptr_terminal(obj)
    assert stats.alive() == 0

    print("Create (b)")
    obj = m.UniquePtrHeld(1)
    print("Call")
    obj_ref = m.unique_ptr_pass_through(obj)
    print("Check")
    assert stats.alive() == 1
    assert obj.value() == 1
    assert obj == obj_ref
    del obj
    del obj_ref
    gc.collect()
    assert stats.alive() == 0
except:
    traceback.print_exc()
