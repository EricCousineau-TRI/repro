import sys
import weakref
# import gc

obj_stats = ConstructorStats.get(m.UniquePtrHeld)

# Now try with keep-alive containers.
for i, keep_cls in enumerate([m.ContainerKeepAlive, m.ContainerExposeOwnership]):
    c_keep_stats = ConstructorStats.get(keep_cls)
    obj = m.UniquePtrHeld(i + 1)
    print("create")
    c_keep = keep_cls(obj)
    c_keep_wref = weakref.ref(c_keep)
    print("refcount: {}".format(sys.getrefcount(c_keep)))
    assert obj_stats.alive() == 1
    assert c_keep_stats.alive() == 1
    print("del")
    del c_keep
    # print("gc")
    # gc.collect()
    print("check wref")
    # Everything should have stayed alive.
    assert c_keep_wref() is not None
    assert c_keep_stats.alive() == 1
    assert obj_stats.alive() == 1
    # Now release the object. This should have released the container as a patient.
    c_keep_wref().release()
    # gc.collect()
    assert obj_stats.alive() == 1
    assert c_keep_stats.alive() == 0
    del obj
    # gc.collect()
