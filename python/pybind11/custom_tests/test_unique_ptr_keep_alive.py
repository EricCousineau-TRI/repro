import sys
import weakref
# import gc

# keep_cls_list = [m.ContainerKeepAlive]
# keep_cls_list = [m.ContainerKeepAlive, m.ContainerExposeOwnership]
keep_cls_list = [m.ContainerExposeOwnership]
for i, keep_cls in enumerate(keep_cls_list):
    obj = m.UniquePtrHeld(i + 1)
    print("create")
    c_keep = keep_cls(obj)
    print("refcount: {}".format(sys.getrefcount(c_keep)))
    c_keep_wref = weakref.ref(c_keep)
    print(c_keep)
    print("refcount: {}".format(sys.getrefcount(c_keep)))

    print("del")
    print("refcount: {}".format(sys.getrefcount(c_keep)))
    del c_keep

    print("check wref")
    assert c_keep_wref() is not None
    c_keep_wref().release()
    del obj
