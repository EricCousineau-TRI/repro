import gc
import _test_keep_alive_transient as m

stats = m.ConstructorStats.get(m.UniquePtrHeld)

obj = m.UniquePtrHeld(1)
assert stats.alive() == 1

c = m.Container("Container: c")
print(c)
c.add(obj)

d = m.Container("Container: d")
print(d)
m.sentinel()
c.transfer(d)

del c
gc.collect()
del d
gc.collect()
assert stats.alive() == 1, "Should still be alive!"
print(obj)
