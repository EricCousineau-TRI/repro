# import trace
import _test_keep_alive_transient as m

stats = m.ConstructorStats.get(m.UniquePtrHeld)

obj = m.UniquePtrHeld(1)
assert stats.alive() == 1

c = m.Container()
c.add(obj)

d = m.Container()
c.transfer(d)

del d
del c
assert stats.alive() == 1, "Should still be alive!"
print(obj)
