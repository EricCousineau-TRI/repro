class Base(object):
  pass

class Child(Base):
  pass

x = Base()
y = [Base, Base]
z = Child()

def print_referrers(obj):
    import gc
    import inspect
    refs = gc.get_referrers(Base)
    for ref in refs:
        print(ref)
        try:
            lines = inspect.getsourcelines(ref)
            print(lines[1:])
            print("".join(lines[0]))
        except Exception as e:
            pass
        print("---")
print_referrers(Base)
