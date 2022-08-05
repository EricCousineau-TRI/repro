ignore_globals = None
my_globals = dict()
my_locals = dict()

def my_exec(code):
    exec(code, my_globals, my_locals)

def print_vars():
    filtered_globals = {k: v for k, v in my_globals.items() if k not in ignore_globals}
    print(f"globals: {filtered_globals}\nlocals: {my_locals}")

my_exec("")
ignore_globals = set(my_globals.keys())
my_exec("b = 1")
print_vars()
my_exec("def my_func(x): return x + b")
print_vars()
my_exec("print(my_func(2))")

# see: https://stackoverflow.com/questions/28950735/closure-lost-during-callback-defined-in-exec
