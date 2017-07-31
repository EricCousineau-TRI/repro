print_value = @py.py_module.print_value

print_value(2)
print_value('Hello world')

f = @(x) x + 1
print_value(f)
