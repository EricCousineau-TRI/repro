# For more info, see:
# https://stackoverflow.com/a/14132912/7829525

# To run this, run `../../repro.sh`.

# Case 1: Use absolute name (GOOD)
import my_package.my_submodule as case1
# Case 2: Relative path (GOOD)
from . import my_submodule as case2
# Case 3: Short name (BAD: Duplicated module)
import my_submodule as case3

print(f"case1.__name__: {case1.__name__}")
print(f"case2.__name__: {case2.__name__}")
print(f"case3.__name__: {case3.__name__}")

# To show Case 3 being bad, we will mutate the `x` variable.
# Case 1 and 2 will match, but Case3 will not, because it's an entirely
# separate module.
case1.x = 123

print(f"case1.x: {case1.x}")
print(f"case2.x: {case2.x}")
print(f"case3.x: {case3.x} (BAD)")
