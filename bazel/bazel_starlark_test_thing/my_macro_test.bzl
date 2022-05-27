load(":my_macro.bzl", "add_two_numbers")

def run_my_macro_tests():
    a = 1
    b = 2
    # c_expected = 3  # see success
    c_expected = 4  # see failure
    c_actual = add_two_numbers(a, b)
    if c_expected != c_actual:
        fail()
