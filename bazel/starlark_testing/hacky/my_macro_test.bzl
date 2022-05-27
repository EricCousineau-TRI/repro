load(":my_macro.bzl", "add_two_numbers")

def run_my_macro_tests():
    c_expected = 3  # see success
    # c_expected = 4  # see failure
    if c_expected != add_two_numbers(1, 2):
        fail()
