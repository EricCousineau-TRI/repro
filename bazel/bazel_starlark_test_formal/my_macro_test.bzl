load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load(":my_macro.bzl", "add_two_numbers")

def _my_macro_test_impl(ctx):
    env = unittest.begin(ctx)
    expected = 3  # good
    # expected = 4  # bad
    asserts.equals(env, expected, add_two_numbers(1, 2))
    return unittest.end(env)

my_macro_test = unittest.make(_my_macro_test_impl)

def my_macro_test_suite(name):
    unittest.suite(
        name,
        my_macro_test,
    )
