# Check how tags propagate for `test_suite`

From docs:
https://bazel.build/reference/be/general#test_suite_args

    Only test rules that match all of the positive tags and none of the negative tags will be included in the test suite. Note that this does not mean that error checking for dependencies on tests that are filtered out is skipped; the dependencies on skipped tests still need to be legal (e.g. not blocked by visibility constraints).

Unclear what "match" means here in terms of propagating tags.

## Example


