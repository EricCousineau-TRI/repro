# Check how tags propagate for `test_suite`

From docs:
https://bazel.build/reference/be/general#test_suite_args

    Only test rules that match all of the positive tags and none of the negative tags will be included in the test suite. Note that this does not mean that error checking for dependencies on tests that are filtered out is skipped; the dependencies on skipped tests still need to be legal (e.g. not blocked by visibility constraints).

Unclear what "match" means here in terms of propagating tags.

## Example

```
$ ./repro.sh
+ bazel test //tests/...
//tests:a_test FAILED
//tests:b_test FAILED
//tests:c_test FAILED

+ bazel test //tests/... --test_tag_filters=a
//tests:a_test FAILED

+ bazel test //tests/... --test_tag_filters=-b
//tests:a_test FAILED
//tests:c_test FAILED

+ bazel test //suites:no_tags
//tests:a_test FAILED
//tests:b_test FAILED
//tests:c_test FAILED

+ bazel test //suites:no_tags --test_tag_filters=a
//tests:a_test FAILED

+ bazel test //suites:no_tags --test_tag_filters=-b
//tests:a_test FAILED
//tests:c_test FAILED

+ bazel test //suites:a_tag
//tests:a_test FAILED

+ bazel test //suites:minus_b_tag
//tests:a_test FAILED
//tests:c_test FAILED
```
