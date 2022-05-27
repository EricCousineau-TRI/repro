# Different ways to test simpel starlark macros

Comparing hacky approach against more formal approach.

Look at `my_macro_tests.bzl` in each workspace, and play with changing it to
fail.

## formal

Using https://bazel.build/rules/testing

```
cd formal
bazel test ...
```

Pro:
- More flexible
- Actual tests
- Library covers way more than just macros
Con:
- Hefty deps
- Complex workspace deps (multi-pass `load()` in `WORKSPACE`)
- Failure (as of v1.2.0) doesn't actually give good code trace?

## hacky

```
cd hacky
bazel build ...
```

Pro:
- Not many deps
- Failure happens at exact line
Con:
- Just for macros
- Not very expressive
- [Loading phase](https://docs.bazel.build/versions/main/skylark/concepts.html#evaluation-model) error
