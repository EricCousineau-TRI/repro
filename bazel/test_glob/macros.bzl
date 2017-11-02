
def glob_print(patterns, exclude = []):
    # Test dictionary merging.
    a = {"a": "old", "b": "old"}
    b = {"a": "new", "c": "new"}
    print(a + b)
    print(b + a)

    # Test struct merging.
    a = struct(a="old", b="old")
    b = struct(a="new", c="new")
    # Invalid:
    # print(dir(a))
    # print(dict(a))
    # print(a.merge(b))
    # print(b + a)

    files = native.glob(patterns, exclude)
    print("Patterns: {}".format(patterns))
    print("Files: {}".format(files))
    return files
