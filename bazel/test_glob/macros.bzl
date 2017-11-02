def glob_print(patterns, exclude = []):
    files = native.glob(patterns, exclude)
    print("Patterns: {}".format(patterns))
    print("Files: {}".format(files))
    return files
