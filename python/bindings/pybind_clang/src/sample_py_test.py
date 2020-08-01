import sample as mut


class MyVisitor:
    def visit(self, decl):
        print(decl.getQualifiedNameAsString())
        return True


def main():
    argv = [__file__, "src/example.cc"]
    op = mut.CommonOptionsParser(argv)
    tool = mut.ClangTool(op.getCompilations(), op.getSourcePathList())

    visitor = MyVisitor()
    tool.run(visitor.visit)
    print("Done")


if __name__ == "__main__":
    main()


"""
Output:

Error while trying to load a compilation database:
Could not auto-detect compilation database for file "src/example.cc"
No compilation database found in .../sample_py_test.runfiles/pybind_clang/src or any parent directory
fixed-compilation-database: Error while opening fixed database: No such file or directory
json-compilation-database: Error while opening JSON database: No such file or directory
Running without flags.
n::m::C
Done
"""
