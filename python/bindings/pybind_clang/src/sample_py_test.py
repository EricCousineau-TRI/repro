import sample as mut


class MyVisitor:
    def set_ci(self, ci):
        pass

    def visit_decl(self, d):
        print(type(d))
        d.dump()
        print(d.getDeclKindName())
        if isinstance(d, mut.NamedDecl):
            print(d.getQualifiedNameAsString())
        return True


def main():
    # WARNING: This is exhibiting singleton behavior. This setup is wrong
    # (using LibTooling setup, mimicing Binder). Should instead use CastXML
    # setup?
    for _ in range(1):
        argv = [__file__, "src/example.cc"]
        op = mut.CommonOptionsParser(argv)
        tool = mut.ClangTool(op.getCompilations(), op.getSourcePathList())
        visitor = MyVisitor()
        tool.run(visitor)
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
