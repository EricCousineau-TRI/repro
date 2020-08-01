import pybind_clang.sample as mut
from pybind_clang.execution import traced, reexecute_if_unbuffered


class MyVisitor:
    def set_ci(self, ci):
        pass

    def visit_decl(self, d):
        d.dump()
        print(d.getDeclKindName(), type(d))
        if isinstance(d, mut.NamedDecl):
            print(d.getQualifiedNameAsString())
        print("---")
        return True


@traced
def main():
    # WARNING: This is exhibiting singleton behavior. This setup is wrong
    # (using LibTooling setup, mimicing Binder). Should instead use CastXML
    # setup?
    for _ in range(1):
        argv = [__file__, "pybind_clang/test/example.cc"]
        op = mut.CommonOptionsParser(argv)
        tool = mut.ClangTool(op.getCompilations(), op.getSourcePathList())
        visitor = MyVisitor()
        tool.run(visitor)
    print("Done")


if __name__ == "__main__":
    reexecute_if_unbuffered()
    main()
