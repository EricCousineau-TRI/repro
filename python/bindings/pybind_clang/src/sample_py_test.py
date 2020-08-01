import sample as mut


class MyVisitor:
    def __init__(self):
        print("yo")


def main():
    argv = [__file__, "src/example.cc"]
    op = mut.CommonOptionsParser(argv)
    tool = mut.ClangTool(op.getCompilations(), op.getSourcePathList())
    tool.run(MyVisitor)
    print("Done")


if __name__ == "__main__":
    main()
